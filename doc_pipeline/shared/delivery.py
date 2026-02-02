"""Result delivery service for sync and webhook modes."""

from __future__ import annotations

import asyncio
import json

import httpx

from doc_pipeline.config import get_settings
from doc_pipeline.observability import get_logger, get_metrics

from .constants import WEBHOOK_MAX_RETRIES, WEBHOOK_TIMEOUT
from .job_context import JobContext
from .queue import QueueService, get_queue_service

logger = get_logger("delivery")


class DeliveryService:
    """Service for delivering job results via sync or webhook."""

    def __init__(self, queue_service: QueueService | None = None):
        """Initialize delivery service."""
        self._queue = queue_service or get_queue_service()
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for webhook delivery."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(WEBHOOK_TIMEOUT),
                follow_redirects=True,
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def deliver(self, job: JobContext) -> bool:
        """
        Deliver job result based on delivery mode.

        Args:
            job: Completed job with result or error

        Returns:
            True if delivery was successful
        """
        if job.delivery_mode == "sync":
            return await self._deliver_sync(job)
        elif job.delivery_mode == "webhook":
            return await self._deliver_webhook(job)
        else:
            logger.error("unknown_delivery_mode", mode=job.delivery_mode, request_id=job.request_id)
            return False

    async def _deliver_sync(self, job: JobContext) -> bool:
        """Deliver result via Redis Pub/Sub for sync mode."""
        try:
            result_payload = self._build_result_payload(job)
            result_json = json.dumps(result_payload)

            # Publish to channel
            await self._queue.publish_result(job.request_id, result_json)

            # Also cache for potential polling
            await self._queue.cache_result(job.request_id, result_json)

            logger.info(
                "sync_delivery_success",
                request_id=job.request_id,
                has_error=job.error is not None,
            )
            return True

        except Exception as e:
            logger.error(
                "sync_delivery_error",
                request_id=job.request_id,
                error=str(e),
            )
            return False

    async def _deliver_webhook(self, job: JobContext) -> bool:
        """Deliver result via HTTP POST to webhook URL."""
        if not job.webhook_url:
            logger.error("webhook_url_missing", request_id=job.request_id)
            return False

        settings = get_settings()
        metrics = get_metrics()
        max_retries = settings.webhook_max_retries
        result_payload = self._build_result_payload(job)

        client = await self._get_http_client()

        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post(
                    job.webhook_url,
                    json=result_payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Request-ID": job.request_id,
                        "X-Correlation-ID": job.correlation_id or "",
                    },
                )

                if response.is_success:
                    metrics.webhook_deliveries.labels(status="success").inc()
                    logger.info(
                        "webhook_delivery_success",
                        request_id=job.request_id,
                        webhook_url=job.webhook_url,
                        status_code=response.status_code,
                        attempt=attempt,
                    )
                    return True

                # Non-success status code
                logger.warning(
                    "webhook_delivery_failed",
                    request_id=job.request_id,
                    webhook_url=job.webhook_url,
                    status_code=response.status_code,
                    attempt=attempt,
                    max_retries=max_retries,
                )

            except httpx.TimeoutException:
                logger.warning(
                    "webhook_delivery_timeout",
                    request_id=job.request_id,
                    webhook_url=job.webhook_url,
                    attempt=attempt,
                    max_retries=max_retries,
                )

            except httpx.RequestError as e:
                logger.warning(
                    "webhook_delivery_error",
                    request_id=job.request_id,
                    webhook_url=job.webhook_url,
                    error=str(e),
                    attempt=attempt,
                    max_retries=max_retries,
                )

            # Exponential backoff before retry
            if attempt < max_retries:
                await asyncio.sleep(2**attempt)

        # All retries exhausted
        metrics.webhook_deliveries.labels(status="failed").inc()
        logger.error(
            "webhook_delivery_exhausted",
            request_id=job.request_id,
            webhook_url=job.webhook_url,
            max_retries=max_retries,
        )
        return False

    def _build_result_payload(self, job: JobContext) -> dict:
        """Build the result payload for delivery."""
        payload = {
            "request_id": job.request_id,
            "operation": job.operation,
            "status": "error" if job.error else "completed",
            "correlation_id": job.correlation_id,
            "enqueued_at": job.enqueued_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
        }

        if job.error:
            payload["error"] = job.error
        else:
            payload["result"] = job.result

        # Add timing info
        if job.queue_wait_time_seconds is not None:
            payload["queue_wait_ms"] = int(job.queue_wait_time_seconds * 1000)
        if job.processing_time_seconds is not None:
            payload["processing_time_ms"] = int(job.processing_time_seconds * 1000)
        if job.total_time_seconds is not None:
            payload["total_time_ms"] = int(job.total_time_seconds * 1000)

        return payload


# Singleton instance
_delivery_service: DeliveryService | None = None


def get_delivery_service() -> DeliveryService:
    """Get the singleton delivery service instance."""
    global _delivery_service
    if _delivery_service is None:
        _delivery_service = DeliveryService()
    return _delivery_service


async def reset_delivery_service() -> None:
    """Reset the delivery service (for testing)."""
    global _delivery_service
    if _delivery_service:
        await _delivery_service.close()
    _delivery_service = None
