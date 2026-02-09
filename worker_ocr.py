#!/usr/bin/env python3
"""
OCR Worker - Processes OCR jobs from Redis queue.

Thin queue consumer — all GPU work (EasyOCR, orientation correction, PDF
conversion) is delegated to the centralized inference server via Redis.
"""

import asyncio
import contextlib
import signal
import time
from pathlib import Path

import sentry_sdk

from doc_pipeline.config import get_settings
from doc_pipeline.observability import get_logger, get_metrics, setup_logging
from doc_pipeline.shared import DeliveryService, InferenceClient, JobContext, QueueService
from doc_pipeline.shared.constants import QueueName

# Setup
settings = get_settings()
setup_logging(json_format=settings.log_json, log_level=settings.log_level)
logger = get_logger("worker-ocr")
metrics = get_metrics()

# Sentry / GlitchTip
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        server_name="worker-ocr",
    )


class OCRWorker:
    """Stateless OCR worker that delegates to inference server."""

    def __init__(self):
        self.settings = get_settings()
        self.queue = QueueService(queue_name=QueueName.OCR)
        self.delivery = DeliveryService(queue_service=self.queue)
        self._inference_client: InferenceClient | None = None
        self._running = False

    async def run(self):
        """Main worker loop."""
        self._running = True

        await self.queue.connect()

        # Create inference client (shares queue's Redis connection)
        self._inference_client = InferenceClient(
            queue_service=self.queue,
            timeout=self.settings.inference_timeout_seconds,
        )

        logger.info("worker_ocr_started")

        while self._running:
            try:
                job = await self.queue.dequeue(timeout=5.0)
                if job is None:
                    continue

                await self._process_job(job)

            except asyncio.CancelledError:
                logger.info("worker_cancelled")
                break
            except Exception as e:
                logger.exception("worker_error", error=str(e))
                await asyncio.sleep(1)

        # Cleanup
        await self.queue.close()
        await self.delivery.close()
        logger.info("worker_ocr_stopped")

    async def _process_job(self, job: JobContext):
        """Process a single OCR job by delegating to inference server."""
        start_time = time.perf_counter()
        queue_wait_ms = job.queue_wait_ms

        logger.info(
            "job_processing_start",
            request_id=job.request_id,
            operation=job.operation,
            delivery_mode=job.delivery_mode,
        )

        try:
            max_pages = job.extra_params.get("max_pages", 10) if job.extra_params else 10

            if not Path(job.image_path).exists():
                raise FileNotFoundError(f"File not found: {job.image_path}")

            # Delegate to inference server
            reply = await self._inference_client.request(
                operation="ocr",
                image_path=job.image_path,
                request_id=job.request_id,
                max_pages=max_pages,
                auto_rotate=(job.extra_params or {}).get("auto_rotate", True),
            )

            result_dict = reply["result"]

            # Add processing time
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            result_dict["processing_time_ms"] = processing_time_ms

            job.result = result_dict
            job.error = None

            logger.info(
                "job_processing_complete",
                request_id=job.request_id,
                operation=job.operation,
                processing_time_ms=processing_time_ms,
                queue_wait_ms=queue_wait_ms,
                total_pages=result_dict.get("total_pages"),
            )

            metrics.jobs_processed.labels(
                operation="ocr",
                status="success",
                delivery_mode=job.delivery_mode,
            ).inc()

        except Exception as e:
            sentry_sdk.capture_exception(e)
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            logger.exception(
                "job_processing_error",
                request_id=job.request_id,
                error=str(e),
            )

            job.result = None
            job.error = str(e)

            metrics.jobs_processed.labels(
                operation="ocr",
                status="error",
                delivery_mode=job.delivery_mode,
            ).inc()

        # Deliver result
        await self.delivery.deliver(job)

        # Cleanup temp file
        with contextlib.suppress(Exception):
            Path(job.image_path).unlink(missing_ok=True)

    def stop(self):
        """Signal worker to stop."""
        self._running = False


# Health check server
async def health_server(worker: OCRWorker):
    """Simple HTTP health check server."""
    from aiohttp import web

    async def health_handler(request):
        return web.json_response(
            {
                "status": "healthy",
                "worker": "ocr",
                "running": worker._running,
            }
        )

    async def metrics_handler(request):
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return web.Response(
            body=generate_latest(),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/metrics", metrics_handler)

    runner = web.AppRunner(app)
    await runner.setup()

    port = settings.worker_ocr_health_port
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("health_server_started", port=port)

    return runner


async def main():
    """Main entry point."""
    worker = OCRWorker()

    def signal_handler(sig, frame):
        logger.info("shutdown_signal_received", signal=sig)
        worker.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    health_runner = await health_server(worker)

    try:
        await worker.run()
    finally:
        await health_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
