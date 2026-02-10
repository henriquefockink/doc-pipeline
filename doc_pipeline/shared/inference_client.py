"""Client for sending inference requests via Redis."""

from __future__ import annotations

import asyncio
import json
import uuid

from doc_pipeline.observability import get_logger

from .constants import QueueName, inference_reply_key
from .queue import QueueService

logger = get_logger("inference_client")


class InferenceError(Exception):
    """Raised when the inference server returns an error."""

    pass


class InferenceTimeoutError(InferenceError):
    """Raised when an inference request times out."""

    pass


class InferenceClient:
    """Client that sends inference requests via Redis and waits for replies."""

    def __init__(self, queue_service: QueueService, timeout: float = 30.0):
        self._queue = queue_service
        self._timeout = timeout

    async def request(
        self,
        operation: str,
        image_path: str,
        request_id: str,
        document_type: str | None = None,
        backend: str | None = None,
        auto_rotate: bool = True,
        max_pages: int = 10,
        extract: bool = True,
        min_confidence: float | None = None,
    ) -> dict:
        """
        Send inference request and wait for reply via Redis polling.

        Args:
            operation: "classify", "extract", "process", or "ocr"
            image_path: Path to image/PDF file on shared volume
            request_id: Job request ID for logging
            document_type: Document type (for extract operation)
            backend: Extraction backend ("vlm", "ocr", "hybrid")
            auto_rotate: Whether to auto-rotate image
            max_pages: Max pages for OCR/PDF
            extract: Whether to extract data (for process operation)
            min_confidence: Min classification confidence (for process)

        Returns:
            Reply dict with "success", "result", "error", etc.
        """
        inference_id = str(uuid.uuid4())
        reply_key = inference_reply_key(inference_id)

        payload = {
            "inference_id": inference_id,
            "request_id": request_id,
            "operation": operation,
            "image_path": image_path,
            "document_type": document_type,
            "backend": backend,
            "auto_rotate": auto_rotate,
            "max_pages": max_pages,
            "extract": extract,
            "min_confidence": min_confidence,
        }
        await self._queue.redis.lpush(QueueName.INFERENCE, json.dumps(payload))
        logger.debug(
            "inference_request_sent",
            inference_id=inference_id,
            request_id=request_id,
            operation=operation,
            document_type=document_type,
        )

        # Poll for reply
        poll_interval = 0.1  # 100ms between polls
        end_time = asyncio.get_running_loop().time() + self._timeout

        while True:
            cached = await self._queue.redis.get(reply_key)
            if cached:
                await self._queue.redis.delete(reply_key)
                reply = json.loads(cached)
                if not reply["success"]:
                    raise InferenceError(reply["error"])
                logger.debug(
                    "inference_reply_received",
                    inference_id=inference_id,
                    operation=operation,
                    inference_time_ms=reply.get("inference_time_ms"),
                )
                return reply

            remaining = end_time - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise InferenceTimeoutError(f"Inference timeout after {self._timeout}s")

            await asyncio.sleep(min(poll_interval, remaining))

    async def extract(self, image_path: str, document_type: str, request_id: str) -> dict:
        """Backward-compatible wrapper: send VLM extract request."""
        return await self.request(
            operation="extract",
            image_path=image_path,
            request_id=request_id,
            document_type=document_type,
        )
