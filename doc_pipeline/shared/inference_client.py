"""Client for sending VLM inference requests via Redis."""

from __future__ import annotations

import asyncio
import json
import uuid

from doc_pipeline.observability import get_logger

from .constants import INFERENCE_REPLY_TTL, QueueName, inference_reply_key
from .queue import QueueService

logger = get_logger("inference_client")


class InferenceError(Exception):
    """Raised when the inference server returns an error."""

    pass


class InferenceTimeoutError(InferenceError):
    """Raised when an inference request times out."""

    pass


class InferenceClient:
    """Client that sends VLM inference requests via Redis and waits for replies."""

    def __init__(self, queue_service: QueueService, timeout: float = 30.0):
        self._queue = queue_service
        self._timeout = timeout

    async def extract(self, image_path: str, document_type: str, request_id: str) -> dict:
        """
        Send inference request and wait for reply via Redis polling.

        1. Generate inference_id
        2. LPUSH request to queue:doc:inference
        3. Poll inference:result:{inference_id} until reply arrives
        4. Return result dict
        """
        inference_id = str(uuid.uuid4())
        reply_key = inference_reply_key(inference_id)

        request = json.dumps(
            {
                "inference_id": inference_id,
                "request_id": request_id,
                "document_type": document_type,
                "image_path": image_path,
            }
        )
        await self._queue.redis.lpush(QueueName.INFERENCE, request)
        logger.debug(
            "inference_request_sent",
            inference_id=inference_id,
            request_id=request_id,
            document_type=document_type,
        )

        # Poll for reply (same pattern as API wait_for_result)
        poll_interval = 0.1  # 100ms between polls (inference is fast)
        end_time = asyncio.get_event_loop().time() + self._timeout

        while True:
            cached = await self._queue.redis.get(reply_key)
            if cached:
                # Clean up the key
                await self._queue.redis.delete(reply_key)
                reply = json.loads(cached)
                if not reply["success"]:
                    raise InferenceError(reply["error"])
                logger.debug(
                    "inference_reply_received",
                    inference_id=inference_id,
                    inference_time_ms=reply.get("inference_time_ms"),
                )
                return reply

            remaining = end_time - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise InferenceTimeoutError(f"Inference timeout after {self._timeout}s")

            await asyncio.sleep(min(poll_interval, remaining))
