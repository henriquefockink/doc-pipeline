"""Client for sending VLM inference requests via Redis."""

from __future__ import annotations

import asyncio
import json
import uuid

from doc_pipeline.observability import get_logger

from .constants import QueueName, inference_reply_channel
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
        Send inference request and wait for reply via Pub/Sub.

        1. Generate inference_id
        2. Subscribe to inference:reply:{inference_id} BEFORE enqueue (avoid race)
        3. LPUSH request to queue:doc:inference
        4. Wait for reply with timeout
        5. Return result dict
        """
        inference_id = str(uuid.uuid4())
        channel = inference_reply_channel(inference_id)

        pubsub = self._queue.redis.pubsub()
        await pubsub.subscribe(channel)

        try:
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

            # Wait for reply (same pattern as subscribe_result)
            async with asyncio.timeout(self._timeout):
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        reply = json.loads(message["data"])
                        if not reply["success"]:
                            raise InferenceError(reply["error"])
                        logger.debug(
                            "inference_reply_received",
                            inference_id=inference_id,
                            inference_time_ms=reply.get("inference_time_ms"),
                        )
                        return reply

            raise InferenceTimeoutError(f"Inference timeout after {self._timeout}s")
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
