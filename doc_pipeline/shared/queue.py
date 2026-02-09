"""Redis queue operations for job processing."""

from __future__ import annotations

import redis.asyncio as redis

from doc_pipeline.config import get_settings
from doc_pipeline.observability import get_logger

from .constants import (
    PROGRESS_TTL,
    RESULT_CACHE_TTL,
    QueueName,
    progress_key,
    result_cache_key,
    result_channel,
)
from .job_context import JobContext

logger = get_logger("queue")


class QueueService:
    """Service for Redis queue operations."""

    def __init__(self, redis_url: str | None = None, queue_name: str | None = None):
        """Initialize queue service.

        Args:
            redis_url: Redis connection URL (default from settings)
            queue_name: Queue name to use (default DOCUMENTS)
        """
        settings = get_settings()
        self._redis_url = redis_url or settings.redis_url
        self._queue_name = queue_name or QueueName.DOCUMENTS
        self._redis: redis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._redis is None:
            settings = get_settings()
            pool = redis.BlockingConnectionPool.from_url(
                self._redis_url,
                max_connections=settings.redis_max_connections,
                timeout=5,  # wait up to 5s for a free connection
                decode_responses=True,
            )
            self._redis = redis.Redis(connection_pool=pool)
            logger.info("redis_connected", url=self._redis_url.split("@")[-1])

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("redis_disconnected")

    @property
    def redis(self) -> redis.Redis:
        """Get Redis client (must be connected first)."""
        if self._redis is None:
            raise RuntimeError("Queue service not connected. Call connect() first.")
        return self._redis

    async def enqueue(self, job: JobContext, queue_name: str | None = None) -> None:
        """
        Add a job to the processing queue.

        Uses LPUSH so workers can BRPOP (FIFO order).

        Args:
            job: JobContext to enqueue
            queue_name: Override queue name (default uses instance queue)
        """
        settings = get_settings()
        target_queue = queue_name or self._queue_name

        # Check queue size limit
        queue_size = await self.get_queue_depth(target_queue)
        if queue_size >= settings.max_queue_size:
            raise QueueFullError(f"Queue is full ({queue_size}/{settings.max_queue_size})")

        await self.redis.lpush(target_queue, job.to_json())
        logger.info(
            "job_enqueued",
            request_id=job.request_id,
            operation=job.operation,
            delivery_mode=job.delivery_mode,
            queue=target_queue,
            queue_depth=queue_size + 1,
        )

    async def dequeue(self, timeout: float = 5.0) -> JobContext | None:
        """
        Get the next job from the queue.

        Uses BRPOP with timeout for blocking pop (FIFO order).
        Returns None if timeout expires with no job.
        """
        result = await self.redis.brpop(self._queue_name, timeout=timeout)
        if result is None:
            return None

        _, job_json = result
        job = JobContext.from_json(job_json)
        logger.debug(
            "job_dequeued",
            request_id=job.request_id,
            operation=job.operation,
            queue=self._queue_name,
        )
        return job

    async def move_to_dlq(self, job: JobContext, error: str) -> None:
        """Move a failed job to the dead letter queue."""
        job.error = error
        await self.redis.lpush(QueueName.DLQ, job.to_json())
        logger.warning(
            "job_moved_to_dlq",
            request_id=job.request_id,
            error=error,
        )

    async def get_queue_depth(self, queue_name: str | None = None) -> int:
        """Get the number of jobs waiting in the queue."""
        target_queue = queue_name or self._queue_name
        return await self.redis.llen(target_queue)

    async def get_dlq_depth(self) -> int:
        """Get the number of jobs in the dead letter queue."""
        return await self.redis.llen(QueueName.DLQ)

    async def set_progress(self, request_id: str, step: str) -> None:
        """Update the progress of a job (for polling)."""
        key = progress_key(request_id)
        await self.redis.setex(key, PROGRESS_TTL, step)
        logger.debug("progress_updated", request_id=request_id, step=step)

    async def get_progress(self, request_id: str) -> str | None:
        """Get the current progress of a job."""
        key = progress_key(request_id)
        return await self.redis.get(key)

    async def cache_result(self, request_id: str, result: str) -> None:
        """Cache job result for polling."""
        key = result_cache_key(request_id)
        await self.redis.setex(key, RESULT_CACHE_TTL, result)
        logger.debug("result_cached", request_id=request_id)

    async def get_cached_result(self, request_id: str) -> str | None:
        """Get cached result for a job."""
        key = result_cache_key(request_id)
        return await self.redis.get(key)

    async def publish_result(self, request_id: str, result: str) -> None:
        """Publish result to Pub/Sub channel for sync mode."""
        channel = result_channel(request_id)
        await self.redis.publish(channel, result)
        logger.debug("result_published", request_id=request_id, channel=channel)

    async def subscribe_result(self, request_id: str, timeout: float) -> str | None:
        """
        Subscribe to result channel and wait for result (sync mode).

        Returns the result JSON or None if timeout expires.
        """
        channel = result_channel(request_id)
        pubsub = self.redis.pubsub()

        try:
            await pubsub.subscribe(channel)
            logger.debug("subscribed_to_result", request_id=request_id, channel=channel)

            # Wait for message with timeout
            async for message in pubsub.listen():
                if message["type"] == "message":
                    return message["data"]
                # Check timeout in listen loop would require async timeout
                # For now, rely on external timeout wrapper
            return None
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()


class QueueFullError(Exception):
    """Raised when the queue is at capacity."""

    pass


# Singleton instance
_queue_service: QueueService | None = None


def get_queue_service() -> QueueService:
    """Get the singleton queue service instance."""
    global _queue_service
    if _queue_service is None:
        _queue_service = QueueService()
    return _queue_service


async def reset_queue_service() -> None:
    """Reset the queue service (for testing)."""
    global _queue_service
    if _queue_service:
        await _queue_service.close()
    _queue_service = None
