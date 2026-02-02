"""Constants for queue-based processing."""


class QueueName:
    """Redis queue names."""

    DOCUMENTS = "queue:doc:documents"  # Main processing queue
    DLQ = "queue:doc:dlq"  # Dead letter queue for failed jobs


# TTLs and timeouts
RESULT_CACHE_TTL = 600  # 10 minutes - cache results for polling
PROGRESS_TTL = 600  # 10 minutes - progress tracking TTL
WEBHOOK_TIMEOUT = 20  # seconds - webhook delivery timeout
WEBHOOK_MAX_RETRIES = 3  # max webhook delivery attempts


# Redis key patterns
def result_channel(request_id: str) -> str:
    """Return the Pub/Sub channel name for a request result."""
    return f"results:{request_id}"


def progress_key(request_id: str) -> str:
    """Return the key for tracking job progress."""
    return f"progress:{request_id}"


def result_cache_key(request_id: str) -> str:
    """Return the key for caching job results."""
    return f"result:{request_id}"
