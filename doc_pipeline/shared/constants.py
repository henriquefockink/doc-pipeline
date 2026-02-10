"""Constants for queue-based processing."""

# VLM backend identifier used in extraction results
VLM_BACKEND_NAME = "paneas_v3"


class QueueName:
    """Redis queue names."""

    DOCUMENTS = "queue:doc:documents"  # Main processing queue (classify/extract/process)
    OCR = "queue:doc:ocr"  # OCR processing queue (separate worker)
    INFERENCE = "queue:doc:inference"  # VLM inference queue (inference server)
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


def inference_reply_channel(inference_id: str) -> str:
    """Return the Pub/Sub channel name for an inference reply."""
    return f"inference:reply:{inference_id}"


INFERENCE_REPLY_TTL = 120  # 2 minutes - TTL for cached inference replies


def inference_reply_key(inference_id: str) -> str:
    """Return the Redis key for caching an inference reply (polling mode)."""
    return f"inference:result:{inference_id}"
