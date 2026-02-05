"""
Worker metrics aggregation via Redis.

Workers push their metrics to Redis periodically.
API reads and aggregates them in /metrics endpoint.
"""

import asyncio
import json
import time
from typing import Any

from prometheus_client import REGISTRY
from prometheus_client.metrics import MetricWrapperBase

# Redis key prefix for worker metrics
WORKER_METRICS_PREFIX = "metrics:worker:"
# TTL for worker metrics (seconds) - metrics expire if worker stops pushing
WORKER_METRICS_TTL = 60


def serialize_metrics() -> dict[str, Any]:
    """
    Serialize current process metrics to a JSON-serializable dict.

    Returns dict with structure:
    {
        "timestamp": 1234567890.123,
        "metrics": {
            "doc_pipeline_jobs_processed_total": {
                "type": "counter",
                "values": [
                    {"labels": {"operation": "process", "status": "success", "delivery_mode": "sync"}, "value": 100}
                ]
            },
            "doc_pipeline_worker_processing_seconds": {
                "type": "histogram",
                "values": [
                    {"labels": {"operation": "process"}, "buckets": {...}, "count": 100, "sum": 500.5}
                ]
            }
        }
    }
    """
    result = {
        "timestamp": time.time(),
        "metrics": {},
    }

    # Only export doc_pipeline worker metrics (not internal prometheus metrics)
    # NOTE: prometheus_client strips "_total" suffix from Counter metric.name internally
    # So "doc_pipeline_jobs_processed_total" becomes "doc_pipeline_jobs_processed" in metric.name
    worker_metric_names = [
        "doc_pipeline_jobs_processed",  # Counter (internally stripped _total)
        "doc_pipeline_worker_processing_seconds",
        "doc_pipeline_queue_wait_seconds",
        "doc_pipeline_documents_processed",  # Counter (internally stripped _total)
        "doc_pipeline_classification_confidence",
        "doc_pipeline_queue_depth",
        "doc_pipeline_webhook_deliveries",  # Counter (internally stripped _total)
    ]

    for metric in REGISTRY.collect():
        if metric.name not in worker_metric_names:
            continue

        metric_data = {
            "type": metric.type,
            "values": [],
        }

        for sample in metric.samples:
            # Skip _created samples
            if sample.name.endswith("_created"):
                continue

            sample_data = {
                "name": sample.name,
                "labels": dict(sample.labels),
                "value": sample.value,
            }
            metric_data["values"].append(sample_data)

        if metric_data["values"]:
            result["metrics"][metric.name] = metric_data

    return result


def format_aggregated_metrics(workers_data: dict[str, dict]) -> str:
    """
    Format aggregated worker metrics as Prometheus text format.

    Args:
        workers_data: Dict of {worker_id: serialized_metrics}

    Returns:
        Prometheus text format string
    """
    if not workers_data:
        return ""

    lines = []

    # Collect all metrics across workers
    all_metrics: dict[str, dict] = {}

    for worker_id, data in workers_data.items():
        if not data or "metrics" not in data:
            continue

        for metric_name, metric_data in data["metrics"].items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = {
                    "type": metric_data["type"],
                    "samples": [],
                }

            for sample in metric_data["values"]:
                # Add worker_id label
                labels = sample["labels"].copy()
                labels["worker_id"] = worker_id
                all_metrics[metric_name]["samples"].append({
                    "name": sample["name"],
                    "labels": labels,
                    "value": sample["value"],
                })

    # Format as Prometheus text
    for metric_name, metric_data in sorted(all_metrics.items()):
        # HELP and TYPE lines
        lines.append(f"# HELP {metric_name} Aggregated from workers")
        lines.append(f"# TYPE {metric_name} {metric_data['type']}")

        # Sample lines
        for sample in metric_data["samples"]:
            labels_str = ",".join(
                f'{k}="{v}"' for k, v in sorted(sample["labels"].items())
            )
            lines.append(f"{sample['name']}{{{labels_str}}} {sample['value']}")

        lines.append("")  # Blank line between metrics

    return "\n".join(lines)


class WorkerMetricsPusher:
    """
    Background task that periodically pushes metrics to Redis.
    """

    def __init__(self, redis_client, worker_id: str, interval: float = 10.0):
        """
        Args:
            redis_client: Async Redis client
            worker_id: Unique identifier for this worker (e.g., "docid-1")
            interval: Push interval in seconds
        """
        self.redis = redis_client
        self.worker_id = worker_id
        self.interval = interval
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        """Start the background push task."""
        self._running = True
        self._task = asyncio.create_task(self._push_loop())

    async def stop(self):
        """Stop the background push task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Clean up our key from Redis
        key = f"{WORKER_METRICS_PREFIX}{self.worker_id}"
        try:
            await self.redis.delete(key)
        except Exception:
            pass

    async def _push_loop(self):
        """Main loop that pushes metrics periodically."""
        while self._running:
            try:
                await self._push_metrics()
            except Exception as e:
                # Log but don't crash
                import logging
                logging.getLogger("worker_metrics").warning(
                    f"Failed to push metrics: {e}"
                )

            await asyncio.sleep(self.interval)

    async def _push_metrics(self):
        """Push current metrics to Redis."""
        key = f"{WORKER_METRICS_PREFIX}{self.worker_id}"
        data = serialize_metrics()

        await self.redis.set(
            key,
            json.dumps(data),
            ex=WORKER_METRICS_TTL,
        )


async def get_aggregated_worker_metrics(redis_client) -> str:
    """
    Read all worker metrics from Redis and return aggregated Prometheus format.

    Args:
        redis_client: Async Redis client

    Returns:
        Prometheus text format string with all worker metrics
    """
    try:
        # Get all worker metric keys
        keys = []
        async for key in redis_client.scan_iter(f"{WORKER_METRICS_PREFIX}*"):
            keys.append(key)

        if not keys:
            return ""

        # Get all values
        values = await redis_client.mget(keys)

        # Parse and aggregate
        workers_data = {}
        for key, value in zip(keys, values):
            if value is None:
                continue

            # Extract worker_id from key
            if isinstance(key, bytes):
                key = key.decode()
            worker_id = key.replace(WORKER_METRICS_PREFIX, "")

            try:
                if isinstance(value, bytes):
                    value = value.decode()
                workers_data[worker_id] = json.loads(value)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

        return format_aggregated_metrics(workers_data)

    except Exception as e:
        import logging
        logging.getLogger("worker_metrics").warning(
            f"Failed to aggregate worker metrics: {e}"
        )
        return ""
