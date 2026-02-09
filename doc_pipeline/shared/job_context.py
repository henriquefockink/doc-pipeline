"""Job context for queue-based processing."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class JobContext:
    """
    Carries all job state between API and worker via Redis.

    This is the serializable context that gets enqueued and processed.
    """

    request_id: str
    image_path: str  # Path to temp image file
    operation: str  # "classify", "extract", or "process"

    # Operation parameters
    document_type: str | None = None  # For /extract endpoint
    extract: bool = True  # For /process endpoint
    min_confidence: float = 0.5

    # Client info (for logging/metrics)
    client_name: str | None = None
    api_key_prefix: str | None = None

    # Delivery settings
    delivery_mode: str = "sync"  # "sync" or "webhook"
    webhook_url: str | None = None
    correlation_id: str | None = None

    # Timestamps
    enqueued_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    completed_at: str | None = None

    # Extra parameters (operation-specific)
    extra_params: dict[str, Any] | None = None

    # Result (populated by worker)
    result: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def create(
        cls,
        image_path: str,
        operation: str,
        *,
        document_type: str | None = None,
        extract: bool = True,
        min_confidence: float = 0.5,
        client_name: str | None = None,
        api_key_prefix: str | None = None,
        delivery_mode: str = "sync",
        webhook_url: str | None = None,
        correlation_id: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> JobContext:
        """Create a new JobContext with a generated request_id."""
        return cls(
            request_id=str(uuid.uuid4()),
            image_path=image_path,
            operation=operation,
            document_type=document_type,
            extract=extract,
            min_confidence=min_confidence,
            client_name=client_name,
            api_key_prefix=api_key_prefix,
            delivery_mode=delivery_mode,
            webhook_url=webhook_url,
            correlation_id=correlation_id,
            extra_params=extra_params,
        )

    def mark_started(self) -> None:
        """Mark the job as started."""
        self.started_at = datetime.now(UTC).isoformat()

    def mark_completed(
        self, result: dict[str, Any] | None = None, error: str | None = None
    ) -> None:
        """Mark the job as completed with result or error."""
        self.completed_at = datetime.now(UTC).isoformat()
        self.result = result
        self.error = error

    def to_json(self) -> str:
        """Serialize to JSON string for Redis."""
        return json.dumps(
            {
                "request_id": self.request_id,
                "image_path": self.image_path,
                "operation": self.operation,
                "document_type": self.document_type,
                "extract": self.extract,
                "min_confidence": self.min_confidence,
                "client_name": self.client_name,
                "api_key_prefix": self.api_key_prefix,
                "delivery_mode": self.delivery_mode,
                "webhook_url": self.webhook_url,
                "correlation_id": self.correlation_id,
                "extra_params": self.extra_params,
                "enqueued_at": self.enqueued_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "result": self.result,
                "error": self.error,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> JobContext:
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(
            request_id=d["request_id"],
            image_path=d["image_path"],
            operation=d["operation"],
            document_type=d.get("document_type"),
            extract=d.get("extract", True),
            min_confidence=d.get("min_confidence", 0.5),
            client_name=d.get("client_name"),
            api_key_prefix=d.get("api_key_prefix"),
            delivery_mode=d.get("delivery_mode", "sync"),
            webhook_url=d.get("webhook_url"),
            correlation_id=d.get("correlation_id"),
            extra_params=d.get("extra_params"),
            enqueued_at=d.get("enqueued_at", ""),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            result=d.get("result"),
            error=d.get("error"),
        )

    @property
    def queue_wait_time_seconds(self) -> float | None:
        """Calculate time spent waiting in queue (enqueued -> started)."""
        if not self.enqueued_at or not self.started_at:
            return None
        enqueued = datetime.fromisoformat(self.enqueued_at)
        started = datetime.fromisoformat(self.started_at)
        return (started - enqueued).total_seconds()

    @property
    def queue_wait_ms(self) -> float:
        """Calculate time spent waiting in queue in milliseconds."""
        if not self.enqueued_at:
            return 0.0
        enqueued = datetime.fromisoformat(self.enqueued_at)
        # If started, use started time, otherwise use now
        end = datetime.fromisoformat(self.started_at) if self.started_at else datetime.now(UTC)
        return (end - enqueued).total_seconds() * 1000

    @property
    def processing_time_seconds(self) -> float | None:
        """Calculate processing time (started -> completed)."""
        if not self.started_at or not self.completed_at:
            return None
        started = datetime.fromisoformat(self.started_at)
        completed = datetime.fromisoformat(self.completed_at)
        return (completed - started).total_seconds()

    @property
    def total_time_seconds(self) -> float | None:
        """Calculate total time (enqueued -> completed)."""
        if not self.enqueued_at or not self.completed_at:
            return None
        enqueued = datetime.fromisoformat(self.enqueued_at)
        completed = datetime.fromisoformat(self.completed_at)
        return (completed - enqueued).total_seconds()
