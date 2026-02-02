#!/usr/bin/env python3
"""
Worker for processing document jobs from Redis queue.

This worker runs with GPU access and processes jobs sequentially
to avoid GPU contention.
"""

import asyncio
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from doc_pipeline import DocumentPipeline
from doc_pipeline.config import get_settings
from doc_pipeline.observability import get_logger, get_metrics, setup_logging
from doc_pipeline.shared import (
    DeliveryService,
    JobContext,
    QueueService,
    get_delivery_service,
    get_queue_service,
)

# Setup logging
settings = get_settings()
setup_logging(
    json_format=settings.log_json,
    log_level=settings.log_level,
)

logger = get_logger("worker")


class DocumentWorker:
    """Worker that processes document jobs from Redis queue."""

    def __init__(self):
        self.pipeline: DocumentPipeline | None = None
        self.queue: QueueService = get_queue_service()
        self.delivery: DeliveryService = get_delivery_service()
        self.metrics = get_metrics()
        self._running = False
        self._current_job: JobContext | None = None

    async def start(self) -> None:
        """Initialize and start the worker."""
        logger.info("worker_starting")

        # Connect to Redis
        await self.queue.connect()

        # Initialize pipeline
        self.pipeline = DocumentPipeline()

        # Warmup models if configured
        if settings.warmup_on_start:
            logger.info("warmup_start", component="classifier")
            self.pipeline.warmup(load_classifier=True, load_extractor=False)
            logger.info("warmup_complete", component="classifier")

            logger.info("warmup_start", component="extractor")
            self.pipeline.warmup(load_classifier=False, load_extractor=True)
            logger.info("warmup_complete", component="extractor")

        self._running = True
        logger.info("worker_started")

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info("worker_stopping")
        self._running = False

        # Wait for current job to complete
        if self._current_job:
            logger.info("waiting_for_current_job", request_id=self._current_job.request_id)

        # Cleanup
        await self.delivery.close()
        await self.queue.close()

        if self.pipeline:
            self.pipeline.unload_extractor()

        logger.info("worker_stopped")

    async def run(self) -> None:
        """Main worker loop."""
        while self._running:
            try:
                # Update queue depth metric
                depth = await self.queue.get_queue_depth()
                self.metrics.queue_depth.set(depth)

                # Wait for next job
                job = await self.queue.dequeue(timeout=5.0)
                if job is None:
                    continue

                self._current_job = job
                await self._process_job(job)
                self._current_job = None

            except asyncio.CancelledError:
                logger.info("worker_cancelled")
                break
            except Exception as e:
                logger.error("worker_loop_error", error=str(e), error_type=type(e).__name__)
                # Brief pause before retrying
                await asyncio.sleep(1.0)

    async def _process_job(self, job: JobContext) -> None:
        """Process a single job."""
        if self.pipeline is None:
            logger.error("pipeline_not_initialized", request_id=job.request_id)
            return

        job.mark_started()
        start_time = time.perf_counter()

        logger.info(
            "job_processing_start",
            request_id=job.request_id,
            operation=job.operation,
            client=job.client_name,
            delivery_mode=job.delivery_mode,
        )

        # Record queue wait time
        if job.queue_wait_time_seconds is not None:
            self.metrics.queue_wait_seconds.observe(job.queue_wait_time_seconds)

        try:
            # Update progress
            await self.queue.set_progress(job.request_id, "processing")

            # Load image
            if not Path(job.image_path).exists():
                raise FileNotFoundError(f"Image not found: {job.image_path}")

            image = Image.open(job.image_path).convert("RGB")

            # Process based on operation
            if job.operation == "classify":
                result = self.pipeline.classify(image)
                job.mark_completed(result=result.model_dump())

            elif job.operation == "extract":
                if not job.document_type:
                    raise ValueError("document_type required for extract operation")
                result = self.pipeline.extract(image, job.document_type)
                job.mark_completed(result=result.model_dump())

            elif job.operation == "process":
                result = self.pipeline.process(
                    image,
                    extract=job.extract,
                    min_confidence=job.min_confidence,
                )
                job.mark_completed(result=result.model_dump())

            else:
                raise ValueError(f"Unknown operation: {job.operation}")

            # Record success metrics
            processing_time = time.perf_counter() - start_time
            self.metrics.worker_processing_seconds.labels(operation=job.operation).observe(
                processing_time
            )
            self.metrics.jobs_processed.labels(
                operation=job.operation,
                status="success",
                delivery_mode=job.delivery_mode,
            ).inc()

            logger.info(
                "job_processing_complete",
                request_id=job.request_id,
                operation=job.operation,
                processing_time_ms=round(processing_time * 1000, 2),
                queue_wait_ms=round((job.queue_wait_time_seconds or 0) * 1000, 2),
            )

        except Exception as e:
            # Record error
            job.mark_completed(error=str(e))
            self.metrics.jobs_processed.labels(
                operation=job.operation,
                status="error",
                delivery_mode=job.delivery_mode,
            ).inc()

            logger.error(
                "job_processing_error",
                request_id=job.request_id,
                operation=job.operation,
                error=str(e),
                error_type=type(e).__name__,
            )

        finally:
            # Cleanup temp file
            try:
                if Path(job.image_path).exists():
                    os.unlink(job.image_path)
            except Exception as e:
                logger.warning("temp_file_cleanup_error", path=job.image_path, error=str(e))

        # Deliver result
        await self.queue.set_progress(job.request_id, "delivering")
        success = await self.delivery.deliver(job)

        if not success and job.delivery_mode == "webhook":
            # Move to DLQ if webhook delivery failed
            await self.queue.move_to_dlq(job, "Webhook delivery failed")


# Global worker instance
worker: DocumentWorker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage worker lifecycle."""
    global worker
    worker = DocumentWorker()
    await worker.start()

    # Start worker loop in background
    worker_task = asyncio.create_task(worker.run())

    yield

    # Shutdown
    await worker.stop()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


# Health check app
app = FastAPI(
    title="doc-pipeline Worker",
    description="Worker health check endpoint",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Worker health check."""
    if worker is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Worker not initialized"},
        )

    queue_depth = await worker.queue.get_queue_depth()
    dlq_depth = await worker.queue.get_dlq_depth()

    return {
        "status": "ok",
        "worker_running": worker._running,
        "current_job": worker._current_job.request_id if worker._current_job else None,
        "queue_depth": queue_depth,
        "dlq_depth": dlq_depth,
        "pipeline_loaded": worker.pipeline is not None,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def main():
    """Run the worker."""
    import uvicorn

    settings = get_settings()

    # Handle signals
    def signal_handler(signum, frame):
        logger.info("signal_received", signal=signum)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(
        "worker_server_start",
        host="0.0.0.0",
        port=settings.worker_health_port,
    )

    uvicorn.run(
        "worker:app",
        host="0.0.0.0",
        port=settings.worker_health_port,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
