#!/usr/bin/env python3
"""
OCR Worker - Processes OCR jobs from Redis queue.

Thin queue consumer — all GPU work (EasyOCR, orientation correction, PDF
conversion) is delegated to the centralized inference server via Redis.
"""

import asyncio
import contextlib
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import sentry_sdk
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from doc_pipeline.config import get_settings
from doc_pipeline.observability import get_logger, get_metrics, setup_logging
from doc_pipeline.observability.worker_metrics import WorkerMetricsPusher
from doc_pipeline.shared import (
    DeliveryService,
    InferenceClient,
    JobContext,
    QueueService,
)
from doc_pipeline.shared.constants import QueueName

# Setup
settings = get_settings()
setup_logging(json_format=settings.log_json, log_level=settings.log_level)
logger = get_logger("worker-ocr")

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
        self.queue = QueueService(queue_name=QueueName.OCR)
        self.delivery = DeliveryService(queue_service=self.queue)
        self.metrics = get_metrics()
        self._inference_client: InferenceClient | None = None
        self._metrics_pusher: WorkerMetricsPusher | None = None
        self._running = False
        self._current_job: JobContext | None = None
        self._worker_id = "ocr-1"

    async def start(self):
        """Initialize and start the worker."""
        logger.info("worker_ocr_starting", worker_id=self._worker_id)

        await self.queue.connect()

        # Create inference client (shares queue's Redis connection)
        self._inference_client = InferenceClient(
            queue_service=self.queue,
            timeout=settings.inference_timeout_seconds,
        )

        # Start metrics pusher
        self._metrics_pusher = WorkerMetricsPusher(
            redis_client=self.queue._redis,
            worker_id=self._worker_id,
            interval=10.0,
        )
        await self._metrics_pusher.start()
        logger.info("metrics_pusher_started", worker_id=self._worker_id)

        self._running = True
        logger.info("worker_ocr_started", worker_id=self._worker_id)

    async def stop(self):
        """Stop the worker gracefully."""
        logger.info("worker_ocr_stopping", worker_id=self._worker_id)
        self._running = False

        if self._current_job:
            logger.info("waiting_for_current_job", request_id=self._current_job.request_id)

        if self._metrics_pusher:
            await self._metrics_pusher.stop()
            logger.info("metrics_pusher_stopped", worker_id=self._worker_id)

        await self.delivery.close()
        await self.queue.close()
        logger.info("worker_ocr_stopped", worker_id=self._worker_id)

    async def run(self):
        """Main worker loop."""
        while self._running:
            try:
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
                await asyncio.sleep(1)

    async def _process_job(self, job: JobContext):
        """Process a single OCR job by delegating to inference server."""
        job.mark_started()
        start_time = time.perf_counter()

        logger.info(
            "job_processing_start",
            request_id=job.request_id,
            operation=job.operation,
            delivery_mode=job.delivery_mode,
        )

        if job.queue_wait_time_seconds is not None:
            self.metrics.queue_wait_seconds.observe(job.queue_wait_time_seconds)

        try:
            await self.queue.set_progress(job.request_id, "processing")

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
            processing_time = time.perf_counter() - start_time
            result_dict["processing_time_ms"] = round(processing_time * 1000, 2)

            job.mark_completed(result=result_dict)

            # Record success metrics
            self.metrics.worker_processing_seconds.labels(operation="ocr").observe(processing_time)
            self.metrics.jobs_processed.labels(
                operation="ocr",
                status="success",
                delivery_mode=job.delivery_mode,
            ).inc()

            logger.info(
                "job_processing_complete",
                request_id=job.request_id,
                operation=job.operation,
                processing_time_ms=round(processing_time * 1000, 2),
                queue_wait_ms=round((job.queue_wait_time_seconds or 0) * 1000, 2),
                total_pages=result_dict.get("total_pages"),
            )

        except Exception as e:
            sentry_sdk.capture_exception(e)

            job.mark_completed(error=str(e) or f"{type(e).__name__}: timeout")
            self.metrics.jobs_processed.labels(
                operation="ocr",
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
        try:
            await self.queue.set_progress(job.request_id, "delivering")
            success = await self.delivery.deliver(job)

            if not success and job.delivery_mode == "webhook":
                await self.queue.move_to_dlq(job, "Webhook delivery failed")
        except Exception as e:
            logger.error(
                "delivery_error",
                request_id=job.request_id,
                error=str(e),
                error_type=type(e).__name__,
            )


# Global worker instance
worker: OCRWorker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage worker lifecycle."""
    global worker
    worker = OCRWorker()
    await worker.start()

    worker_task = asyncio.create_task(worker.run())

    yield

    await worker.stop()
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task


# Health check app
app = FastAPI(
    title="doc-pipeline OCR Worker",
    description="OCR Worker health check endpoint",
    version="0.2.0",
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

    return {
        "status": "ok",
        "worker": "ocr",
        "worker_running": worker._running,
        "current_job": worker._current_job.request_id if worker._current_job else None,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def main():
    """Run the OCR worker."""
    import uvicorn

    settings = get_settings()

    logger.info(
        "worker_ocr_server_start",
        host="0.0.0.0",
        port=settings.worker_ocr_health_port,
    )

    uvicorn.run(
        "worker_ocr:app",
        host="0.0.0.0",
        port=settings.worker_ocr_health_port,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
