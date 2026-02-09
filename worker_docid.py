#!/usr/bin/env python3
"""
Worker for processing document jobs from Redis queue.

Thin queue consumer — all GPU work (classification, extraction, orientation,
OCR) is delegated to the centralized inference server via Redis.
"""

import asyncio
import contextlib
import os
import signal
import sys
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

# Sentry / GlitchTip
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        server_name=os.environ.get("WORKER_ID", f"worker-docid-{settings.worker_health_port}"),
    )


class DocumentWorker:
    """Stateless worker that delegates all GPU work to inference server."""

    def __init__(self):
        self.queue: QueueService = get_queue_service()
        self.delivery: DeliveryService = get_delivery_service()
        self.metrics = get_metrics()
        self._inference_client = InferenceClient(
            queue_service=self.queue,
            timeout=settings.inference_timeout_seconds,
        )
        self._running = False
        self._current_job: JobContext | None = None
        self._in_flight_count: int = 0
        self._max_concurrent = settings.worker_concurrent_jobs
        self._worker_id = os.environ.get("WORKER_ID", f"docid-{settings.worker_health_port}")
        self._metrics_pusher: WorkerMetricsPusher | None = None

    async def start(self) -> None:
        """Initialize and start the worker."""
        logger.info("worker_starting", worker_id=self._worker_id)

        await self.queue.connect()

        # Start metrics pusher
        self._metrics_pusher = WorkerMetricsPusher(
            redis_client=self.queue._redis,
            worker_id=self._worker_id,
            interval=10.0,
        )
        await self._metrics_pusher.start()
        logger.info("metrics_pusher_started", worker_id=self._worker_id)

        self._running = True
        logger.info(
            "worker_started",
            worker_id=self._worker_id,
            concurrent_jobs=self._max_concurrent,
        )

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info("worker_stopping", worker_id=self._worker_id)
        self._running = False

        if self._current_job:
            logger.info("waiting_for_current_job", request_id=self._current_job.request_id)
        if self._in_flight_count > 0:
            logger.info("waiting_for_in_flight_jobs", count=self._in_flight_count)

        if self._metrics_pusher:
            await self._metrics_pusher.stop()
            logger.info("metrics_pusher_stopped", worker_id=self._worker_id)

        await self.delivery.close()
        await self.queue.close()
        logger.info("worker_stopped", worker_id=self._worker_id)

    async def run(self) -> None:
        """Main worker loop — dispatches to sequential or concurrent mode."""
        if self._max_concurrent > 1:
            await self._run_concurrent()
        else:
            await self._run_sequential()

    async def _run_sequential(self) -> None:
        """Sequential job processing."""
        while self._running:
            try:
                depth = await self.queue.get_queue_depth()
                self.metrics.queue_depth.set(depth)

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
                await asyncio.sleep(1.0)

    async def _run_concurrent(self) -> None:
        """Concurrent job processing — keeps inference queue fed for batching."""
        semaphore = asyncio.Semaphore(self._max_concurrent)
        tasks: set[asyncio.Task] = set()

        logger.info(
            "concurrent_mode_started",
            max_concurrent=self._max_concurrent,
        )

        while self._running:
            try:
                depth = await self.queue.get_queue_depth()
                self.metrics.queue_depth.set(depth)

                await semaphore.acquire()

                job = await self.queue.dequeue(timeout=5.0)
                if job is None:
                    semaphore.release()
                    continue

                self._in_flight_count += 1
                task = asyncio.create_task(self._process_job_concurrent(job, semaphore))
                tasks.add(task)
                task.add_done_callback(tasks.discard)

            except asyncio.CancelledError:
                logger.info("worker_cancelled", in_flight=self._in_flight_count)
                break
            except Exception as e:
                semaphore.release()
                logger.error("worker_loop_error", error=str(e), error_type=type(e).__name__)
                await asyncio.sleep(1.0)

        if tasks:
            logger.info("waiting_for_in_flight_tasks", count=len(tasks))
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_job_concurrent(self, job: JobContext, semaphore: asyncio.Semaphore) -> None:
        """Wrapper for _process_job that releases the semaphore when done."""
        try:
            await self._process_job(job)
        finally:
            self._in_flight_count -= 1
            semaphore.release()

    async def _process_job(self, job: JobContext) -> None:
        """Process a single job by delegating to inference server."""
        job.mark_started()
        start_time = time.perf_counter()

        backend = (job.extra_params or {}).get("backend", "vlm")

        logger.info(
            "job_processing_start",
            request_id=job.request_id,
            operation=job.operation,
            backend=backend,
            client=job.client_name,
            delivery_mode=job.delivery_mode,
        )

        if job.queue_wait_time_seconds is not None:
            self.metrics.queue_wait_seconds.observe(job.queue_wait_time_seconds)

        try:
            await self.queue.set_progress(job.request_id, "processing")

            if not Path(job.image_path).exists():
                raise FileNotFoundError(f"File not found: {job.image_path}")

            # Delegate everything to inference server
            reply = await self._inference_client.request(
                operation=job.operation,
                image_path=job.image_path,
                request_id=job.request_id,
                document_type=job.document_type,
                backend=backend,
                auto_rotate=(job.extra_params or {}).get("auto_rotate", True),
                max_pages=(job.extra_params or {}).get("max_pages", 10),
                extract=job.extract,
                min_confidence=job.min_confidence,
            )

            result_dict = reply["result"]
            job.mark_completed(result=result_dict)

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

            # Record business metrics from result
            if job.operation in ("classify", "process") and result_dict:
                classification = result_dict.get("classification", {})
                doc_type = classification.get("document_type", "unknown")
                confidence = classification.get("confidence", 0)
                self.metrics.documents_processed.labels(
                    document_type=doc_type, operation=job.operation
                ).inc()
                self.metrics.classification_confidence.labels(document_type=doc_type).observe(
                    confidence
                )
            elif job.operation == "extract" and result_dict:
                extraction = result_dict.get("extraction", {})
                doc_type = extraction.get("document_type") or job.document_type or "unknown"
                self.metrics.documents_processed.labels(
                    document_type=doc_type, operation=job.operation
                ).inc()

            logger.info(
                "job_processing_complete",
                request_id=job.request_id,
                operation=job.operation,
                processing_time_ms=round(processing_time * 1000, 2),
                queue_wait_ms=round((job.queue_wait_time_seconds or 0) * 1000, 2),
            )

        except Exception as e:
            sentry_sdk.capture_exception(e)

            job.mark_completed(error=str(e) or f"{type(e).__name__}: timeout")
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
            # Cleanup temp files
            try:
                if Path(job.image_path).exists():
                    os.unlink(job.image_path)
            except Exception as e:
                logger.warning("temp_file_cleanup_error", path=job.image_path, error=str(e))

        # Deliver result
        await self.queue.set_progress(job.request_id, "delivering")
        success = await self.delivery.deliver(job)

        if not success and job.delivery_mode == "webhook":
            await self.queue.move_to_dlq(job, "Webhook delivery failed")


# Global worker instance
worker: DocumentWorker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage worker lifecycle."""
    global worker
    worker = DocumentWorker()
    await worker.start()

    worker_task = asyncio.create_task(worker.run())

    yield

    await worker.stop()
    worker_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await worker_task


# Health check app
app = FastAPI(
    title="doc-pipeline Worker",
    description="Worker health check endpoint",
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

    queue_depth = await worker.queue.get_queue_depth()
    dlq_depth = await worker.queue.get_dlq_depth()

    return {
        "status": "ok",
        "worker_running": worker._running,
        "current_job": worker._current_job.request_id if worker._current_job else None,
        "in_flight_jobs": worker._in_flight_count,
        "max_concurrent": worker._max_concurrent,
        "queue_depth": queue_depth,
        "dlq_depth": dlq_depth,
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
        "worker_docid:app",
        host="0.0.0.0",
        port=settings.worker_health_port,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
