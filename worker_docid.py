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

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from doc_pipeline import DocumentPipeline
from doc_pipeline.config import ExtractorBackend, get_settings
from doc_pipeline.observability import get_logger, get_metrics, setup_logging
from doc_pipeline.ocr import OCREngine
from doc_pipeline.preprocessing import OrientationCorrector
from doc_pipeline.utils import validate_cpf
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

    # Map public backend names to internal ones
    BACKEND_MAP = {
        "vlm": "qwen-vl",
        "ocr": "easy-ocr",
        "hybrid": "hybrid",
    }

    def __init__(self):
        self.pipeline: DocumentPipeline | None = None
        self.pipeline_ocr: DocumentPipeline | None = None  # OCR fallback pipeline
        self.pipeline_hybrid: DocumentPipeline | None = None  # Hybrid pipeline
        self.queue: QueueService = get_queue_service()
        self.delivery: DeliveryService = get_delivery_service()
        self.metrics = get_metrics()
        # Shared OCR engine for orientation correction and hybrid extractor
        # This avoids loading EasyOCR multiple times (~4-8GB VRAM savings)
        self._shared_ocr_engine: OCREngine | None = None
        self.orientation_corrector = OrientationCorrector(use_text_detection=True)
        self._running = False
        self._current_job: JobContext | None = None

    def _get_shared_ocr_engine(self) -> OCREngine:
        """Get or create the shared OCR engine (lazy loaded)."""
        if self._shared_ocr_engine is None:
            logger.info("initializing_shared_ocr_engine")
            self._shared_ocr_engine = OCREngine(lang="pt", use_gpu=True)
        return self._shared_ocr_engine

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

    def _get_pipeline(self, job: JobContext) -> DocumentPipeline:
        """Get the appropriate pipeline based on job backend."""
        backend = (job.extra_params or {}).get("backend", "vlm")

        if backend == "ocr":
            # Lazy init OCR pipeline
            if self.pipeline_ocr is None:
                logger.info("initializing_ocr_pipeline")
                self.pipeline_ocr = DocumentPipeline(
                    extractor_backend=ExtractorBackend.EASY_OCR,
                    ocr_engine=self._get_shared_ocr_engine(),
                )
            return self.pipeline_ocr

        if backend == "hybrid":
            # Lazy init hybrid pipeline
            if self.pipeline_hybrid is None:
                logger.info("initializing_hybrid_pipeline")
                self.pipeline_hybrid = DocumentPipeline(
                    extractor_backend=ExtractorBackend.HYBRID,
                    ocr_engine=self._get_shared_ocr_engine(),
                )
            return self.pipeline_hybrid

        return self.pipeline  # type: ignore

    async def _process_job(self, job: JobContext) -> None:
        """Process a single job."""
        if self.pipeline is None:
            logger.error("pipeline_not_initialized", request_id=job.request_id)
            return

        job.mark_started()
        start_time = time.perf_counter()

        # Get backend from job
        backend = (job.extra_params or {}).get("backend", "vlm")

        logger.info(
            "job_processing_start",
            request_id=job.request_id,
            operation=job.operation,
            backend=backend,
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

            # Orientation correction (can be disabled via auto_rotate=False)
            auto_rotate = (job.extra_params or {}).get("auto_rotate", True)
            if auto_rotate:
                # Ensure orientation corrector uses shared OCR engine
                if self.orientation_corrector._ocr_engine is None:
                    self.orientation_corrector._ocr_engine = self._get_shared_ocr_engine()
                orientation_result = self.orientation_corrector.correct(image)
                if orientation_result.was_corrected:
                    logger.info(
                        "image_orientation_corrected",
                        rotation=orientation_result.rotation_applied.value,
                        method=orientation_result.correction_method,
                        request_id=job.request_id,
                    )
                    image = orientation_result.image
                image_correction_info = orientation_result.to_dict()
            else:
                logger.info("orientation_correction_skipped", request_id=job.request_id)
                image_correction_info = {"was_corrected": False, "rotation_applied": 0, "skipped": True}

            # Get pipeline (vlm or ocr based on backend param)
            pipeline = self._get_pipeline(job)

            # Process based on operation
            if job.operation == "classify":
                result = pipeline.classify(image)
                result_dict = result.model_dump()
                result_dict["image_correction"] = image_correction_info
                job.mark_completed(result=result_dict)

            elif job.operation == "extract":
                if not job.document_type:
                    raise ValueError("document_type required for extract operation")
                result = pipeline.extract(image, job.document_type)
                result_dict = result.model_dump()
                result_dict["image_correction"] = image_correction_info

                # Validate CPF if requested
                should_validate_cpf = (job.extra_params or {}).get("validate_cpf", True)
                if should_validate_cpf and result.data and hasattr(result.data, "cpf"):
                    cpf_validation = validate_cpf(result.data.cpf)
                    result_dict["cpf_validation"] = cpf_validation

                job.mark_completed(result=result_dict)

            elif job.operation == "process":
                result = pipeline.process(
                    image,
                    extract=job.extract,
                    min_confidence=job.min_confidence,
                )
                result_dict = result.model_dump()
                result_dict["image_correction"] = image_correction_info

                # Validate CPF if requested and extraction was done
                should_validate_cpf = (job.extra_params or {}).get("validate_cpf", True)
                if should_validate_cpf and result.extraction and result.extraction.data:
                    if hasattr(result.extraction.data, "cpf"):
                        cpf_validation = validate_cpf(result.extraction.data.cpf)
                        result_dict["cpf_validation"] = cpf_validation

                job.mark_completed(result=result_dict)

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

            # Record business metrics (document type and confidence)
            if job.operation == "classify":
                doc_type = result.document_type.value
                self.metrics.documents_processed.labels(
                    document_type=doc_type, operation=job.operation
                ).inc()
                self.metrics.classification_confidence.labels(
                    document_type=doc_type
                ).observe(result.confidence)
            elif job.operation == "process":
                doc_type = result.classification.document_type.value
                self.metrics.documents_processed.labels(
                    document_type=doc_type, operation=job.operation
                ).inc()
                self.metrics.classification_confidence.labels(
                    document_type=doc_type
                ).observe(result.classification.confidence)
            elif job.operation == "extract":
                doc_type = result.document_type.value
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
        "worker_docid:app",
        host="0.0.0.0",
        port=settings.worker_health_port,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
