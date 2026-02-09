#!/usr/bin/env python3
"""
Worker for processing document jobs from Redis queue.

This worker runs with GPU access and processes jobs sequentially
to avoid GPU contention.
"""

import asyncio
import os
import re
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import sentry_sdk
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from doc_pipeline import DocumentPipeline
from doc_pipeline.config import ExtractorBackend, get_settings
from doc_pipeline.observability import get_logger, get_metrics, setup_logging
from doc_pipeline.observability.worker_metrics import WorkerMetricsPusher
from doc_pipeline.ocr import OCREngine
from doc_pipeline.ocr.converter import PDFConverter, is_pdf
from doc_pipeline.preprocessing import OrientationCorrector
from doc_pipeline.schemas import CINData, CNHData, DocumentType, ExtractionResult, RGData
from doc_pipeline.shared import (
    DeliveryService,
    InferenceClient,
    JobContext,
    QueueService,
    get_delivery_service,
    get_queue_service,
)
from doc_pipeline.utils import fix_cpf_rg_swap, is_valid_cpf, normalize_cpf, validate_cpf

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
        # Shared OCR engine for hybrid/easyocr extractors
        # This avoids loading EasyOCR multiple times (~4-8GB VRAM savings)
        self._shared_ocr_engine: OCREngine | None = None
        self.orientation_corrector = OrientationCorrector(
            use_text_detection=settings.orientation_enabled,
            device=settings.orientation_device or settings.classifier_device,
            confidence_threshold=settings.orientation_confidence_threshold,
        )
        # Inference client for remote VLM (when inference server is enabled)
        self._inference_client: InferenceClient | None = None
        if settings.inference_server_enabled:
            self._inference_client = InferenceClient(
                queue_service=self.queue,
                timeout=settings.inference_timeout_seconds,
            )
        self._running = False
        self._current_job: JobContext | None = None
        self._in_flight_count: int = 0
        self._max_concurrent = settings.worker_concurrent_jobs if settings.inference_server_enabled else 1
        # Worker ID for metrics aggregation (from env or derived from port)
        self._worker_id = os.environ.get("WORKER_ID", f"docid-{settings.worker_health_port}")
        self._metrics_pusher: WorkerMetricsPusher | None = None

    def _get_shared_ocr_engine(self) -> OCREngine:
        """Get or create the shared OCR engine (lazy loaded)."""
        if self._shared_ocr_engine is None:
            logger.info("initializing_shared_ocr_engine")
            self._shared_ocr_engine = OCREngine(lang="pt", use_gpu=True)
        return self._shared_ocr_engine

    async def start(self) -> None:
        """Initialize and start the worker."""
        logger.info("worker_starting", worker_id=self._worker_id)

        # Connect to Redis
        await self.queue.connect()

        # Start metrics pusher (pushes to Redis for API aggregation)
        self._metrics_pusher = WorkerMetricsPusher(
            redis_client=self.queue._redis,
            worker_id=self._worker_id,
            interval=10.0,
        )
        await self._metrics_pusher.start()
        logger.info("metrics_pusher_started", worker_id=self._worker_id)

        # Initialize pipeline with shared OCR engine
        shared_ocr = self._get_shared_ocr_engine()
        self.pipeline = DocumentPipeline(
            ocr_engine=shared_ocr,
        )

        # Pass OCR engine to orientation corrector for text-box direction detection
        self.orientation_corrector.set_ocr_engine(shared_ocr)

        # Warmup models if configured
        if settings.warmup_on_start:
            logger.info("warmup_start", component="classifier")
            self.pipeline.warmup(load_classifier=True, load_extractor=False)
            logger.info("warmup_complete", component="classifier")

            # Only load local extractor if NOT using inference server
            if not self._inference_client:
                logger.info("warmup_start", component="extractor")
                self.pipeline.warmup(load_classifier=False, load_extractor=True)
                logger.info("warmup_complete", component="extractor")
            else:
                logger.info("extractor_warmup_skipped", reason="inference_server_enabled")

            if settings.orientation_enabled:
                logger.info("warmup_start", component="orientation")
                self.orientation_corrector.warmup()
                logger.info("warmup_complete", component="orientation")

        self._running = True
        logger.info(
            "worker_started",
            worker_id=self._worker_id,
            concurrent_jobs=self._max_concurrent,
            inference_server=bool(self._inference_client),
        )

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info("worker_stopping", worker_id=self._worker_id)
        self._running = False

        # Wait for current/in-flight jobs to complete
        if self._current_job:
            logger.info("waiting_for_current_job", request_id=self._current_job.request_id)
        if self._in_flight_count > 0:
            logger.info("waiting_for_in_flight_jobs", count=self._in_flight_count)

        # Stop metrics pusher
        if self._metrics_pusher:
            await self._metrics_pusher.stop()
            logger.info("metrics_pusher_stopped", worker_id=self._worker_id)

        # Cleanup
        await self.delivery.close()
        await self.queue.close()

        if self.pipeline:
            self.pipeline.unload_extractor()

        logger.info("worker_stopped", worker_id=self._worker_id)

    async def run(self) -> None:
        """Main worker loop — dispatches to sequential or concurrent mode."""
        if self._max_concurrent > 1:
            await self._run_concurrent()
        else:
            await self._run_sequential()

    async def _run_sequential(self) -> None:
        """Sequential job processing (original behavior)."""
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
        """Concurrent job processing for use with inference server.

        Dequeues up to N jobs and processes them in parallel via asyncio tasks.
        This keeps the inference queue fed so the server can form batches.
        """
        semaphore = asyncio.Semaphore(self._max_concurrent)
        tasks: set[asyncio.Task] = set()

        logger.info(
            "concurrent_mode_started",
            max_concurrent=self._max_concurrent,
        )

        while self._running:
            try:
                # Update queue depth metric
                depth = await self.queue.get_queue_depth()
                self.metrics.queue_depth.set(depth)

                # Wait for a semaphore slot (blocks if N jobs already in-flight)
                await semaphore.acquire()

                # Dequeue next job
                job = await self.queue.dequeue(timeout=5.0)
                if job is None:
                    semaphore.release()
                    continue

                # Process in background task
                self._in_flight_count += 1
                task = asyncio.create_task(
                    self._process_job_concurrent(job, semaphore)
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)

            except asyncio.CancelledError:
                logger.info("worker_cancelled", in_flight=self._in_flight_count)
                break
            except Exception as e:
                semaphore.release()
                logger.error("worker_loop_error", error=str(e), error_type=type(e).__name__)
                await asyncio.sleep(1.0)

        # Wait for remaining in-flight tasks
        if tasks:
            logger.info("waiting_for_in_flight_tasks", count=len(tasks))
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_job_concurrent(
        self, job: JobContext, semaphore: asyncio.Semaphore
    ) -> None:
        """Wrapper for _process_job that releases the semaphore when done."""
        try:
            await self._process_job(job)
        finally:
            self._in_flight_count -= 1
            semaphore.release()

    def _get_pipeline(self, job: JobContext) -> DocumentPipeline:
        """Get the appropriate pipeline based on job backend."""
        backend = (job.extra_params or {}).get("backend", "vlm")

        # Map API backend names to internal ExtractorBackend values
        backend_map = {
            "vlm": ExtractorBackend.QWEN_VL,
            "ocr": ExtractorBackend.EASY_OCR,
            "hybrid": ExtractorBackend.HYBRID,
        }
        requested_backend = backend_map.get(backend, ExtractorBackend.QWEN_VL)

        # Check if main pipeline already uses the requested backend
        # This avoids loading duplicate models
        if self.pipeline and self.pipeline._extractor_backend == requested_backend:
            return self.pipeline

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

    def _should_use_remote_inference(self, backend: str) -> bool:
        """Check if remote inference should be used for this backend."""
        return self._inference_client is not None and backend != "ocr"

    async def _extract_remote(
        self, image_path: str, doc_type_str: str, request_id: str, backend: str, image: Image.Image
    ) -> ExtractionResult:
        """
        Extract data using remote inference server.

        For hybrid mode, also validates CPF locally and falls back to OCR if invalid.
        """
        reply = await self._inference_client.extract(
            image_path=image_path,
            document_type=doc_type_str,
            request_id=request_id,
        )
        extraction_data = reply["result"]

        # Fix CPF/RG swap (VLM sometimes confuses them) and normalize CPF
        doc_type = DocumentType(doc_type_str)
        extraction_data = fix_cpf_rg_swap(extraction_data)

        # Hybrid mode: validate CPF locally, fallback to OCR if invalid
        if backend == "hybrid":
            cpf = extraction_data.get("cpf")
            if not is_valid_cpf(cpf):
                logger.info(
                    "hybrid_cpf_invalid_trying_ocr",
                    request_id=request_id,
                    vlm_cpf=cpf,
                )
                ocr_engine = self._get_shared_ocr_engine()
                ocr_text, _ = ocr_engine.extract_text(image)
                # Extract CPF from OCR text using regex (both formats)
                cpf_patterns = [
                    r"\b(\d{3}\.\d{3}\.\d{3}-\d{2})\b",  # ###.###.###-##
                    r"\b(\d{9})/(\d{2})\b",  # #########/##
                ]
                found_cpf = None
                for pattern in cpf_patterns:
                    matches = re.findall(pattern, ocr_text)
                    if matches:
                        if isinstance(matches[0], tuple):
                            # #########/## format — normalize
                            candidate = normalize_cpf(matches[0][0] + matches[0][1])
                        else:
                            candidate = matches[0]
                        if is_valid_cpf(candidate):
                            found_cpf = candidate
                            break
                if found_cpf:
                    logger.info(
                        "hybrid_ocr_cpf_found",
                        request_id=request_id,
                        ocr_cpf=found_cpf,
                    )
                    extraction_data["cpf"] = found_cpf

        # Build ExtractionResult from dict
        if doc_type.is_rg:
            data = RGData(**{k: v for k, v in extraction_data.items() if k in RGData.model_fields})
        elif doc_type.is_cin:
            data = CINData(
                **{k: v for k, v in extraction_data.items() if k in CINData.model_fields}
            )
        else:
            data = CNHData(
                **{k: v for k, v in extraction_data.items() if k in CNHData.model_fields}
            )

        return ExtractionResult(
            document_type=doc_type,
            data=data,
            raw_text=reply.get("raw_response"),
            backend="paneas_v2",
        )

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

        corrected_image_path: str | None = None
        original_pdf_path: str | None = None

        try:
            # Update progress
            await self.queue.set_progress(job.request_id, "processing")

            # Load image (convert PDF first page if needed)
            if not Path(job.image_path).exists():
                raise FileNotFoundError(f"File not found: {job.image_path}")

            if is_pdf(job.image_path):
                original_pdf_path = job.image_path
                converter = PDFConverter(dpi=200)
                pages = converter.convert(job.image_path, max_pages=1)
                if not pages:
                    raise ValueError("PDF has no pages")
                image = pages[0].convert("RGB")
                # Save converted image for inference server
                converted_path = job.image_path.rsplit(".", 1)[0] + "_page1.png"
                image.save(converted_path)
                # Update job path so inference server uses the image
                job.image_path = converted_path
                logger.info(
                    "pdf_converted_to_image",
                    request_id=job.request_id,
                    converted_path=converted_path,
                )
            else:
                image = Image.open(job.image_path).convert("RGB")

            # Orientation correction (can be disabled via auto_rotate=False)
            auto_rotate = (job.extra_params or {}).get("auto_rotate", True)
            if auto_rotate:
                orientation_result = self.orientation_corrector.correct(image)
                if orientation_result.was_corrected:
                    logger.info(
                        "image_orientation_corrected",
                        rotation=orientation_result.rotation_applied.value,
                        method=orientation_result.correction_method,
                        request_id=job.request_id,
                    )
                    image = orientation_result.image
                    # Save corrected image to disk for inference server
                    if self._should_use_remote_inference(backend):
                        corrected_image_path = job.image_path.replace(".", "_corrected.", 1)
                        image.save(corrected_image_path)
                image_correction_info = orientation_result.to_dict()
            else:
                logger.info("orientation_correction_skipped", request_id=job.request_id)
                image_correction_info = {
                    "was_corrected": False,
                    "rotation_applied": 0,
                    "skipped": True,
                }

            # Image path for inference server (corrected or original)
            inference_image_path = corrected_image_path or job.image_path
            use_remote = self._should_use_remote_inference(backend)

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

                if use_remote:
                    result = await self._extract_remote(
                        inference_image_path, job.document_type, job.request_id, backend, image
                    )
                else:
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
                if use_remote and job.extract:
                    # Classify locally, then extract remotely
                    classification = pipeline.classify(image)
                    result_dict = {}

                    if classification.confidence < (job.min_confidence or settings.min_confidence):
                        from doc_pipeline.schemas import PipelineResult

                        result = PipelineResult(
                            classification=classification,
                            extraction=None,
                            success=False,
                            error=f"Confiança ({classification.confidence:.1%}) abaixo do mínimo",
                        )
                        result_dict = result.model_dump()
                    else:
                        doc_type_str = classification.document_type.value
                        extraction = await self._extract_remote(
                            inference_image_path, doc_type_str, job.request_id, backend, image
                        )
                        result_dict = {
                            "file_path": None,
                            "classification": classification.model_dump(),
                            "extraction": extraction.model_dump(),
                            "image_correction": None,
                            "success": True,
                            "error": None,
                        }
                        # For metrics tracking
                        result = type(
                            "Result",
                            (),
                            {
                                "classification": classification,
                                "extraction": extraction,
                            },
                        )()

                    result_dict["image_correction"] = image_correction_info

                    # Validate CPF if requested
                    should_validate_cpf = (job.extra_params or {}).get("validate_cpf", True)
                    extraction_obj = getattr(result, "extraction", None)
                    if (
                        should_validate_cpf
                        and extraction_obj
                        and extraction_obj.data
                        and hasattr(extraction_obj.data, "cpf")
                    ):
                        cpf_validation = validate_cpf(extraction_obj.data.cpf)
                        result_dict["cpf_validation"] = cpf_validation

                    job.mark_completed(result=result_dict)

                else:
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
                self.metrics.classification_confidence.labels(document_type=doc_type).observe(
                    result.confidence
                )
            elif job.operation == "process":
                doc_type = result.classification.document_type.value
                self.metrics.documents_processed.labels(
                    document_type=doc_type, operation=job.operation
                ).inc()
                self.metrics.classification_confidence.labels(document_type=doc_type).observe(
                    result.classification.confidence
                )
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
            sentry_sdk.capture_exception(e)

            # Record error
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
            # Cleanup original PDF (job.image_path was updated to the converted PNG)
            if original_pdf_path:
                try:
                    if Path(original_pdf_path).exists():
                        os.unlink(original_pdf_path)
                except Exception as e:
                    logger.warning(
                        "original_pdf_cleanup_error", path=original_pdf_path, error=str(e)
                    )
            # Cleanup corrected image if it was created for inference server
            if corrected_image_path:
                try:
                    if Path(corrected_image_path).exists():
                        os.unlink(corrected_image_path)
                except Exception as e:
                    logger.warning(
                        "corrected_file_cleanup_error", path=corrected_image_path, error=str(e)
                    )

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
        "in_flight_jobs": worker._in_flight_count,
        "max_concurrent": worker._max_concurrent,
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
