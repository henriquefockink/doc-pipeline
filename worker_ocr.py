#!/usr/bin/env python3
"""
OCR Worker - Processes OCR jobs from Redis queue using PaddleOCR.

This worker is separate from the main classification worker because:
1. PaddleOCR is much faster (~0.2s vs 3s)
2. Uses less VRAM (~2GB vs 16GB)
3. Different scaling requirements (usually 1 worker is enough)
"""

import asyncio
import signal
import sys
import time
from pathlib import Path

from PIL import Image

from doc_pipeline.config import get_settings
from doc_pipeline.observability import get_logger, get_metrics, setup_logging
from doc_pipeline.ocr import OCREngine, PDFConverter
from doc_pipeline.ocr.converter import is_pdf
from doc_pipeline.schemas import OCRPageResult, OCRResult
from doc_pipeline.shared import DeliveryService, JobContext, QueueService
from doc_pipeline.shared.constants import QueueName

# Setup
settings = get_settings()
setup_logging(json_format=settings.log_json, log_level=settings.log_level)
logger = get_logger("worker-ocr")
metrics = get_metrics()


class OCRWorker:
    """Worker that processes OCR jobs from Redis queue."""

    def __init__(self):
        self.settings = get_settings()
        self.queue = QueueService(queue_name=QueueName.OCR)
        # Pass same queue service to delivery so they share the connection
        self.delivery = DeliveryService(queue_service=self.queue)
        self._running = False

        # Lazy-loaded components
        self._ocr_engine: OCREngine | None = None
        self._pdf_converter: PDFConverter | None = None

    @property
    def ocr_engine(self) -> OCREngine:
        """Lazy load OCR engine."""
        if self._ocr_engine is None:
            self._ocr_engine = OCREngine(
                lang=self.settings.ocr_language,
                use_gpu=self.settings.ocr_use_gpu,
                show_log=False,
            )
        return self._ocr_engine

    @property
    def pdf_converter(self) -> PDFConverter:
        """Lazy load PDF converter."""
        if self._pdf_converter is None:
            # 150 DPI - good balance between quality and speed
            # 200 DPI = ~6s/page (alta qualidade, lento)
            # 150 DPI = ~4s/page (boa qualidade, aceitável)
            # 100 DPI = ~3s/page (qualidade ruim)
            self._pdf_converter = PDFConverter(dpi=150)
        return self._pdf_converter

    def warmup(self):
        """Warmup models for faster first inference."""
        logger.info("warmup_start", component="ocr")
        self.ocr_engine.warmup()
        logger.info("warmup_complete", component="ocr")

    async def run(self):
        """Main worker loop."""
        self._running = True

        # Connect to Redis
        await self.queue.connect()

        logger.info("worker_ocr_started")

        # Warmup if configured
        if self.settings.warmup_on_start:
            self.warmup()

        while self._running:
            try:
                # Wait for job from queue
                job = await self.queue.dequeue(timeout=5.0)

                if job is None:
                    continue

                await self._process_job(job)

            except asyncio.CancelledError:
                logger.info("worker_cancelled")
                break
            except Exception as e:
                logger.exception("worker_error", error=str(e))
                await asyncio.sleep(1)

        # Cleanup
        await self.queue.close()
        await self.delivery.close()
        logger.info("worker_ocr_stopped")

    async def _process_job(self, job: JobContext):
        """Process a single OCR job."""
        start_time = time.perf_counter()
        queue_wait_ms = job.queue_wait_ms

        logger.info(
            "job_processing_start",
            request_id=job.request_id,
            operation=job.operation,
            delivery_mode=job.delivery_mode,
        )

        try:
            # Get file path and max pages from job
            file_path = Path(job.image_path)
            max_pages = job.extra_params.get("max_pages", 10) if job.extra_params else 10

            # Process based on file type
            if is_pdf(file_path):
                result = await self._process_pdf(job.request_id, file_path, max_pages)
            else:
                result = await self._process_image(job.request_id, file_path)

            # Record success
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            result_dict = result.model_dump()
            result_dict["processing_time_ms"] = processing_time_ms

            job.result = result_dict
            job.error = None

            logger.info(
                "job_processing_complete",
                request_id=job.request_id,
                operation=job.operation,
                processing_time_ms=processing_time_ms,
                queue_wait_ms=queue_wait_ms,
                total_pages=result.total_pages,
            )

            # Record metrics
            metrics.jobs_processed.labels(
                operation="ocr",
                status="success",
                delivery_mode=job.delivery_mode,
            ).inc()

        except Exception as e:
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            logger.exception(
                "job_processing_error",
                request_id=job.request_id,
                error=str(e),
            )

            job.result = None
            job.error = str(e)

            metrics.jobs_processed.labels(
                operation="ocr",
                status="error",
                delivery_mode=job.delivery_mode,
            ).inc()

        # Deliver result
        await self.delivery.deliver(job)

        # Cleanup temp file
        try:
            Path(job.image_path).unlink(missing_ok=True)
        except Exception:
            pass

    async def _process_pdf(
        self,
        request_id: str,
        file_path: Path,
        max_pages: int,
    ) -> OCRResult:
        """Process PDF file."""
        import time

        # Convert PDF to images
        t0 = time.perf_counter()
        images = self.pdf_converter.convert(file_path, max_pages=max_pages)
        convert_time = (time.perf_counter() - t0) * 1000
        logger.info(
            "pdf_converted",
            request_id=request_id,
            pages=len(images),
            convert_time_ms=convert_time,
        )

        # OCR each page
        pages = []
        for i, img in enumerate(images, start=1):
            t0 = time.perf_counter()
            text, confidence = self.ocr_engine.extract_text(img)
            ocr_time = (time.perf_counter() - t0) * 1000
            logger.info(
                "page_ocr_complete",
                request_id=request_id,
                page=i,
                ocr_time_ms=ocr_time,
                image_size=f"{img.width}x{img.height}",
            )
            pages.append(OCRPageResult(
                page=i,
                text=text,
                confidence=confidence,
            ))

        return OCRResult(
            request_id=request_id,
            total_pages=len(pages),
            pages=pages,
            processing_time_ms=0,  # Will be set by caller
            file_type="pdf",
            language="pt",
        )

    async def _process_image(
        self,
        request_id: str,
        file_path: Path,
    ) -> OCRResult:
        """Process image file."""
        img = Image.open(file_path)
        text, confidence = self.ocr_engine.extract_text(img)

        return OCRResult(
            request_id=request_id,
            total_pages=1,
            pages=[OCRPageResult(
                page=1,
                text=text,
                confidence=confidence,
            )],
            processing_time_ms=0,
            file_type="image",
            language="pt",
        )

    def stop(self):
        """Signal worker to stop."""
        self._running = False


# Health check server
async def health_server(worker: OCRWorker):
    """Simple HTTP health check server."""
    from aiohttp import web

    async def health_handler(request):
        return web.json_response({
            "status": "healthy",
            "worker": "ocr",
            "running": worker._running,
        })

    async def metrics_handler(request):
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return web.Response(
            body=generate_latest(),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/metrics", metrics_handler)

    runner = web.AppRunner(app)
    await runner.setup()

    port = settings.worker_ocr_health_port
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("health_server_started", port=port)

    return runner


async def main():
    """Main entry point."""
    worker = OCRWorker()

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("shutdown_signal_received", signal=sig)
        worker.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start health server
    health_runner = await health_server(worker)

    try:
        await worker.run()
    finally:
        await health_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
