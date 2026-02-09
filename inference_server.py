#!/usr/bin/env python3
"""
Centralized inference server for all GPU models.

Runs as a standalone process with GPU access. Workers send inference
requests via Redis queue, this server processes them and replies via Redis keys.

Models loaded:
- EfficientNet classifier (~200MB VRAM)
- EasyOCR engine (~2-4GB VRAM)
- Qwen VLM extractor (~12GB VRAM)
- Orientation corrector (shares EasyOCR)
- PDF converter (CPU only)

Supports batched VLM inference: collects multiple extract/process requests
and processes them in a single GPU forward pass for higher throughput.
"""

import asyncio
import contextlib
import json
import re
import signal
import sys
import time
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from doc_pipeline.classifier.adapter import ClassifierAdapter
from doc_pipeline.config import get_settings
from doc_pipeline.extractors.qwen_vl import QwenVLExtractor
from doc_pipeline.extractors.vllm_client import VLLMClient
from doc_pipeline.observability import get_logger, setup_logging
from doc_pipeline.ocr import OCREngine
from doc_pipeline.ocr.converter import PDFConverter, is_pdf
from doc_pipeline.preprocessing import OrientationCorrector
from doc_pipeline.prompts import CIN_EXTRACTION_PROMPT, CNH_EXTRACTION_PROMPT, RG_EXTRACTION_PROMPT
from doc_pipeline.schemas import CINData, CNHData, RGData
from doc_pipeline.shared.constants import INFERENCE_REPLY_TTL, QueueName, inference_reply_key
from doc_pipeline.shared.queue import QueueService, get_queue_service
from doc_pipeline.utils import fix_cpf_rg_swap, is_valid_cpf, normalize_cpf, validate_cpf

# Setup logging
settings = get_settings()
setup_logging(
    json_format=settings.log_json,
    log_level=settings.log_level,
)

logger = get_logger("inference_server")

# Sentry / GlitchTip
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        server_name="inference-server",
    )


# Prometheus metrics for inference server
inference_metrics_requests_total = Counter(
    "inference_requests_total",
    "Total inference requests processed",
    ["operation", "status"],
)
inference_metrics_duration_seconds = Histogram(
    "inference_duration_seconds",
    "Time spent on inference",
    ["operation"],
    buckets=(0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0),
)
inference_metrics_queue_depth = Gauge(
    "inference_queue_depth",
    "Number of pending inference requests",
)
inference_metrics_batch_size = Histogram(
    "inference_batch_size",
    "Number of requests per batch",
    buckets=(1, 2, 3, 4, 5, 6, 8, 10, 12, 16),
)


class InferenceServer:
    """Server that processes all inference requests from Redis queue."""

    def __init__(self):
        self.queue: QueueService = get_queue_service()
        self.extractor: QwenVLExtractor | None = None
        self.vllm_client: VLLMClient | None = None
        self.classifier: ClassifierAdapter | None = None
        self.ocr_engine: OCREngine | None = None
        self.orientation_corrector: OrientationCorrector | None = None
        self.pdf_converter: PDFConverter | None = None
        self._running = False
        self._current_batch_size: int = 0
        self._batch_size = settings.inference_batch_size
        self._batch_timeout_ms = settings.inference_batch_timeout_ms

    async def start(self) -> None:
        """Initialize and start the inference server with all models."""
        logger.info("inference_server_starting")

        # Connect to Redis
        await self.queue.connect()

        # Load classifier (EfficientNet, ~200MB)
        logger.info(
            "loading_classifier",
            model_path=str(settings.classifier_model_path),
            model_type=settings.classifier_model_type,
            device=settings.classifier_device,
        )
        self.classifier = ClassifierAdapter(
            model_path=settings.classifier_model_path,
            model_type=settings.classifier_model_type,
            device=settings.classifier_device,
            fp8=settings.classifier_fp8,
        )
        logger.info("classifier_loaded")

        # Load OCR engine (EasyOCR, ~2-4GB)
        logger.info("loading_ocr_engine", lang=settings.ocr_language)
        self.ocr_engine = OCREngine(
            lang=settings.ocr_language,
            use_gpu=settings.ocr_use_gpu,
        )
        self.ocr_engine.warmup()
        logger.info("ocr_engine_loaded")

        # Load orientation corrector (shares EasyOCR)
        self.orientation_corrector = OrientationCorrector(
            use_text_detection=settings.orientation_enabled,
            device=settings.classifier_device,
            confidence_threshold=settings.orientation_confidence_threshold,
        )
        self.orientation_corrector.set_ocr_engine(self.ocr_engine)
        logger.info("orientation_corrector_loaded")

        # Load PDF converter (CPU only)
        self.pdf_converter = PDFConverter(dpi=200)
        logger.info("pdf_converter_loaded")

        # Load VLM backend: vLLM (external) or HuggingFace (local)
        if settings.vllm_enabled:
            logger.info(
                "loading_vllm_client",
                base_url=settings.vllm_base_url,
                model=settings.vllm_model,
            )
            self.vllm_client = VLLMClient(
                base_url=settings.vllm_base_url,
                model=settings.vllm_model,
                max_tokens=settings.vllm_max_tokens,
                timeout=settings.vllm_timeout,
            )
            await self.vllm_client.start()

            # Wait for vLLM to be ready (up to 5 min)
            for attempt in range(60):
                if await self.vllm_client.health_check():
                    logger.info("vllm_server_ready")
                    break
                logger.info("vllm_waiting_for_server", attempt=attempt + 1)
                await asyncio.sleep(5.0)
            else:
                raise RuntimeError("vLLM server not ready after 5 minutes")
        else:
            logger.info(
                "loading_vlm_model",
                model=settings.extractor_model_qwen,
                device=settings.extractor_device,
            )
            self.extractor = QwenVLExtractor(
                model_name=settings.extractor_model_qwen,
                device=settings.extractor_device,
            )
            self.extractor.load_model()
            logger.info("vlm_model_loaded")

        self._running = True
        logger.info(
            "inference_server_started",
            batch_size=self._batch_size,
            batch_timeout_ms=self._batch_timeout_ms,
        )

    async def stop(self) -> None:
        """Stop the inference server gracefully."""
        logger.info("inference_server_stopping")
        self._running = False

        if self.vllm_client:
            await self.vllm_client.stop()

        if self.extractor:
            self.extractor.unload_model()

        await self.queue.close()
        logger.info("inference_server_stopped")

    async def _collect_batch(self) -> list[dict]:
        """Collect a batch of requests from the queue."""
        result = await self.queue.redis.brpop(QueueName.INFERENCE, timeout=5.0)
        if result is None:
            return []

        _, request_json = result
        batch = [json.loads(request_json)]

        if self._batch_size <= 1:
            return batch

        # Try to fill the batch (non-blocking with short timeout)
        deadline = time.perf_counter() + self._batch_timeout_ms / 1000
        while len(batch) < self._batch_size:
            remaining_ms = (deadline - time.perf_counter()) * 1000
            if remaining_ms <= 0:
                break
            extra = await self.queue.redis.rpop(QueueName.INFERENCE)
            if extra is None:
                await asyncio.sleep(min(0.01, remaining_ms / 1000))
                extra = await self.queue.redis.rpop(QueueName.INFERENCE)
                if extra is None:
                    break
            batch.append(json.loads(extra))

        return batch

    async def run(self) -> None:
        """Main server loop - consume batches from inference queue."""
        while self._running:
            try:
                depth = await self.queue.redis.llen(QueueName.INFERENCE)
                inference_metrics_queue_depth.set(depth)

                batch = await self._collect_batch()
                if not batch:
                    continue

                self._current_batch_size = len(batch)
                await self._process_batch_smart(batch)
                self._current_batch_size = 0

            except asyncio.CancelledError:
                logger.info("inference_server_cancelled")
                break
            except Exception as e:
                logger.error(
                    "inference_loop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._current_batch_size = 0
                await asyncio.sleep(1.0)

    async def _vlm_generate(self, image: Image.Image, prompt: str) -> str:
        """Generate VLM response via vLLM or local HuggingFace."""
        if self.vllm_client:
            return await self.vllm_client.generate(image, prompt)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extractor._generate, image, prompt)

    async def _vlm_generate_batch(self, images: list[Image.Image], prompts: list[str]) -> list[str]:
        """Generate VLM responses for a batch via vLLM or local HuggingFace."""
        if self.vllm_client:
            return await self.vllm_client.generate_batch(images, prompts)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extractor._generate_batch, images, prompts)

    def _parse_vlm_json(self, text: str) -> dict:
        """Parse JSON from VLM response using the appropriate parser."""
        if self.vllm_client:
            return VLLMClient.parse_json(text)
        return self.extractor._parse_json(text)

    async def _process_batch_smart(self, batch: list[dict]) -> None:
        """Process a mixed batch: route by operation, batch VLM calls together."""
        # Separate by operation type
        vlm_requests = []  # extract/process requests that need VLM
        fast_requests = []  # classify/ocr requests (no VLM needed)

        for req in batch:
            operation = req.get("operation", "extract")
            if operation in ("extract", "process"):
                vlm_requests.append(req)
            else:
                fast_requests.append(req)

        # Process fast requests individually (classify ~200ms, ocr ~2s)
        for req in fast_requests:
            await self._process_single(req)

        # Process VLM requests as a batch
        if vlm_requests:
            if len(vlm_requests) == 1:
                await self._process_single(vlm_requests[0])
            else:
                await self._process_vlm_batch(vlm_requests)

    # ------------------------------------------------------------------
    # Image loading & preprocessing helpers
    # ------------------------------------------------------------------

    def _load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB."""
        return Image.open(image_path).convert("RGB")

    def _correct_orientation(
        self, image: Image.Image, auto_rotate: bool
    ) -> tuple[Image.Image, dict]:
        """Correct image orientation, return (image, correction_info)."""
        if not auto_rotate:
            return image, {
                "was_corrected": False,
                "rotation_applied": 0,
                "skipped": True,
            }

        result = self.orientation_corrector.correct(image)
        return result.image, result.to_dict()

    def _convert_pdf_first_page(self, file_path: str) -> Image.Image:
        """Convert first page of PDF to image."""
        pages = self.pdf_converter.convert(file_path, max_pages=1)
        if not pages:
            raise ValueError("PDF has no pages")
        return pages[0].convert("RGB")

    # ------------------------------------------------------------------
    # VLM helpers (shared by extract and process)
    # ------------------------------------------------------------------

    def _get_prompt(self, doc_type: str) -> str:
        """Get the extraction prompt for a document type."""
        if doc_type.startswith("rg"):
            return RG_EXTRACTION_PROMPT
        elif doc_type.startswith("cnh"):
            return CNH_EXTRACTION_PROMPT
        elif doc_type.startswith("cin"):
            return CIN_EXTRACTION_PROMPT
        else:
            raise ValueError(f"Unknown document type: {doc_type}")

    def _build_extraction_result(self, raw_text: str, doc_type: str) -> dict:
        """Parse VLM response and build structured extraction result dict."""
        data = self._parse_vlm_json(raw_text)
        data = fix_cpf_rg_swap(data)

        if doc_type.startswith("rg"):
            model = RGData(**{k: v for k, v in data.items() if k in RGData.model_fields})
        elif doc_type.startswith("cin"):
            model = CINData(**{k: v for k, v in data.items() if k in CINData.model_fields})
        else:
            model = CNHData(**{k: v for k, v in data.items() if k in CNHData.model_fields})

        return model.model_dump()

    def _hybrid_cpf_validation(
        self, extraction_data: dict, image: Image.Image, request_id: str
    ) -> dict:
        """Hybrid mode: validate CPF via EasyOCR fallback if VLM CPF is invalid."""
        cpf = extraction_data.get("cpf")
        if is_valid_cpf(cpf):
            return extraction_data

        logger.info(
            "hybrid_cpf_invalid_trying_ocr",
            request_id=request_id,
            vlm_cpf=cpf,
        )
        ocr_text, _ = self.ocr_engine.extract_text(image)
        cpf_patterns = [
            r"\b(\d{3}\.\d{3}\.\d{3}-\d{2})\b",
            r"\b(\d{9})/(\d{2})\b",
        ]
        for pattern in cpf_patterns:
            matches = re.findall(pattern, ocr_text)
            if matches:
                if isinstance(matches[0], tuple):
                    candidate = normalize_cpf(matches[0][0] + matches[0][1])
                else:
                    candidate = matches[0]
                if is_valid_cpf(candidate):
                    logger.info(
                        "hybrid_ocr_cpf_found",
                        request_id=request_id,
                        ocr_cpf=candidate,
                    )
                    extraction_data["cpf"] = candidate
                    return extraction_data

        return extraction_data

    # ------------------------------------------------------------------
    # Operation handlers
    # ------------------------------------------------------------------

    def _handle_classify(self, request: dict, image: Image.Image) -> dict:
        """Handle classify operation: orient + classify via EfficientNet."""
        classification = self.classifier.classify(image)
        return {
            "classification": classification.model_dump(),
        }

    async def _handle_extract_single(self, request: dict, image: Image.Image) -> dict:
        """Handle extract operation for a single request (uses VLM)."""
        doc_type = request["document_type"]
        backend = request.get("backend", "vlm")
        prompt = self._get_prompt(doc_type)
        raw_text = await self._vlm_generate(image, prompt)
        extraction_data = self._build_extraction_result(raw_text, doc_type)

        # Hybrid CPF validation
        if backend == "hybrid":
            extraction_data = self._hybrid_cpf_validation(
                extraction_data, image, request["request_id"]
            )

        # CPF validation result
        cpf_validation = None
        if extraction_data.get("cpf"):
            cpf_validation = validate_cpf(extraction_data["cpf"])

        return {
            "extraction": {
                "document_type": doc_type,
                "data": extraction_data,
                "raw_text": None,
                "backend": "paneas_v2",
            },
            "cpf_validation": cpf_validation,
            # Backward compat for old extract() callers
            "result": extraction_data,
        }

    async def _handle_process_single(self, request: dict, image: Image.Image) -> dict:
        """Handle process (classify + extract) for a single request."""
        do_extract = request.get("extract", True)
        min_confidence = request.get("min_confidence") or settings.min_confidence
        backend = request.get("backend", "vlm")

        # Classify
        classification = self.classifier.classify(image)
        classification_dict = classification.model_dump()

        if not do_extract:
            return {
                "file_path": None,
                "classification": classification_dict,
                "extraction": None,
                "success": True,
                "error": None,
            }

        # Check confidence
        if classification.confidence < min_confidence:
            return {
                "file_path": None,
                "classification": classification_dict,
                "extraction": None,
                "success": False,
                "error": f"Confiança ({classification.confidence:.1%}) abaixo do mínimo",
            }

        # Extract via VLM
        doc_type_str = classification.document_type.value
        prompt = self._get_prompt(doc_type_str)
        raw_text = await self._vlm_generate(image, prompt)
        extraction_data = self._build_extraction_result(raw_text, doc_type_str)

        # Hybrid CPF validation
        if backend == "hybrid":
            extraction_data = self._hybrid_cpf_validation(
                extraction_data, image, request["request_id"]
            )

        # Build extraction result dict
        extraction_dict = {
            "document_type": doc_type_str,
            "data": extraction_data,
            "raw_text": None,
            "backend": "paneas_v2",
        }

        result = {
            "file_path": None,
            "classification": classification_dict,
            "extraction": extraction_dict,
            "success": True,
            "error": None,
        }

        # CPF validation
        if extraction_data.get("cpf"):
            result["cpf_validation"] = validate_cpf(extraction_data["cpf"])

        return result

    def _handle_ocr(self, request: dict) -> dict:
        """Handle OCR operation: convert PDF/image, orient, OCR each page."""
        file_path = request["image_path"]
        max_pages = request.get("max_pages", 10)
        auto_rotate = request.get("auto_rotate", True)
        request_id = request["request_id"]

        if is_pdf(file_path):
            images = self.pdf_converter.convert(file_path, max_pages=max_pages)
            file_type = "pdf"
        else:
            images = [Image.open(file_path).convert("RGB")]
            file_type = "image"

        pages = []
        for i, img in enumerate(images, start=1):
            if auto_rotate:
                result = self.orientation_corrector.correct(img)
                if result.was_corrected:
                    logger.info(
                        "page_orientation_corrected",
                        request_id=request_id,
                        page=i,
                        rotation=result.rotation_applied.value,
                        method=result.correction_method,
                    )
                img = result.image

            text, confidence = self.ocr_engine.extract_text(img)
            pages.append(
                {
                    "page": i,
                    "text": text,
                    "confidence": confidence,
                }
            )

        return {
            "request_id": request_id,
            "total_pages": len(pages),
            "pages": pages,
            "file_type": file_type,
            "language": settings.ocr_language,
        }

    # ------------------------------------------------------------------
    # Single request processing
    # ------------------------------------------------------------------

    async def _process_single(self, request: dict) -> None:
        """Process a single inference request (any operation)."""
        inference_id = request["inference_id"]
        reply_key = inference_reply_key(inference_id)
        operation = request.get("operation", "extract")
        start_time = time.perf_counter()

        logger.info(
            "inference_processing_start",
            inference_id=inference_id,
            request_id=request["request_id"],
            operation=operation,
            document_type=request.get("document_type"),
        )

        try:
            # OCR handles its own image loading (multi-page PDF)
            if operation == "ocr":
                result = self._handle_ocr(request)
            else:
                # Load image (PDF → first page for classify/extract/process)
                if is_pdf(request["image_path"]):
                    image = self._convert_pdf_first_page(request["image_path"])
                else:
                    image = self._load_image(request["image_path"])

                # Orientation correction
                auto_rotate = request.get("auto_rotate", True)
                image, correction_info = self._correct_orientation(image, auto_rotate)

                # Route to handler
                if operation == "classify":
                    result = self._handle_classify(request, image)
                elif operation == "extract":
                    result = await self._handle_extract_single(request, image)
                elif operation == "process":
                    result = await self._handle_process_single(request, image)
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                result["image_correction"] = correction_info

            inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            reply = {
                "inference_id": inference_id,
                "success": True,
                "result": result,
                "inference_time_ms": inference_time_ms,
                "error": None,
            }

            inference_metrics_requests_total.labels(operation=operation, status="success").inc()
            inference_metrics_duration_seconds.labels(operation=operation).observe(
                inference_time_ms / 1000
            )
            inference_metrics_batch_size.observe(1)

            logger.info(
                "inference_processing_complete",
                inference_id=inference_id,
                operation=operation,
                inference_time_ms=inference_time_ms,
            )

        except Exception as e:
            sentry_sdk.capture_exception(e)
            inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            reply = {
                "inference_id": inference_id,
                "success": False,
                "result": None,
                "inference_time_ms": inference_time_ms,
                "error": str(e),
            }
            inference_metrics_requests_total.labels(operation=operation, status="error").inc()
            logger.error(
                "inference_processing_error",
                inference_id=inference_id,
                operation=operation,
                error=str(e),
                error_type=type(e).__name__,
            )

        await self.queue.redis.setex(reply_key, INFERENCE_REPLY_TTL, json.dumps(reply))

    # ------------------------------------------------------------------
    # Batched VLM processing (extract/process only)
    # ------------------------------------------------------------------

    async def _process_vlm_batch(self, batch: list[dict]) -> None:
        """Process a batch of extract/process requests with batched VLM inference."""
        start_time = time.perf_counter()
        batch_size = len(batch)

        logger.info(
            "batch_inference_start",
            batch_size=batch_size,
            request_ids=[r["request_id"] for r in batch],
        )

        inference_metrics_batch_size.observe(batch_size)

        # Phase 1: Preprocess each request (load image, orient, classify if process)
        preprocessed = []  # (index, image, doc_type, classification_dict_or_None, skip_reason)
        errors = {}

        for i, req in enumerate(batch):
            try:
                operation = req.get("operation", "extract")

                # Load image
                if is_pdf(req["image_path"]):
                    image = self._convert_pdf_first_page(req["image_path"])
                else:
                    image = self._load_image(req["image_path"])

                # Orient
                auto_rotate = req.get("auto_rotate", True)
                image, correction_info = self._correct_orientation(image, auto_rotate)

                if operation == "process":
                    do_extract = req.get("extract", True)
                    min_confidence = req.get("min_confidence") or settings.min_confidence

                    classification = self.classifier.classify(image)
                    classification_dict = classification.model_dump()

                    if not do_extract or classification.confidence < min_confidence:
                        # No VLM needed — send reply directly
                        skip_result = {
                            "file_path": None,
                            "classification": classification_dict,
                            "extraction": None,
                            "image_correction": correction_info,
                            "success": not do_extract
                            or classification.confidence >= min_confidence,
                            "error": None
                            if not do_extract
                            else f"Confiança ({classification.confidence:.1%}) abaixo do mínimo",
                        }
                        preprocessed.append((i, None, None, skip_result, correction_info))
                        continue

                    doc_type = classification.document_type.value
                    preprocessed.append((i, image, doc_type, classification_dict, correction_info))
                else:
                    # extract operation
                    doc_type = req["document_type"]
                    preprocessed.append((i, image, doc_type, None, correction_info))

            except Exception as e:
                errors[i] = str(e)

        # Phase 2: Collect VLM-eligible items for batch inference
        vlm_items = []  # (preprocessed_index, image, prompt)
        for pp_idx, (_i, image, doc_type, _class_dict_or_skip, _corr) in enumerate(preprocessed):
            if image is None:
                continue  # skipped (no extract needed)
            prompt = self._get_prompt(doc_type)
            vlm_items.append((pp_idx, image, prompt))

        # Run batched VLM
        raw_texts = []
        if vlm_items:
            images_for_vlm = [item[1] for item in vlm_items]
            prompts_for_vlm = [item[2] for item in vlm_items]
            try:
                raw_texts = await self._vlm_generate_batch(images_for_vlm, prompts_for_vlm)
            except Exception as e:
                sentry_sdk.capture_exception(e)
                logger.error(
                    "batch_vlm_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    batch_size=len(vlm_items),
                )
                # Mark all VLM items as errors
                for pp_idx, _, _ in vlm_items:
                    i = preprocessed[pp_idx][0]
                    errors[i] = str(e)
                raw_texts = []

        batch_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "batch_inference_complete",
            batch_size=batch_size,
            vlm_count=len(vlm_items),
            error_count=len(errors),
            inference_time_ms=batch_time_ms,
        )

        # Phase 3: Build replies
        vlm_text_idx = 0
        for _pp_idx, (i, image, doc_type, class_or_skip, correction_info) in enumerate(
            preprocessed
        ):
            req = batch[i]
            inference_id = req["inference_id"]
            reply_key = inference_reply_key(inference_id)
            operation = req.get("operation", "extract")

            if i in errors:
                reply = {
                    "inference_id": inference_id,
                    "success": False,
                    "result": None,
                    "inference_time_ms": batch_time_ms,
                    "error": errors[i],
                }
                inference_metrics_requests_total.labels(operation=operation, status="error").inc()
            elif image is None:
                # Skipped (process with low confidence / no extract)
                reply = {
                    "inference_id": inference_id,
                    "success": True,
                    "result": class_or_skip,  # pre-built skip result
                    "inference_time_ms": batch_time_ms,
                    "error": None,
                }
                inference_metrics_requests_total.labels(operation=operation, status="success").inc()
            else:
                # Has VLM result
                try:
                    raw_text = raw_texts[vlm_text_idx]
                    vlm_text_idx += 1
                    extraction_data = self._build_extraction_result(raw_text, doc_type)

                    backend = req.get("backend", "vlm")
                    if backend == "hybrid":
                        extraction_data = self._hybrid_cpf_validation(
                            extraction_data, image, req["request_id"]
                        )

                    if operation == "process":
                        result = {
                            "file_path": None,
                            "classification": class_or_skip,  # classification_dict
                            "extraction": {
                                "document_type": doc_type,
                                "data": extraction_data,
                                "raw_text": None,
                                "backend": "paneas_v2",
                            },
                            "image_correction": correction_info,
                            "success": True,
                            "error": None,
                        }
                        if extraction_data.get("cpf"):
                            result["cpf_validation"] = validate_cpf(extraction_data["cpf"])
                    else:
                        # extract
                        result = {
                            "extraction": {
                                "document_type": doc_type,
                                "data": extraction_data,
                                "raw_text": None,
                                "backend": "paneas_v2",
                            },
                            "image_correction": correction_info,
                            "result": extraction_data,  # backward compat
                        }
                        if extraction_data.get("cpf"):
                            result["cpf_validation"] = validate_cpf(extraction_data["cpf"])

                    reply = {
                        "inference_id": inference_id,
                        "success": True,
                        "result": result,
                        "inference_time_ms": batch_time_ms,
                        "error": None,
                    }
                    inference_metrics_requests_total.labels(
                        operation=operation, status="success"
                    ).inc()
                    inference_metrics_duration_seconds.labels(operation=operation).observe(
                        batch_time_ms / 1000
                    )
                except Exception as e:
                    vlm_text_idx += 1
                    reply = {
                        "inference_id": inference_id,
                        "success": False,
                        "result": None,
                        "inference_time_ms": batch_time_ms,
                        "error": str(e),
                    }
                    inference_metrics_requests_total.labels(
                        operation=operation, status="error"
                    ).inc()

            await self.queue.redis.setex(reply_key, INFERENCE_REPLY_TTL, json.dumps(reply))

        # Reply for requests that errored in preprocessing (not in preprocessed)
        for i, error_msg in errors.items():
            # Only handle if not already handled via preprocessed loop
            already_handled = any(pp[0] == i for pp in preprocessed)
            if already_handled:
                continue

            req = batch[i]
            inference_id = req["inference_id"]
            reply_key = inference_reply_key(inference_id)
            operation = req.get("operation", "extract")
            reply = {
                "inference_id": inference_id,
                "success": False,
                "result": None,
                "inference_time_ms": batch_time_ms,
                "error": error_msg,
            }
            inference_metrics_requests_total.labels(operation=operation, status="error").inc()
            await self.queue.redis.setex(reply_key, INFERENCE_REPLY_TTL, json.dumps(reply))


# Global server instance
server: InferenceServer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage inference server lifecycle."""
    global server
    server = InferenceServer()
    await server.start()

    # Start inference loop in background
    server_task = asyncio.create_task(server.run())

    yield

    # Shutdown
    await server.stop()
    server_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await server_task


# Health check app
app = FastAPI(
    title="doc-pipeline Inference Server",
    description="Centralized inference server for all GPU models",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Inference server health check."""
    if server is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Server not initialized"},
        )

    queue_depth = await server.queue.redis.llen(QueueName.INFERENCE)

    vlm_backend = "vllm" if server.vllm_client else "huggingface"
    if server.vllm_client:
        vlm_ready = await server.vllm_client.health_check()
    else:
        vlm_ready = server.extractor is not None and server.extractor._model is not None

    return {
        "status": "ok",
        "server_running": server._running,
        "current_batch_size": server._current_batch_size,
        "queue_depth": queue_depth,
        "vlm_backend": vlm_backend,
        "models_loaded": {
            "vlm": vlm_ready,
            "classifier": server.classifier is not None,
            "ocr": server.ocr_engine is not None,
            "orientation": server.orientation_corrector is not None,
        },
        "config": {
            "batch_size": server._batch_size,
            "batch_timeout_ms": server._batch_timeout_ms,
        },
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def main():
    """Run the inference server."""
    import uvicorn

    settings = get_settings()

    # Handle signals
    def signal_handler(signum, frame):
        logger.info("signal_received", signal=signum)
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    port = settings.inference_server_health_port

    logger.info(
        "inference_server_start",
        host="0.0.0.0",
        port=port,
        model=settings.extractor_model_qwen,
        device=settings.extractor_device,
        batch_size=settings.inference_batch_size,
        batch_timeout_ms=settings.inference_batch_timeout_ms,
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
