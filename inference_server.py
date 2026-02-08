#!/usr/bin/env python3
"""
Inference server for VLM (Qwen) extraction.

Runs as a standalone process with GPU access. Workers send inference
requests via Redis queue, this server processes them and replies via Pub/Sub.

Supports batched inference: collects multiple requests and processes them
in a single GPU forward pass for higher throughput.
"""

import asyncio
import contextlib
import json
import signal
import sys
import time
from contextlib import asynccontextmanager

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

from doc_pipeline.config import get_settings
from doc_pipeline.extractors.qwen_vl import QwenVLExtractor
from doc_pipeline.observability import get_logger, setup_logging
from doc_pipeline.prompts import CIN_EXTRACTION_PROMPT, CNH_EXTRACTION_PROMPT, RG_EXTRACTION_PROMPT
from doc_pipeline.schemas import CINData, CNHData, RGData
from doc_pipeline.shared.constants import INFERENCE_REPLY_TTL, QueueName, inference_reply_key
from doc_pipeline.shared.queue import QueueService, get_queue_service
from doc_pipeline.utils import fix_cpf_rg_swap

# Setup logging
settings = get_settings()
setup_logging(
    json_format=settings.log_json,
    log_level=settings.log_level,
)

logger = get_logger("inference_server")


# Prometheus metrics for inference server
inference_metrics_requests_total = Counter(
    "inference_requests_total",
    "Total inference requests processed",
    ["document_type", "status"],
)
inference_metrics_duration_seconds = Histogram(
    "inference_duration_seconds",
    "Time spent on VLM inference",
    ["document_type"],
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0),
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
    """Server that processes VLM inference requests from Redis queue with batching."""

    def __init__(self):
        self.queue: QueueService = get_queue_service()
        self.extractor: QwenVLExtractor | None = None
        self._running = False
        self._current_batch_size: int = 0
        self._batch_size = settings.inference_batch_size
        self._batch_timeout_ms = settings.inference_batch_timeout_ms

    async def start(self) -> None:
        """Initialize and start the inference server."""
        logger.info("inference_server_starting")

        # Connect to Redis
        await self.queue.connect()

        # Load VLM model
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

        if self.extractor:
            self.extractor.unload_model()

        await self.queue.close()
        logger.info("inference_server_stopped")

    async def _collect_batch(self) -> list[dict]:
        """Collect a batch of requests from the queue.

        Waits for the first request (blocking), then grabs more
        non-blocking up to batch_size or batch_timeout.
        """
        # Wait for first request (blocking)
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
            # Non-blocking pop
            extra = await self.queue.redis.rpop(QueueName.INFERENCE)
            if extra is None:
                # Queue empty, wait a tiny bit for more to arrive
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
                # Update queue depth metric
                depth = await self.queue.redis.llen(QueueName.INFERENCE)
                inference_metrics_queue_depth.set(depth)

                # Collect batch
                batch = await self._collect_batch()
                if not batch:
                    continue

                self._current_batch_size = len(batch)

                if len(batch) == 1:
                    await self._process_single(batch[0])
                else:
                    await self._process_batch(batch)

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

    def _build_result(self, raw_text: str, doc_type: str) -> dict:
        """Parse VLM response and build structured result."""
        data = self.extractor._parse_json(raw_text)
        data = fix_cpf_rg_swap(data)

        if doc_type.startswith("rg"):
            model = RGData(**{k: v for k, v in data.items() if k in RGData.model_fields})
        elif doc_type.startswith("cin"):
            model = CINData(**{k: v for k, v in data.items() if k in CINData.model_fields})
        else:
            model = CNHData(**{k: v for k, v in data.items() if k in CNHData.model_fields})

        return model.model_dump()

    async def _process_single(self, request: dict) -> None:
        """Process a single inference request (no batching)."""
        inference_id = request["inference_id"]
        reply_key = inference_reply_key(inference_id)
        doc_type = request["document_type"]
        start_time = time.perf_counter()

        logger.info(
            "inference_processing_start",
            inference_id=inference_id,
            request_id=request["request_id"],
            document_type=doc_type,
            batch_size=1,
        )

        try:
            image = Image.open(request["image_path"]).convert("RGB")
            prompt = self._get_prompt(doc_type)
            raw_text = self.extractor._generate(image, prompt)
            result = self._build_result(raw_text, doc_type)

            inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            reply = {
                "inference_id": inference_id,
                "success": True,
                "result": result,
                "inference_time_ms": inference_time_ms,
                "error": None,
            }

            inference_metrics_requests_total.labels(
                document_type=doc_type, status="success"
            ).inc()
            inference_metrics_duration_seconds.labels(document_type=doc_type).observe(
                inference_time_ms / 1000
            )
            inference_metrics_batch_size.observe(1)

            logger.info(
                "inference_processing_complete",
                inference_id=inference_id,
                document_type=doc_type,
                inference_time_ms=inference_time_ms,
                batch_size=1,
            )

        except Exception as e:
            inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            reply = {
                "inference_id": inference_id,
                "success": False,
                "result": None,
                "inference_time_ms": inference_time_ms,
                "error": str(e),
            }
            inference_metrics_requests_total.labels(
                document_type=doc_type, status="error"
            ).inc()
            logger.error(
                "inference_processing_error",
                inference_id=inference_id,
                error=str(e),
                error_type=type(e).__name__,
            )

        await self.queue.redis.setex(reply_key, INFERENCE_REPLY_TTL, json.dumps(reply))

    async def _process_batch(self, batch: list[dict]) -> None:
        """Process a batch of inference requests in a single forward pass."""
        start_time = time.perf_counter()
        batch_size = len(batch)

        logger.info(
            "batch_inference_start",
            batch_size=batch_size,
            request_ids=[r["request_id"] for r in batch],
        )

        inference_metrics_batch_size.observe(batch_size)

        # Load all images and determine prompts
        images = []
        prompts = []
        valid_indices = []
        errors = {}

        for i, request in enumerate(batch):
            try:
                image = Image.open(request["image_path"]).convert("RGB")
                prompt = self._get_prompt(request["document_type"])
                images.append(image)
                prompts.append(prompt)
                valid_indices.append(i)
            except Exception as e:
                errors[i] = str(e)

        # Run batched inference for all valid requests
        raw_texts = []
        if images:
            try:
                raw_texts = self.extractor._generate_batch(images, prompts)
            except Exception as e:
                # If batch inference fails, mark all as errors
                logger.error(
                    "batch_inference_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    batch_size=len(images),
                )
                for i in valid_indices:
                    errors[i] = str(e)
                raw_texts = []

        batch_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "batch_inference_complete",
            batch_size=batch_size,
            valid_count=len(valid_indices),
            error_count=len(errors),
            inference_time_ms=batch_time_ms,
            ms_per_item=round(batch_time_ms / batch_size, 1),
        )

        # Build replies and publish
        text_idx = 0
        for i, request in enumerate(batch):
            inference_id = request["inference_id"]
            reply_key = inference_reply_key(inference_id)
            doc_type = request["document_type"]

            if i in errors:
                reply = {
                    "inference_id": inference_id,
                    "success": False,
                    "result": None,
                    "inference_time_ms": batch_time_ms,
                    "error": errors[i],
                }
                inference_metrics_requests_total.labels(
                    document_type=doc_type, status="error"
                ).inc()
            else:
                try:
                    result = self._build_result(raw_texts[text_idx], doc_type)
                    reply = {
                        "inference_id": inference_id,
                        "success": True,
                        "result": result,
                        "inference_time_ms": batch_time_ms,
                        "error": None,
                    }
                    inference_metrics_requests_total.labels(
                        document_type=doc_type, status="success"
                    ).inc()
                    inference_metrics_duration_seconds.labels(
                        document_type=doc_type
                    ).observe(batch_time_ms / 1000)
                except Exception as e:
                    reply = {
                        "inference_id": inference_id,
                        "success": False,
                        "result": None,
                        "inference_time_ms": batch_time_ms,
                        "error": str(e),
                    }
                    inference_metrics_requests_total.labels(
                        document_type=doc_type, status="error"
                    ).inc()
                text_idx += 1

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
    description="VLM inference server health check",
    version="0.1.0",
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

    return {
        "status": "ok",
        "server_running": server._running,
        "current_batch_size": server._current_batch_size,
        "queue_depth": queue_depth,
        "model_loaded": server.extractor is not None
        and server.extractor._model is not None,
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
