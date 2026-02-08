#!/usr/bin/env python3
"""
API REST para o pipeline de classificação e extração de documentos.

Esta API enfileira jobs no Redis para processamento pelo worker (GPU).
Suporta modo sync (espera resultado) e webhook (retorno assíncrono).
"""

import asyncio
import io
import json
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Annotated

import redis.exceptions
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from doc_pipeline import PipelineResult
from doc_pipeline.auth import AuthInfo, require_api_key
from doc_pipeline.config import get_settings
from doc_pipeline.observability import (
    PrometheusMiddleware,
    get_logger,
    get_metrics,
    metrics_endpoint,
    setup_logging,
)
from doc_pipeline.observability.worker_metrics import get_aggregated_worker_metrics
from doc_pipeline.schemas import (
    ClassificationResult,
    DocumentType,
    ExtractionResult,
    OCRResult,
)
from doc_pipeline.shared import JobContext, QueueService, get_queue_service
from doc_pipeline.shared.constants import QueueName
from doc_pipeline.shared.queue import QueueFullError

# Setup logging antes de tudo
settings = get_settings()
setup_logging(
    json_format=settings.log_json,
    log_level=settings.log_level,
)

logger = get_logger("api")

# Global queue service
queue_service: QueueService | None = None

# Rate limiter - key by client IP or API key
def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key from API key header or IP address."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return f"apikey:{api_key[:8]}"  # Use first 8 chars of API key
    return get_remote_address(request)

# Create limiter — only use Redis backend when rate limiting is actually enabled.
# When disabled, use memory:// to avoid creating a separate Redis connection pool
# that can exhaust connections under high concurrency.
if settings.rate_limit_enabled:
    limiter = Limiter(
        key_func=get_rate_limit_key,
        storage_uri=settings.redis_url,
        storage_options={"max_connections": settings.redis_max_connections},
        swallow_errors=True,
        in_memory_fallback_enabled=True,
        enabled=True,
    )
else:
    limiter = Limiter(
        key_func=get_rate_limit_key,
        storage_uri="memory://",
        enabled=False,
    )


class DeliveryMode(str, Enum):
    """Delivery modes for job results."""

    SYNC = "sync"
    WEBHOOK = "webhook"


class ExtractorBackend(str, Enum):
    """
    Extractor backends for document data extraction.

    Tradeoffs:
    - **hybrid** (default): EasyOCR + VLM. Melhor precisão em CPF/números (~15s).
      Combina OCR preciso para dígitos com VLM para estruturar campos.
    - **vlm**: Apenas Qwen VL. Mais rápido (~5s) mas pode confundir dígitos (4↔6).
      Bom para documentos com boa qualidade de imagem.
    - **ocr**: Apenas EasyOCR. Mais rápido (~2s), retorna texto bruto.
      Útil quando só precisa do texto sem estruturação.
    """

    HYBRID = "hybrid"
    VLM = "vlm"
    OCR = "ocr"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida da aplicação."""
    global queue_service

    settings = get_settings()
    logger.info(
        "startup",
        redis_url=settings.redis_url.split("@")[-1],  # Hide password
        auth_env_keys=len(settings.api_keys_list),
        auth_database=bool(settings.database_url),
    )

    # Connect to Redis
    queue_service = get_queue_service()
    await queue_service.connect()

    logger.info("startup_complete")
    yield

    # Cleanup
    logger.info("shutdown_start")
    if queue_service:
        await queue_service.close()
    logger.info("shutdown_complete")


app = FastAPI(
    title="doc-pipeline API",
    description="API para classificação e extração de dados de documentos brasileiros (RG/CNH)",
    version="0.2.0",
    lifespan=lifespan,
)

# Rate limiting setup
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Adiciona middleware de métricas (api_only=True filtra apenas endpoints da API)
app.add_middleware(PrometheusMiddleware)


# Response models
class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    queue_depth: int


class QueuedResponse(BaseModel):
    """Response for async (webhook) mode."""

    request_id: str
    status: str
    message: str
    correlation_id: str | None = None


class ClassesResponse(BaseModel):
    classes: list[str]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


async def save_temp_image(upload_file: UploadFile) -> str:
    """Save uploaded file to temp location and return path."""
    # Read file content
    contents = await upload_file.read()

    # Validate it's a valid image
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception as e:
        raise ValueError(f"Invalid image file: {e}")

    # Generate unique filename
    ext = os.path.splitext(upload_file.filename or "image.jpg")[1] or ".jpg"
    filename = f"{uuid.uuid4()}{ext}"

    # Save to temp directory
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "doc-pipeline", filename)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(contents)

    return temp_path


async def save_temp_file(upload_file: UploadFile, allowed_extensions: set[str]) -> str:
    """Save uploaded file (PDF or image) to temp location and return path."""
    # Read file content
    contents = await upload_file.read()

    # Get extension
    ext = os.path.splitext(upload_file.filename or "file")[1].lower() or ""

    # Validate extension
    if ext not in allowed_extensions:
        raise ValueError(f"Invalid file type: {ext}. Allowed: {allowed_extensions}")

    # Validate file based on type
    if ext == ".pdf":
        # Check PDF magic bytes
        if not contents.startswith(b"%PDF"):
            raise ValueError("Invalid PDF file")
    else:
        # Validate it's a valid image
        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")

    # Generate unique filename
    filename = f"{uuid.uuid4()}{ext}"

    # Save to temp directory
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "doc-pipeline", filename)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)

    with open(temp_path, "wb") as f:
        f.write(contents)

    return temp_path


async def wait_for_result(request_id: str, timeout: float) -> dict:
    """Wait for job result via Redis polling."""
    if queue_service is None:
        raise RuntimeError("Queue service not connected")

    poll_interval = 0.3  # 300ms between polls
    end_time = asyncio.get_event_loop().time() + timeout

    while True:
        try:
            cached = await queue_service.get_cached_result(request_id)
        except Exception as e:
            logger.error("poll_redis_error", error=str(e), error_type=type(e).__name__, request_id=request_id)
            raise RuntimeError(str(e)) from e
        if cached:
            return json.loads(cached)

        remaining = end_time - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError()

        await asyncio.sleep(min(poll_interval, remaining))


@app.get("/metrics")
async def metrics():
    """Endpoint para Prometheus scraping.

    Includes:
    - API metrics (requests, latency, etc.)
    - Autoscaler metrics (from file)
    - Aggregated worker metrics (from Redis)
    """
    from fastapi.responses import PlainTextResponse
    from prometheus_client import CONTENT_TYPE_LATEST

    # Get base metrics (API + autoscaler)
    base_response = metrics_endpoint()
    content = base_response.body

    # Add aggregated worker metrics from Redis
    if queue_service and queue_service._redis:
        try:
            worker_metrics = await get_aggregated_worker_metrics(queue_service._redis)
            if worker_metrics:
                content = content + b"\n# Worker metrics (aggregated from Redis)\n"
                content = content + worker_metrics.encode()
        except Exception:
            pass  # Don't fail metrics endpoint if worker aggregation fails

    return PlainTextResponse(
        content=content,
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Verifica status da API e Redis."""
    if queue_service is None:
        return HealthResponse(
            status="degraded",
            redis_connected=False,
            queue_depth=0,
        )

    try:
        depth = await queue_service.get_queue_depth()
        return HealthResponse(
            status="ok",
            redis_connected=True,
            queue_depth=depth,
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            redis_connected=False,
            queue_depth=0,
        )


@app.get("/classes", response_model=ClassesResponse)
async def list_classes(auth: AuthInfo = Depends(require_api_key)):
    """Lista classes de documentos suportadas."""
    return ClassesResponse(classes=[d.value for d in DocumentType])


@app.post("/classify")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}")
async def classify(
    request: Request,
    arquivo: Annotated[UploadFile, File(description="Imagem do documento")],
    delivery_mode: Annotated[DeliveryMode, Query(description="Modo de entrega")] = DeliveryMode.SYNC,
    webhook_url: Annotated[str | None, Query(description="URL para webhook (se modo=webhook)")] = None,
    correlation_id: Annotated[str | None, Query(description="ID de correlação")] = None,
    auth: AuthInfo = Depends(require_api_key),
):
    """
    Classifica uma imagem de documento.

    Retorna o tipo do documento e a confiança da classificação.
    """
    if queue_service is None:
        raise HTTPException(status_code=503, detail="Queue service not connected")

    metrics = get_metrics()
    client = auth.client_name or "unknown"

    # Validate webhook mode
    if delivery_mode == DeliveryMode.WEBHOOK and not webhook_url:
        raise HTTPException(
            status_code=400,
            detail="webhook_url is required when delivery_mode=webhook",
        )

    try:
        # Save image to temp file
        image_path = await save_temp_image(arquivo)

        # Create job
        job = JobContext.create(
            image_path=image_path,
            operation="classify",
            client_name=auth.client_name,
            api_key_prefix=auth.api_key_prefix,
            delivery_mode=delivery_mode.value,
            webhook_url=webhook_url,
            correlation_id=correlation_id,
        )

        # Enqueue
        await queue_service.enqueue(job)

        metrics.requests_by_client.labels(
            client=client,
            endpoint="/classify",
            status="202" if delivery_mode == DeliveryMode.WEBHOOK else "pending",
        ).inc()

        logger.info(
            "classify_enqueued",
            request_id=job.request_id,
            delivery_mode=delivery_mode.value,
            filename=arquivo.filename,
            client=client,
        )

        if delivery_mode == DeliveryMode.WEBHOOK:
            return JSONResponse(
                status_code=202,
                content=QueuedResponse(
                    request_id=job.request_id,
                    status="queued",
                    message="Job queued for processing. Result will be POSTed to webhook_url.",
                    correlation_id=correlation_id,
                ).model_dump(),
            )

        # Sync mode - wait for result
        settings = get_settings()
        try:
            result = await wait_for_result(job.request_id, settings.sync_timeout_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Timeout waiting for result after {settings.sync_timeout_seconds}s",
            )

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        metrics.requests_by_client.labels(
            client=client,
            endpoint="/classify",
            status="200",
        ).inc()

        # Return the actual classification result
        return ClassificationResult(**result["result"])

    except QueueFullError as e:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/classify",
            status="503",
        ).inc()
        logger.warning("classify_queue_full", error=str(e), client=client)
        raise HTTPException(status_code=503, detail=str(e))

    except HTTPException:
        raise

    except (ConnectionError, redis.exceptions.ConnectionError) as e:
        metrics.requests_by_client.labels(client=client, endpoint="/classify", status="503").inc()
        logger.error("classify_connection_error", error=str(e), client=client)
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/classify",
            status="500",
        ).inc()
        logger.error(
            "classify_error",
            error=str(e),
            error_type=type(e).__name__,
            filename=arquivo.filename,
            client=client,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}")
async def extract(
    request: Request,
    arquivo: Annotated[UploadFile, File(description="Imagem do documento")],
    doc_type: Annotated[str, Query(description="Tipo do documento (rg_frente, cnh_frente, etc)")],
    backend: Annotated[ExtractorBackend, Query(description="Backend de extração (hybrid=default, vlm, ocr)")] = ExtractorBackend.HYBRID,
    validate_cpf: Annotated[bool, Query(description="Validar CPF extraído (dígitos verificadores)")] = True,
    auto_rotate: Annotated[bool, Query(description="Corrigir orientação automaticamente")] = True,
    delivery_mode: Annotated[DeliveryMode, Query(description="Modo de entrega")] = DeliveryMode.SYNC,
    webhook_url: Annotated[str | None, Query(description="URL para webhook (se modo=webhook)")] = None,
    correlation_id: Annotated[str | None, Query(description="ID de correlação")] = None,
    auth: AuthInfo = Depends(require_api_key),
):
    """
    Extrai dados de uma imagem de documento.

    Requer que o tipo do documento seja especificado (rg_frente, cnh_frente, etc).

    ## Backends de Extração

    | Backend | Tempo | Precisão CPF | Uso |
    |---------|-------|--------------|-----|
    | **hybrid** | ~15s | ✅ Alta | Padrão. Melhor para documentos com números críticos |
    | **vlm** | ~5s | ⚠️ Média | Quando velocidade é prioridade |
    | **ocr** | ~2s | ✅ Alta | Apenas texto bruto, sem estruturação |

    ## Correção de Orientação

    Com `auto_rotate=True` (padrão), imagens rotacionadas são corrigidas automaticamente.
    Use `auto_rotate=False` se a imagem já está na orientação correta (economiza ~4s).
    """
    if queue_service is None:
        raise HTTPException(status_code=503, detail="Queue service not connected")

    metrics = get_metrics()
    client = auth.client_name or "unknown"

    # Validate document type (map legacy "cni_" prefix to "cin_")
    if doc_type and doc_type.startswith("cni_"):
        doc_type = doc_type.replace("cni_", "cin_", 1)
    try:
        document_type = DocumentType(doc_type)
    except ValueError:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/extract",
            status="400",
        ).inc()
        logger.warning(
            "extract_invalid_type",
            doc_type=doc_type,
            filename=arquivo.filename,
            client=client,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de documento inválido: {doc_type}. Valores válidos: {[d.value for d in DocumentType]}",
        )

    # Validate webhook mode
    if delivery_mode == DeliveryMode.WEBHOOK and not webhook_url:
        raise HTTPException(
            status_code=400,
            detail="webhook_url is required when delivery_mode=webhook",
        )

    try:
        # Save image to temp file
        image_path = await save_temp_image(arquivo)

        # Create job
        job = JobContext.create(
            image_path=image_path,
            operation="extract",
            document_type=document_type.value,
            client_name=auth.client_name,
            api_key_prefix=auth.api_key_prefix,
            delivery_mode=delivery_mode.value,
            webhook_url=webhook_url,
            correlation_id=correlation_id,
            extra_params={"backend": backend.value, "validate_cpf": validate_cpf, "auto_rotate": auto_rotate},
        )

        # Enqueue
        await queue_service.enqueue(job)

        logger.info(
            "extract_enqueued",
            request_id=job.request_id,
            doc_type=doc_type,
            backend=backend.value,
            auto_rotate=auto_rotate,
            delivery_mode=delivery_mode.value,
            filename=arquivo.filename,
            client=client,
        )

        if delivery_mode == DeliveryMode.WEBHOOK:
            return JSONResponse(
                status_code=202,
                content=QueuedResponse(
                    request_id=job.request_id,
                    status="queued",
                    message="Job queued for processing. Result will be POSTed to webhook_url.",
                    correlation_id=correlation_id,
                ).model_dump(),
            )

        # Sync mode - wait for result
        settings = get_settings()
        try:
            result = await wait_for_result(job.request_id, settings.sync_timeout_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Timeout waiting for result after {settings.sync_timeout_seconds}s",
            )

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        metrics.requests_by_client.labels(
            client=client,
            endpoint="/extract",
            status="200",
        ).inc()

        return ExtractionResult(**result["result"])

    except QueueFullError as e:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/extract",
            status="503",
        ).inc()
        logger.warning("extract_queue_full", error=str(e), client=client)
        raise HTTPException(status_code=503, detail=str(e))

    except HTTPException:
        raise

    except (ConnectionError, redis.exceptions.ConnectionError) as e:
        metrics.requests_by_client.labels(client=client, endpoint="/extract", status="503").inc()
        logger.error("extract_connection_error", error=str(e), client=client)
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/extract",
            status="500",
        ).inc()
        logger.error(
            "extract_error",
            error=str(e),
            error_type=type(e).__name__,
            doc_type=doc_type,
            filename=arquivo.filename,
            client=client,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}")
async def process(
    request: Request,
    arquivo: Annotated[UploadFile, File(description="Imagem do documento")],
    extract: Annotated[bool, Query(description="Se deve extrair dados")] = True,
    min_confidence: Annotated[float, Query(ge=0.0, le=1.0, description="Confiança mínima")] = 0.5,
    backend: Annotated[ExtractorBackend, Query(description="Backend de extração (hybrid=default, vlm, ocr)")] = ExtractorBackend.HYBRID,
    validate_cpf: Annotated[bool, Query(description="Validar CPF extraído (dígitos verificadores)")] = True,
    auto_rotate: Annotated[bool, Query(description="Corrigir orientação automaticamente")] = True,
    delivery_mode: Annotated[DeliveryMode, Query(description="Modo de entrega")] = DeliveryMode.SYNC,
    webhook_url: Annotated[str | None, Query(description="URL para webhook (se modo=webhook)")] = None,
    correlation_id: Annotated[str | None, Query(description="ID de correlação")] = None,
    auth: AuthInfo = Depends(require_api_key),
):
    """
    Pipeline completo: classifica e extrai dados de uma imagem de documento.

    Este é o endpoint principal que executa classificação e extração em sequência.

    ## Backends de Extração

    | Backend | Tempo | Precisão CPF | Uso |
    |---------|-------|--------------|-----|
    | **hybrid** | ~15s | ✅ Alta | Padrão. Melhor para documentos com números críticos |
    | **vlm** | ~5s | ⚠️ Média | Quando velocidade é prioridade |
    | **ocr** | ~2s | ✅ Alta | Apenas texto bruto, sem estruturação |

    ## Correção de Orientação

    Com `auto_rotate=True` (padrão), imagens rotacionadas são corrigidas automaticamente.
    Isso adiciona ~4s ao processamento. Use `auto_rotate=False` se a imagem já está correta.
    """
    if queue_service is None:
        raise HTTPException(status_code=503, detail="Queue service not connected")

    metrics = get_metrics()
    client = auth.client_name or "unknown"

    # Validate webhook mode
    if delivery_mode == DeliveryMode.WEBHOOK and not webhook_url:
        raise HTTPException(
            status_code=400,
            detail="webhook_url is required when delivery_mode=webhook",
        )

    try:
        # Save image to temp file
        image_path = await save_temp_image(arquivo)

        # Create job
        job = JobContext.create(
            image_path=image_path,
            operation="process",
            extract=extract,
            min_confidence=min_confidence,
            client_name=auth.client_name,
            api_key_prefix=auth.api_key_prefix,
            delivery_mode=delivery_mode.value,
            webhook_url=webhook_url,
            correlation_id=correlation_id,
            extra_params={"backend": backend.value, "validate_cpf": validate_cpf, "auto_rotate": auto_rotate},
        )

        # Enqueue
        await queue_service.enqueue(job)

        logger.info(
            "process_enqueued",
            request_id=job.request_id,
            extract=extract,
            backend=backend.value,
            auto_rotate=auto_rotate,
            delivery_mode=delivery_mode.value,
            filename=arquivo.filename,
            client=client,
        )

        if delivery_mode == DeliveryMode.WEBHOOK:
            return JSONResponse(
                status_code=202,
                content=QueuedResponse(
                    request_id=job.request_id,
                    status="queued",
                    message="Job queued for processing. Result will be POSTed to webhook_url.",
                    correlation_id=correlation_id,
                ).model_dump(),
            )

        # Sync mode - wait for result
        settings = get_settings()
        try:
            result = await wait_for_result(job.request_id, settings.sync_timeout_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Timeout waiting for result after {settings.sync_timeout_seconds}s",
            )

        if result.get("error") or result.get("result") is None:
            error_detail = result.get("error") or "Processing failed (no result)"
            raise HTTPException(status_code=500, detail=error_detail)

        metrics.requests_by_client.labels(
            client=client,
            endpoint="/process",
            status="200",
        ).inc()

        return PipelineResult(**result["result"])

    except QueueFullError as e:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/process",
            status="503",
        ).inc()
        logger.warning("process_queue_full", error=str(e), client=client)
        raise HTTPException(status_code=503, detail=str(e))

    except HTTPException:
        raise

    except (ConnectionError, redis.exceptions.ConnectionError) as e:
        metrics.requests_by_client.labels(client=client, endpoint="/process", status="503").inc()
        logger.error("process_connection_error", error=str(e), client=client)
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/process",
            status="500",
        ).inc()
        logger.error(
            "process_error",
            error=str(e),
            error_type=type(e).__name__,
            filename=arquivo.filename,
            client=client,
        )
        raise HTTPException(status_code=500, detail=str(e))


# OCR allowed file extensions
OCR_ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


@app.post("/ocr")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}")
async def ocr(
    request: Request,
    arquivo: Annotated[UploadFile, File(description="PDF ou imagem para OCR")],
    max_pages: Annotated[int, Query(ge=1, le=50, description="Máximo de páginas (PDF)")] = 10,
    delivery_mode: Annotated[DeliveryMode, Query(description="Modo de entrega")] = DeliveryMode.SYNC,
    webhook_url: Annotated[str | None, Query(description="URL para webhook (se modo=webhook)")] = None,
    correlation_id: Annotated[str | None, Query(description="ID de correlação")] = None,
    auth: AuthInfo = Depends(require_api_key),
):
    """
    Extrai texto de PDF ou imagem usando OCR (PaddleOCR).

    Este endpoint aceita:
    - PDFs (múltiplas páginas)
    - Imagens (JPEG, PNG, TIFF, BMP, WebP)

    Retorna o texto extraído de cada página.
    """
    if queue_service is None:
        raise HTTPException(status_code=503, detail="Queue service not connected")

    metrics = get_metrics()
    client = auth.client_name or "unknown"

    # Validate webhook mode
    if delivery_mode == DeliveryMode.WEBHOOK and not webhook_url:
        raise HTTPException(
            status_code=400,
            detail="webhook_url is required when delivery_mode=webhook",
        )

    try:
        # Save file to temp location
        file_path = await save_temp_file(arquivo, OCR_ALLOWED_EXTENSIONS)

        # Create job for OCR queue
        job = JobContext.create(
            image_path=file_path,
            operation="ocr",
            client_name=auth.client_name,
            api_key_prefix=auth.api_key_prefix,
            delivery_mode=delivery_mode.value,
            webhook_url=webhook_url,
            correlation_id=correlation_id,
            extra_params={"max_pages": max_pages},
        )

        # Enqueue to OCR queue (separate from main queue)
        await queue_service.enqueue(job, queue_name=QueueName.OCR)

        logger.info(
            "ocr_enqueued",
            request_id=job.request_id,
            max_pages=max_pages,
            delivery_mode=delivery_mode.value,
            filename=arquivo.filename,
            client=client,
        )

        if delivery_mode == DeliveryMode.WEBHOOK:
            return JSONResponse(
                status_code=202,
                content=QueuedResponse(
                    request_id=job.request_id,
                    status="queued",
                    message="OCR job queued. Result will be POSTed to webhook_url.",
                    correlation_id=correlation_id,
                ).model_dump(),
            )

        # Sync mode - wait for result
        settings = get_settings()
        try:
            result = await wait_for_result(job.request_id, settings.sync_timeout_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Timeout waiting for OCR result after {settings.sync_timeout_seconds}s",
            )

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        metrics.requests_by_client.labels(
            client=client,
            endpoint="/ocr",
            status="200",
        ).inc()

        return OCRResult(**result["result"])

    except QueueFullError as e:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/ocr",
            status="503",
        ).inc()
        logger.warning("ocr_queue_full", error=str(e), client=client)
        raise HTTPException(status_code=503, detail=str(e))

    except HTTPException:
        raise

    except (ConnectionError, redis.exceptions.ConnectionError) as e:
        metrics.requests_by_client.labels(client=client, endpoint="/ocr", status="503").inc()
        logger.error("ocr_connection_error", error=str(e), client=client)
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/ocr",
            status="500",
        ).inc()
        logger.error(
            "ocr_error",
            error=str(e),
            error_type=type(e).__name__,
            filename=arquivo.filename,
            client=client,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{request_id}/status")
async def get_job_status(
    request_id: str,
    auth: AuthInfo = Depends(require_api_key),
):
    """Get the status of a queued job."""
    if queue_service is None:
        raise HTTPException(status_code=503, detail="Queue service not connected")

    # Check if result is cached
    cached = await queue_service.get_cached_result(request_id)
    if cached:
        result = json.loads(cached)
        return {
            "request_id": request_id,
            "status": "completed" if not result.get("error") else "error",
            "result": result,
        }

    # Check progress
    progress = await queue_service.get_progress(request_id)
    if progress:
        return {
            "request_id": request_id,
            "status": "processing",
            "progress": progress,
        }

    return {
        "request_id": request_id,
        "status": "unknown",
        "message": "Job not found or expired",
    }


# ============================================================
# Warmup endpoint - scales workers before expected load
# ============================================================

class WarmupRequest(BaseModel):
    """Request to warmup workers."""
    workers: int = Field(default=3, ge=1, le=8, description="Number of workers to keep running (max 8 via warmup)")
    duration_minutes: int = Field(default=30, ge=5, le=120, description="Duration to keep workers up (5-120 min)")


class WarmupResponse(BaseModel):
    """Response from warmup endpoint."""
    status: str
    workers_requested: int
    duration_minutes: int
    warmup_until: str
    message: str


async def require_warmup_api_key(request: Request) -> None:
    """Validate warmup-specific API key."""
    settings = get_settings()

    if not settings.warmup_api_key:
        raise HTTPException(
            status_code=503,
            detail="Warmup endpoint not configured. Set DOC_PIPELINE_WARMUP_API_KEY.",
        )

    # Check header
    api_key = request.headers.get("X-Warmup-Key") or request.headers.get("X-API-Key")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Warmup-Key header",
        )

    if api_key != settings.warmup_api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid warmup API key",
        )


@app.post("/warmup", response_model=WarmupResponse)
async def warmup_workers(
    request: Request,
    body: WarmupRequest,
    _: None = Depends(require_warmup_api_key),
):
    """
    Pre-scale workers before expected high load.

    This endpoint sets a "warmup lock" in Redis that prevents the autoscaler
    from scaling down workers for the specified duration.

    **Requires dedicated API key** (X-Warmup-Key header).

    Use this when you know a large batch of documents is coming and want
    to avoid cold-start latency.
    """
    if queue_service is None:
        raise HTTPException(status_code=503, detail="Queue service not connected")

    from datetime import datetime, timezone, timedelta

    # Calculate warmup end time
    warmup_until = datetime.now(timezone.utc) + timedelta(minutes=body.duration_minutes)
    warmup_until_ts = int(warmup_until.timestamp())

    # Set warmup lock in Redis
    # Key: autoscaler:warmup - Value: {"workers": N, "until": timestamp}
    # TTL: duration_minutes + 1 minute buffer
    warmup_data = {
        "workers": body.workers,
        "until": warmup_until_ts,
        "requested_at": int(datetime.now(timezone.utc).timestamp()),
    }

    ttl_seconds = (body.duration_minutes + 1) * 60

    await queue_service._redis.set(
        "autoscaler:warmup",
        json.dumps(warmup_data),
        ex=ttl_seconds,
    )

    logger.info(
        "warmup_requested",
        workers=body.workers,
        duration_minutes=body.duration_minutes,
        warmup_until=warmup_until.isoformat(),
    )

    return WarmupResponse(
        status="warming_up",
        workers_requested=body.workers,
        duration_minutes=body.duration_minutes,
        warmup_until=warmup_until.isoformat(),
        message=f"Autoscaler will maintain {body.workers} workers for {body.duration_minutes} minutes.",
    )


@app.delete("/warmup")
async def cancel_warmup(
    request: Request,
    _: None = Depends(require_warmup_api_key),
):
    """
    Cancel warmup and allow normal autoscaling.

    **Requires dedicated API key** (X-Warmup-Key header).
    """
    if queue_service is None:
        raise HTTPException(status_code=503, detail="Queue service not connected")

    deleted = await queue_service._redis.delete("autoscaler:warmup")

    if deleted:
        logger.info("warmup_cancelled")
        return {"status": "cancelled", "message": "Warmup cancelled. Normal autoscaling resumed."}
    else:
        return {"status": "not_active", "message": "No active warmup to cancel."}


@app.get("/warmup/status")
async def warmup_status(
    request: Request,
    _: None = Depends(require_warmup_api_key),
):
    """
    Get current warmup status.

    **Requires dedicated API key** (X-Warmup-Key header).
    """
    if queue_service is None:
        raise HTTPException(status_code=503, detail="Queue service not connected")

    from datetime import datetime, timezone

    warmup_data = await queue_service._redis.get("autoscaler:warmup")

    if not warmup_data:
        return {
            "status": "inactive",
            "message": "No active warmup. Normal autoscaling is active.",
        }

    data = json.loads(warmup_data)
    warmup_until = datetime.fromtimestamp(data["until"], tz=timezone.utc)
    remaining = warmup_until - datetime.now(timezone.utc)
    remaining_minutes = max(0, int(remaining.total_seconds() / 60))

    return {
        "status": "active",
        "workers_requested": data["workers"],
        "warmup_until": warmup_until.isoformat(),
        "remaining_minutes": remaining_minutes,
    }


def main():
    """Inicia o servidor da API."""
    import uvicorn

    settings = get_settings()
    logger.info(
        "server_start",
        host=settings.api_host,
        port=settings.api_port,
    )
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        access_log=False,  # Desabilita access log do uvicorn, usamos structlog
    )


if __name__ == "__main__":
    main()
