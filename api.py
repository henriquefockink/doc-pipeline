#!/usr/bin/env python3
"""
API REST para o pipeline de classificação e extração de documentos.
"""

import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from doc_pipeline import DocumentPipeline, PipelineResult
from doc_pipeline.extractors import GOTOCRExtractor
from doc_pipeline.auth import AuthInfo, close_pool, require_api_key
from doc_pipeline.config import get_settings
from doc_pipeline.observability import (
    PrometheusMiddleware,
    get_logger,
    get_metrics,
    metrics_endpoint,
    setup_logging,
)
from doc_pipeline.schemas import (
    ClassificationResult,
    DocumentType,
    ExtractionResult,
    GenericExtractionResult,
)

# Setup logging antes de tudo
settings = get_settings()
setup_logging(
    json_format=settings.log_json,
    log_level=settings.log_level,
)

logger = get_logger("api")

# Global pipeline instance
pipeline: DocumentPipeline | None = None

# GOT-OCR extractor para PDFs (lazy-loaded)
_got_ocr_extractor: GOTOCRExtractor | None = None


def _get_got_ocr_extractor() -> GOTOCRExtractor:
    """Retorna o extractor GOT-OCR para PDFs (lazy-loaded)."""
    global _got_ocr_extractor
    if _got_ocr_extractor is None:
        settings = get_settings()
        _got_ocr_extractor = GOTOCRExtractor(
            model_name=settings.extractor_model_got,
            device=settings.extractor_device,
        )
    return _got_ocr_extractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida da aplicação."""
    global pipeline

    settings = get_settings()
    logger.info(
        "startup",
        classifier_model=str(settings.classifier_model_path),
        extractor_backend=settings.extractor_backend.value,
        warmup=settings.warmup_on_start,
        auth_env_keys=len(settings.api_keys_list),
        auth_database=bool(settings.database_url),
    )

    pipeline = DocumentPipeline()

    if settings.warmup_on_start:
        logger.info("warmup_start", component="classifier")
        pipeline.warmup(load_classifier=True, load_extractor=False)
        logger.info("warmup_complete", component="classifier")

        logger.info("warmup_start", component="extractor")
        pipeline.warmup(load_classifier=False, load_extractor=True)
        logger.info("warmup_complete", component="extractor")

        # Warmup do GOT-OCR para PDFs (carrega em paralelo ao Qwen-VL)
        logger.info("warmup_start", component="got-ocr")
        got_extractor = _get_got_ocr_extractor()
        got_extractor.load_model()
        logger.info("warmup_complete", component="got-ocr")

    logger.info("startup_complete")
    yield

    # Cleanup
    logger.info("shutdown_start")
    if pipeline:
        pipeline.unload_extractor()
    if _got_ocr_extractor:
        _got_ocr_extractor.unload_model()
    await close_pool()
    logger.info("shutdown_complete")


app = FastAPI(
    title="doc-pipeline API",
    description="API para classificação e extração de dados de documentos brasileiros (RG/CNH)",
    version="0.1.0",
    lifespan=lifespan,
)

# Adiciona middleware de métricas (api_only=True filtra apenas endpoints da API)
app.add_middleware(PrometheusMiddleware)


# Response models
class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    extractor_loaded: bool
    extractor_backend: str


class ClassesResponse(BaseModel):
    classes: list[str]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


@app.get("/metrics")
async def metrics():
    """Endpoint para Prometheus scraping."""
    return metrics_endpoint()


@app.get("/health", response_model=HealthResponse)
async def health():
    """Verifica status da API."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        classifier_loaded=pipeline is not None and pipeline._classifier is not None,
        extractor_loaded=pipeline is not None and pipeline._extractor is not None,
        extractor_backend=settings.extractor_backend.value,
    )


@app.get("/classes", response_model=ClassesResponse)
async def list_classes(auth: AuthInfo = Depends(require_api_key)):
    """Lista classes de documentos suportadas."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    return ClassesResponse(classes=pipeline.classes)


@app.post("/classify", response_model=ClassificationResult)
async def classify(
    request: Request,
    arquivo: Annotated[UploadFile, File(description="Imagem do documento")],
    auth: AuthInfo = Depends(require_api_key),
):
    """
    Classifica uma imagem de documento.

    Retorna o tipo do documento e a confiança da classificação.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    metrics = get_metrics()
    start_time = time.perf_counter()
    client = auth.client_name or "unknown"

    try:
        contents = await arquivo.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = pipeline.classify(image)

        # Métricas de negócio
        metrics.documents_processed.labels(
            document_type=result.document_type.value,
            operation="classify",
        ).inc()
        metrics.classification_confidence.labels(
            document_type=result.document_type.value,
        ).observe(result.confidence)
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/classify",
            status="200",
        ).inc()

        duration = time.perf_counter() - start_time
        logger.info(
            "classify_success",
            document_type=result.document_type.value,
            confidence=round(result.confidence, 3),
            duration_ms=round(duration * 1000, 2),
            filename=arquivo.filename,
            client=client,
        )

        return result

    except Exception as e:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/classify",
            status="400",
        ).inc()
        logger.error(
            "classify_error",
            error=str(e),
            error_type=type(e).__name__,
            filename=arquivo.filename,
            client=client,
        )
        raise HTTPException(status_code=400, detail=str(e))


def _is_pdf(contents: bytes, filename: str | None) -> bool:
    """Detecta se o arquivo é um PDF pelo magic number ou extensão."""
    # Magic number do PDF: %PDF
    if contents[:4] == b"%PDF":
        return True
    # Fallback: extensão do arquivo
    if filename and filename.lower().endswith(".pdf"):
        return True
    return False


@app.post("/extract", response_model=ExtractionResult | GenericExtractionResult)
async def extract(
    request: Request,
    arquivo: Annotated[UploadFile, File(description="Imagem ou PDF do documento")],
    doc_type: Annotated[
        str,
        Query(description="Tipo do documento (rg_frente, cnh_frente, generic, etc)"),
    ],
    auth: AuthInfo = Depends(require_api_key),
):
    """
    Extrai dados de uma imagem ou PDF de documento.

    Para doc_type=generic, retorna apenas o texto bruto (OCR puro).
    PDFs são suportados apenas para doc_type=generic.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    metrics = get_metrics()
    client = auth.client_name or "unknown"

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

    start_time = time.perf_counter()
    contents = await arquivo.read()
    is_pdf = _is_pdf(contents, arquivo.filename)

    # PDF só é suportado para tipo generic
    if is_pdf and not document_type.is_generic:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/extract",
            status="400",
        ).inc()
        logger.warning(
            "extract_pdf_not_generic",
            doc_type=doc_type,
            filename=arquivo.filename,
            client=client,
        )
        raise HTTPException(
            status_code=400,
            detail="PDF só é suportado para doc_type=generic. "
            "Para RG/CNH, envie uma imagem.",
        )

    try:
        if document_type.is_generic:
            # OCR puro - retorna texto bruto
            if is_pdf:
                # PDF: usa GOT-OCR automaticamente (suporta PDF nativo)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                    f.write(contents)
                    pdf_path = f.name

                try:
                    got_extractor = _get_got_ocr_extractor()
                    result = got_extractor.extract_generic_from_pdf(pdf_path)
                finally:
                    os.unlink(pdf_path)
            else:
                image = Image.open(io.BytesIO(contents)).convert("RGB")
                result = pipeline.extract_generic(image)
        else:
            # Extração estruturada (RG/CNH)
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = pipeline.extract(image, document_type)

        # Métricas de negócio
        metrics.documents_processed.labels(
            document_type=document_type.value,
            operation="extract",
        ).inc()
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/extract",
            status="200",
        ).inc()

        duration = time.perf_counter() - start_time
        log_extra = {
            "document_type": document_type.value,
            "duration_ms": round(duration * 1000, 2),
            "filename": arquivo.filename,
            "client": client,
            "is_pdf": is_pdf,
            "backend": result.backend if hasattr(result, "backend") else "unknown",
        }
        if is_pdf and hasattr(result, "total_pages"):
            log_extra["total_pages"] = result.total_pages

        logger.info("extract_success", **log_extra)

        return result

    except Exception as e:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/extract",
            status="400",
        ).inc()
        logger.error(
            "extract_error",
            error=str(e),
            error_type=type(e).__name__,
            doc_type=doc_type,
            filename=arquivo.filename,
            client=client,
            is_pdf=is_pdf,
        )
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/process", response_model=PipelineResult)
async def process(
    request: Request,
    arquivo: Annotated[UploadFile, File(description="Imagem do documento")],
    extract: Annotated[bool, Query(description="Se deve extrair dados")] = True,
    min_confidence: Annotated[float, Query(ge=0.0, le=1.0, description="Confiança mínima")] = 0.5,
    auth: AuthInfo = Depends(require_api_key),
):
    """
    Pipeline completo: classifica e extrai dados de uma imagem de documento.

    Este é o endpoint principal que executa classificação e extração em sequência.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    metrics = get_metrics()
    start_time = time.perf_counter()
    client = auth.client_name or "unknown"

    try:
        contents = await arquivo.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = pipeline.process(
            image,
            extract=extract,
            min_confidence=min_confidence,
        )

        # Métricas de negócio
        doc_type = result.classification.document_type.value
        metrics.documents_processed.labels(
            document_type=doc_type,
            operation="process",
        ).inc()
        metrics.classification_confidence.labels(
            document_type=doc_type,
        ).observe(result.classification.confidence)
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/process",
            status="200",
        ).inc()

        duration = time.perf_counter() - start_time
        logger.info(
            "process_success",
            document_type=doc_type,
            confidence=round(result.classification.confidence, 3),
            extracted=result.extraction is not None,
            duration_ms=round(duration * 1000, 2),
            filename=arquivo.filename,
            client=client,
        )

        return result

    except Exception as e:
        metrics.requests_by_client.labels(
            client=client,
            endpoint="/process",
            status="400",
        ).inc()
        logger.error(
            "process_error",
            error=str(e),
            error_type=type(e).__name__,
            filename=arquivo.filename,
            client=client,
        )
        raise HTTPException(status_code=400, detail=str(e))


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
