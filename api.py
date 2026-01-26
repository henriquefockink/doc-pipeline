#!/usr/bin/env python3
"""
API REST para o pipeline de classificação e extração de documentos.
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from PIL import Image
from pydantic import BaseModel

from doc_pipeline import DocumentPipeline, PipelineResult
from doc_pipeline.config import ExtractorBackend, get_settings
from doc_pipeline.schemas import (
    ClassificationResult,
    CNHData,
    DocumentType,
    ExtractionResult,
    RGData,
)


# Global pipeline instance
pipeline: DocumentPipeline | None = None

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Valida a API key se configurada."""
    settings = get_settings()

    # Se não há API key configurada, permite acesso
    if not settings.api_key:
        return "no-auth"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key não fornecida. Use o header X-API-Key.",
        )

    if api_key != settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="API key inválida.",
        )

    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida da aplicação."""
    global pipeline

    settings = get_settings()
    print("Inicializando pipeline...")
    print(f"  Classificador: {settings.classifier_model_path}")
    print(f"  Backend extractor: {settings.extractor_backend.value}")
    print(f"  Warmup: {settings.warmup_on_start}")
    print(f"  Autenticação: {'Ativada' if settings.api_key else 'Desativada'}")

    pipeline = DocumentPipeline()

    if settings.warmup_on_start:
        print("Carregando classificador...")
        pipeline.warmup(load_classifier=True, load_extractor=False)
        print("Carregando extractor...")
        pipeline.warmup(load_classifier=False, load_extractor=True)

    print("Pipeline pronto!")
    yield

    # Cleanup
    if pipeline:
        pipeline.unload_extractor()


app = FastAPI(
    title="doc-pipeline API",
    description="API para classificação e extração de dados de documentos brasileiros (RG/CNH)",
    version="0.1.0",
    lifespan=lifespan,
)


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
async def list_classes(_: str = Depends(verify_api_key)):
    """Lista classes de documentos suportadas."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    return ClassesResponse(classes=pipeline.classes)


@app.post("/classify", response_model=ClassificationResult)
async def classify(
    arquivo: Annotated[UploadFile, File(description="Imagem do documento")],
    _: str = Depends(verify_api_key),
):
    """
    Classifica uma imagem de documento.

    Retorna o tipo do documento e a confiança da classificação.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    try:
        contents = await arquivo.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = pipeline.classify(image)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/extract", response_model=ExtractionResult)
async def extract(
    arquivo: Annotated[UploadFile, File(description="Imagem do documento")],
    doc_type: Annotated[str, Query(description="Tipo do documento (rg_frente, cnh_frente, etc)")],
    _: str = Depends(verify_api_key),
):
    """
    Extrai dados de uma imagem de documento.

    Requer que o tipo do documento seja especificado.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    try:
        document_type = DocumentType(doc_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de documento inválido: {doc_type}. Valores válidos: {[d.value for d in DocumentType]}",
        )

    try:
        contents = await arquivo.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = pipeline.extract(image, document_type)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/process", response_model=PipelineResult)
async def process(
    arquivo: Annotated[UploadFile, File(description="Imagem do documento")],
    extract: Annotated[bool, Query(description="Se deve extrair dados")] = True,
    min_confidence: Annotated[float, Query(ge=0.0, le=1.0, description="Confiança mínima")] = 0.5,
    _: str = Depends(verify_api_key),
):
    """
    Pipeline completo: classifica e extrai dados de uma imagem de documento.

    Este é o endpoint principal que executa classificação e extração em sequência.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline não inicializado")

    try:
        contents = await arquivo.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = pipeline.process(
            image,
            extract=extract,
            min_confidence=min_confidence,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def main():
    """Inicia o servidor da API."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
