"""Métricas Prometheus para a API."""

import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import PlainTextResponse
from fastapi.routing import APIRoute
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware


class Metrics:
    """Container para métricas Prometheus."""

    def __init__(self, namespace: str = "doc_pipeline"):
        self.namespace = namespace

        # Requests totais por endpoint, método e status
        self.requests_total = Counter(
            f"{namespace}_requests_total",
            "Total de requests",
            ["method", "endpoint", "status"],
        )

        # Latência dos requests (histograma)
        self.request_duration_seconds = Histogram(
            f"{namespace}_request_duration_seconds",
            "Duração dos requests em segundos",
            ["method", "endpoint"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )

        # Requests em andamento
        self.requests_in_progress = Gauge(
            f"{namespace}_requests_in_progress",
            "Requests sendo processados",
            ["method", "endpoint"],
        )

        # Erros por tipo
        self.errors_total = Counter(
            f"{namespace}_errors_total",
            "Total de erros",
            ["method", "endpoint", "error_type"],
        )

        # Métricas de negócio
        self.documents_processed = Counter(
            f"{namespace}_documents_processed_total",
            "Total de documentos processados",
            ["document_type", "operation"],
        )

        # Requests por cliente (API key)
        self.requests_by_client = Counter(
            f"{namespace}_requests_by_client_total",
            "Total de requests por cliente",
            ["client", "endpoint", "status"],
        )

        self.classification_confidence = Histogram(
            f"{namespace}_classification_confidence",
            "Confiança das classificações",
            ["document_type"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
        )

        # GPU memory (se disponível)
        self.gpu_memory_used_bytes = Gauge(
            f"{namespace}_gpu_memory_used_bytes",
            "Memória GPU em uso",
        )

        # Queue metrics
        self.queue_depth = Gauge(
            f"{namespace}_queue_depth",
            "Jobs waiting in queue",
        )

        self.queue_wait_seconds = Histogram(
            f"{namespace}_queue_wait_seconds",
            "Time job waited in queue before processing",
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
        )

        self.jobs_processed = Counter(
            f"{namespace}_jobs_processed_total",
            "Total jobs processed by worker",
            ["operation", "status", "delivery_mode"],
        )

        self.webhook_deliveries = Counter(
            f"{namespace}_webhook_deliveries_total",
            "Webhook delivery attempts",
            ["status"],
        )

        self.worker_processing_seconds = Histogram(
            f"{namespace}_worker_processing_seconds",
            "Time spent processing a job (excluding queue wait)",
            ["operation"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        )


# Singleton global
_metrics: Metrics | None = None


def get_metrics() -> Metrics:
    """Retorna a instância global de métricas."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


# Autoscaler metrics file paths (Docker volume first, then local fallback)
AUTOSCALER_METRICS_PATHS = [
    "/tmp/autoscaler-metrics/doc_pipeline_autoscaler.prom",  # Docker volume
    "/tmp/doc_pipeline_autoscaler.prom",  # Local/legacy
]


def metrics_endpoint() -> Response:
    """Endpoint /metrics para Prometheus scraping."""
    content = generate_latest()

    # Append autoscaler metrics if available (try multiple paths)
    for metrics_path in AUTOSCALER_METRICS_PATHS:
        try:
            with open(metrics_path) as f:
                autoscaler_metrics = f.read()
                content = content + b"\n" + autoscaler_metrics.encode()
                break  # Found metrics, stop searching
        except FileNotFoundError:
            continue  # Try next path

    return PlainTextResponse(
        content=content,
        media_type=CONTENT_TYPE_LATEST,
    )


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware que coleta métricas de cada request."""

    # Prefixos de paths para ignorar (arquivos estáticos, assets, etc.)
    DEFAULT_EXCLUDE_PREFIXES = (
        "/metrics",
        "/health",
        "/docs",
        "/redoc",
        "/openapi",
        "/favicon",
        "/static",
        "/assets",
        "/fonts",
        "/_next",
    )

    # Endpoints da API que devem ser monitorados
    API_ENDPOINTS = {"/classify", "/extract", "/process", "/jobs", "/ocr"}

    def __init__(
        self,
        app,
        exclude_prefixes: tuple[str, ...] | None = None,
        api_only: bool = True,
    ):
        super().__init__(app)
        self.exclude_prefixes = exclude_prefixes or self.DEFAULT_EXCLUDE_PREFIXES
        self.api_only = api_only
        self.metrics = get_metrics()

    def _should_track(self, path: str) -> bool:
        """Verifica se o path deve ter métricas coletadas."""
        # Ignora prefixos conhecidos
        for prefix in self.exclude_prefixes:
            if path.startswith(prefix):
                return False

        # Se api_only=True, só coleta para endpoints conhecidos
        if self.api_only:
            normalized = path.rstrip("/") or "/"
            # Check exact match or prefix match for parameterized routes
            if normalized in self.API_ENDPOINTS:
                return True
            # Check for /jobs/{id}/status pattern
            if normalized.startswith("/jobs/"):
                return True
            return False

        return True

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Não coleta métricas para paths excluídos
        if not self._should_track(request.url.path):
            return await call_next(request)

        method = request.method
        # Normaliza path para evitar explosão de cardinalidade
        endpoint = self._normalize_path(request.url.path)

        # Incrementa requests em andamento
        self.metrics.requests_in_progress.labels(
            method=method,
            endpoint=endpoint,
        ).inc()

        start_time = time.perf_counter()
        status_code = 500
        error_type = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception as e:
            error_type = type(e).__name__
            raise

        finally:
            # Calcula duração
            duration = time.perf_counter() - start_time

            # Decrementa requests em andamento
            self.metrics.requests_in_progress.labels(
                method=method,
                endpoint=endpoint,
            ).dec()

            # Registra request
            self.metrics.requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code),
            ).inc()

            # Registra latência
            self.metrics.request_duration_seconds.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            # Registra erro se houve
            if error_type:
                self.metrics.errors_total.labels(
                    method=method,
                    endpoint=endpoint,
                    error_type=error_type,
                ).inc()

    def _normalize_path(self, path: str) -> str:
        """Normaliza path para evitar alta cardinalidade."""
        # Remove trailing slash
        path = path.rstrip("/") or "/"
        # Normalize /jobs/{id}/status to /jobs
        if path.startswith("/jobs/"):
            return "/jobs"
        return path
