"""Logging estruturado com structlog."""

import logging
import sys

import structlog


def setup_logging(json_format: bool = True, log_level: str = "INFO") -> None:
    """
    Configura logging estruturado para a aplicação.

    Args:
        json_format: Se True, logs em JSON. Se False, logs formatados para console.
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR).
    """
    # Configura nivel de log
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Processadores comuns
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # JSON para produção
        renderer = structlog.processors.JSONRenderer()
    else:
        # Colorido para desenvolvimento
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configura logging padrão do Python para capturar logs de libs externas
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # Redireciona logs do uvicorn para structlog
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logging.getLogger(logger_name).handlers = []


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Retorna um logger estruturado.

    Args:
        name: Nome do logger (opcional).

    Returns:
        Logger estruturado.
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()
