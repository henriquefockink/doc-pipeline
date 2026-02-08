"""API Key authentication with hybrid storage (env + PostgreSQL)."""

import asyncio
from dataclasses import dataclass

import asyncpg
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from doc_pipeline.config import get_settings
from doc_pipeline.observability import get_logger

logger = get_logger("auth")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Connection pool (initialized lazily, protected by lock)
_db_pool: asyncpg.Pool | None = None
_db_pool_lock = asyncio.Lock()


async def _get_db_pool() -> asyncpg.Pool | None:
    """Get or create the database connection pool."""
    global _db_pool
    settings = get_settings()
    if not settings.database_url:
        return None
    if _db_pool is not None:
        return _db_pool
    async with _db_pool_lock:
        if _db_pool is None:
            _db_pool = await asyncpg.create_pool(
                settings.database_url,
                min_size=2,
                max_size=20,
            )
    return _db_pool


@dataclass
class AuthInfo:
    """Authentication information returned after successful validation."""

    api_key: str
    client_name: str | None = None
    api_key_prefix: str | None = None  # First 8 chars for audit logging

    @classmethod
    def from_key(cls, api_key: str, client_name: str | None = None) -> "AuthInfo":
        """Create AuthInfo from API key, extracting prefix for logging."""
        prefix = api_key[:8] if len(api_key) >= 8 else api_key
        return cls(
            api_key=api_key,
            client_name=client_name,
            api_key_prefix=f"{prefix}_",
        )


async def _check_db_key(api_key: str, service: str = "ocr") -> dict | None:
    """Check if API key exists in PostgreSQL and has access to the service."""
    pool = await _get_db_pool()
    if pool is None:
        return None

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT client_name, description, active, created_at, services
            FROM api_keys
            WHERE key = $1 AND active = TRUE AND $2 = ANY(services)
            """,
            api_key,
            service,
        )
        if row:
            return {
                "client_name": row["client_name"],
                "description": row["description"],
                "active": row["active"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "services": list(row["services"]) if row["services"] else [],
            }
        return None


async def _has_any_db_keys(service: str = "ocr") -> bool:
    """Check if there are any API keys in PostgreSQL for the given service."""
    pool = await _get_db_pool()
    if pool is None:
        return False

    async with pool.acquire() as conn:
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM api_keys WHERE active = TRUE AND $1 = ANY(services))",
            service,
        )
        return result


async def require_api_key(api_key: str | None = Security(api_key_header)) -> AuthInfo:
    """Validate API key from header for OCR service.

    Checks both:
    - Environment variable API_KEY/API_KEYS (master keys)
    - PostgreSQL storage (dynamic client keys with 'ocr' service access)

    Returns AuthInfo with key, client_name, and api_key_prefix for logging.
    Raises 401 if key is missing or invalid.
    """
    settings = get_settings()
    service = "ocr"

    try:
        # If no API keys configured in env, check if PostgreSQL has any keys
        if not settings.api_keys_list:
            has_db_keys = await _has_any_db_keys(service)
            if not has_db_keys:
                # No auth configured at all - allow access (dev mode)
                logger.debug("auth_dev_mode", reason="no_keys_configured")
                return AuthInfo.from_key("no-auth", client_name="dev-mode")

        if not api_key:
            logger.warning("auth_missing_key")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Include 'X-API-Key' header.",
            )

        # Check env keys first (master keys - have access to all services)
        if api_key in settings.api_keys_list:
            logger.debug("auth_master_key", client="master-key")
            return AuthInfo.from_key(api_key, client_name="master-key")

        # Check PostgreSQL keys (dynamic client keys with service check)
        key_data = await _check_db_key(api_key, service)
        if key_data:
            logger.debug("auth_success", client=key_data["client_name"])
            return AuthInfo.from_key(api_key, client_name=key_data["client_name"])

        logger.warning(
            "auth_invalid_key",
            key_prefix=api_key[:8] if len(api_key) >= 8 else api_key,
            key_length=len(api_key),
            key_suffix=api_key[-4:] if len(api_key) >= 4 else api_key,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid API key or no access to '{service}' service.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("auth_error", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )
