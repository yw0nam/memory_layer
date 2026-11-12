"""Liveness and dependency health routes, their service probes, and the /health access-log filter."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.core import db
from memory_base.core.config import require_env

HEALTH_PROBE_TIMEOUT_SECONDS = 5.0


async def db_healthy() -> bool:
    """Return whether the configured database accepts a simple query."""
    async with db.acquire(timeout=HEALTH_PROBE_TIMEOUT_SECONDS) as conn:
        return bool(await conn.fetchval("SELECT 1"))


async def _models_endpoint_healthy(env_var: str) -> bool:
    """Return whether the vLLM server's /models path answers 2xx, without running inference."""
    base_url = require_env(env_var)
    async with httpx.AsyncClient(timeout=HEALTH_PROBE_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{base_url.rstrip('/')}/models")
    return 200 <= response.status_code < 300


async def embedding_healthy() -> bool:
    """Return whether the embedding endpoint (EMB_URL) is reachable."""
    return await _models_endpoint_healthy("EMB_URL")


async def rerank_healthy() -> bool:
    """Return whether the rerank endpoint (RERANK_URL) is reachable."""
    return await _models_endpoint_healthy("RERANK_URL")


async def _probe(check: Callable[[], Awaitable[bool]]) -> bool:
    """Run a health probe, turning any exception into a false result."""
    try:
        return bool(await check())
    except Exception:
        return False


async def health(request: Request) -> JSONResponse:
    """Report that the process serves HTTP, reaching nothing outside it."""
    del request
    return JSONResponse({"status": "ok"})


async def health_services(request: Request) -> JSONResponse:
    """Report health of the DB, embedding, and rerank dependencies."""
    del request
    db, embedding, rerank = await asyncio.gather(
        _probe(db_healthy),
        _probe(embedding_healthy),
        _probe(rerank_healthy),
    )
    checks = {"db": db, "embedding": embedding, "rerank": rerank}
    required_up = db and embedding and rerank
    status_code = 200 if required_up else 503
    return JSONResponse(
        {"status": "ok" if required_up else "error", "checks": checks},
        status_code=status_code,
    )


class HealthAccessFilter(logging.Filter):
    """Successful liveness probes drown real requests in the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            _, method, path, _, status = record.args
        except (TypeError, ValueError):
            return True
        return not (method == "GET" and path == "/health" and status == 200)
