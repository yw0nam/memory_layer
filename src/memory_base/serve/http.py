"""Shared request/response helpers for serve-layer route modules."""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

TEXT_LIMIT = 2000


async def json_body(request: Request) -> dict[str, Any]:
    """Parse the request body as JSON, requiring a top-level object."""
    body = await request.json()
    if not isinstance(body, dict):
        raise ValueError("JSON body must be an object")
    return body


def error(message: str, status: int = 400) -> JSONResponse:
    """Build a `{"error": message}` JSON response with the given status code."""
    return JSONResponse({"error": message}, status_code=status)
