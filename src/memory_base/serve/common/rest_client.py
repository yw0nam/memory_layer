"""The MCP tools' client for the REST backend, carrying the caller's API key."""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import Context

from memory_base.core.config import SERVICE_TIMEOUT_SECONDS

REST_URL = os.environ.get("REST_URL", "http://localhost:8010")
API_KEY_HEADER = "x-api-key"


class BackendError(ValueError):
    """A non-2xx REST response; `payload` is its JSON body when that names an `error`."""

    def __init__(self, message: str, status: int, payload: dict[str, Any] | None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


def client() -> httpx.AsyncClient:
    # Outlasts every backend ceiling, so the backend's own error arrives before this gives up.
    return httpx.AsyncClient(base_url=REST_URL, timeout=SERVICE_TIMEOUT_SECONDS)


def _raise_backend_error(response: httpx.Response, label: str) -> None:
    try:
        payload = response.json()
        message = payload["error"]
    except (ValueError, KeyError, TypeError):
        payload = None
        body = response.text.strip()[:500]
        message = (
            f"{label} failed: backend returned {response.status_code} {response.reason_phrase}"
        )
        if body:
            message += f": {body}"
    raise BackendError(message, response.status_code, payload)


async def call(method: str, path: str, **kwargs: Any) -> Any:
    """Issue a REST call and return the decoded JSON body.

    Any failure raises ValueError with its reason: BackendError with the backend's `error`
    payload for a non-2xx status, or which call timed out or could not reach the backend.
    """
    label = f"{method} {path}"
    try:
        async with client() as session:
            response = await session.request(method, path, **kwargs)
    except httpx.TimeoutException as exc:
        raise ValueError(f"{label} timed out waiting for the REST backend at {REST_URL}") from exc
    except httpx.TransportError as exc:
        reason = str(exc) or type(exc).__name__
        raise ValueError(
            f"{label} failed: REST backend at {REST_URL} is unreachable ({reason})"
        ) from exc
    if not response.is_success:
        _raise_backend_error(response, label)
    return response.json()


def api_key(ctx: Context | None) -> str | None:
    """The caller's API key: the request's X-API-Key header, or MEMORY_API_KEY for stdio."""
    request = None
    if ctx is not None:
        try:
            request = ctx.request_context.request
        except ValueError:
            request = None
    if request is not None:
        header = request.headers.get(API_KEY_HEADER)
        if header:
            return header
    return os.environ.get("MEMORY_API_KEY")


def auth_headers(ctx: Context | None) -> dict[str, str]:
    key = api_key(ctx)
    return {API_KEY_HEADER: key} if key else {}
