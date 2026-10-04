"""REST route for read-only SQL over tabular document rows."""

from __future__ import annotations

import re

import asyncpg
from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.serve.access import namespaces
from memory_base.serve.common.http import error, json_body
from memory_base.serve.tables import store

RESPONSE_MAX_BYTES = 5 * 1024 * 1024
_SELECT_PREFIX = re.compile(r"^(?:SELECT|WITH)\b", re.IGNORECASE)


def _postgres_message(exc: asyncpg.PostgresError) -> str:
    return getattr(exc, "message", None) or str(exc)


async def query_route(request: Request) -> JSONResponse:
    """Validate and execute a raw SELECT in one namespace permitted by the caller's key."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")

    sql = body.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        return error("sql must be a non-empty string")
    sql = sql.strip()
    if _SELECT_PREFIX.match(sql) is None:
        return error("sql must start with SELECT or WITH")

    namespace = body.get("namespace", request.state.key.home)
    try:
        namespace = namespaces.validate_namespace_name(namespace)
    except namespaces.NamespaceError as exc:
        return error(str(exc))
    if not request.state.key.permits(namespace):
        return error(f"namespace {namespace!r} is outside the caller's allowed set", 403)

    try:
        payload = await store.execute_table_query(sql, namespace)
    except asyncpg.QueryCanceledError as exc:
        return error(_postgres_message(exc), 408)
    except asyncpg.PostgresError as exc:
        return error(_postgres_message(exc))
    except store.UnsupportedTableValueError as exc:
        return error(str(exc))

    response = JSONResponse(payload)
    if len(response.body) > RESPONSE_MAX_BYTES:
        return error("serialized query response exceeds 5 MB", 413)
    return response
