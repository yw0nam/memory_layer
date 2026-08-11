"""Read-only SQL serving for namespace-scoped tabular document rows."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import asyncpg
from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.core import db
from memory_base.serve import namespaces
from memory_base.serve.http import error, json_body

ROW_CAP = 1_000
RESPONSE_MAX_BYTES = 5 * 1024 * 1024
_SELECT_PREFIX = re.compile(r"^(?:SELECT|WITH)\b", re.IGNORECASE)


class UnsupportedTableValueError(ValueError):
    """A query result contains a value the JSON response deliberately rejects."""


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise UnsupportedTableValueError("query result contains an unsupported binary value")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return str(value)


async def execute_table_query(sql: str, namespace: str) -> dict[str, Any]:
    """Execute one prepared SELECT under the restricted role and namespace RLS."""
    async with db.acquire_table_query() as conn:
        async with conn.transaction(readonly=True):
            await conn.execute(f"SET LOCAL app.namespace = {_sql_literal(namespace)}")
            statement = await conn.prepare(sql)
            columns = [attribute.name for attribute in statement.get_attributes()]
            cursor = await statement.cursor()
            records = await cursor.fetch(ROW_CAP + 1)

    truncated = len(records) > ROW_CAP
    rows = [[_normalize_value(value) for value in record] for record in records[:ROW_CAP]]
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }


def _postgres_message(exc: asyncpg.PostgresError) -> str:
    return getattr(exc, "message", None) or str(exc)


async def tables_query_route(request: Request) -> JSONResponse:
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
        payload = await execute_table_query(sql, namespace)
    except asyncpg.QueryCanceledError as exc:
        return error(_postgres_message(exc), 408)
    except asyncpg.PostgresError as exc:
        return error(_postgres_message(exc))
    except UnsupportedTableValueError as exc:
        return error(str(exc))

    response = JSONResponse(payload)
    if len(response.body) > RESPONSE_MAX_BYTES:
        return error("serialized query response exceeds 5 MB", 413)
    return response
