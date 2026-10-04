"""Read-only SQL execution over namespace-scoped tabular document rows."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from memory_base.core import db

ROW_CAP = 1_000


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
