"""MCP tool for read-only SQL over ingested CSV rows."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context

from memory_base.serve.common import rest_client


async def query_table(
    sql: str,
    namespace: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Run a read-only SQL query over ingested CSV rows.

    Find the CSV card with `search_memory` first. Its `ref` is
    `<document_id>#card-N` (the part before `#` is the document_id), and its
    top-level `columns` field lists the available JSON keys. Rows live in `memory.doc_rows`
    as jsonb: use `(data->>'column')::numeric` for numeric calculations and
    `WHERE document_id = '...'` to scope one table.
    The server restricts the query to one permitted namespace and returns at
    most 1,000 rows.
    """
    body: dict[str, Any] = {"sql": sql}
    if namespace is not None:
        body["namespace"] = namespace
    return await rest_client.call(
        "POST",
        "/tables/query",
        json=body,
        headers=rest_client.auth_headers(ctx),
    )
