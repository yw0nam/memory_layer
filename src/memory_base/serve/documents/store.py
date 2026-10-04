"""Storage for ingested documents: their chunks in memory_chunks and table rows in doc_rows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once


async def existing_document_state(
    document_id: str, namespace: str = "default", schema: str | None = None
) -> tuple[str | None, bool] | None:
    schema = PG_SCHEMA if schema is None else schema
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        row = await conn.fetchrow(
            f"""
            SELECT metadata->>'content_hash' AS content_hash,
                   COALESCE(metadata->'table_rows_loaded' = 'true'::jsonb, false)
                     AS table_rows_loaded
            FROM "{schema}".memory_chunks
            WHERE source_type = 'document' AND source_ref = $1 AND namespace = $2
            LIMIT 1
            """,
            document_id,
            namespace,
        )
        if row is None:
            return None
        return row["content_hash"], row["table_rows_loaded"]


async def existing_document_owner(
    document_id: str, namespace: str = "default", schema: str | None = None
) -> tuple[bool, str | None]:
    """Whether a document has stored chunks in this namespace, and its recorded creator."""
    schema = PG_SCHEMA if schema is None else schema
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        row = await conn.fetchrow(
            f"""
            SELECT metadata->>'created_by' AS created_by
            FROM "{schema}".memory_chunks
            WHERE source_type = 'document' AND source_ref = $1 AND namespace = $2
            LIMIT 1
            """,
            document_id,
            namespace,
        )
        if row is None:
            return False, None
        return True, row["created_by"]


async def delete_document_rows(document_id: str, namespace: str = "default") -> int:
    """Delete a document's chunks in one namespace; returns the number of rows removed."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            status = await conn.execute(
                f"""
                DELETE FROM "{PG_SCHEMA}".memory_chunks
                WHERE source_type = 'document' AND source_ref = $1 AND namespace = $2
                """,
                document_id,
                namespace,
            )
            await conn.execute(
                f'DELETE FROM "{PG_SCHEMA}".doc_rows WHERE namespace = $1 AND document_id = $2',
                namespace,
                document_id,
            )
        return int(status.rsplit(" ", 1)[-1])


async def replace_document_rows(
    document_id: str,
    rows: Sequence[dict[str, Any]],
    namespace: str = "default",
    schema: str | None = None,
    table_rows: Sequence[dict[str, Any]] = (),
) -> None:
    """Replace one document's card, chunks, and table rows in a single transaction.

    schema overrides PG_SCHEMA for this call; only the eval harness passes it.
    """
    schema = PG_SCHEMA if schema is None else schema
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await conn.execute(
                f"""
                DELETE FROM "{schema}".memory_chunks
                WHERE source_type = 'document' AND source_ref = $1 AND namespace = $2
                """,
                document_id,
                namespace,
            )
            await conn.execute(
                f'DELETE FROM "{schema}".doc_rows WHERE namespace = $1 AND document_id = $2',
                namespace,
                document_id,
            )
            await conn.executemany(
                f"""
                INSERT INTO "{schema}".memory_chunks
                  (id, source_type, source_ref, chunk_kind, session_id, content_raw,
                   distilled, embedding, ts_last_active, namespace, metadata)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::halfvec,$9,$10,$11::jsonb)
                """,
                [
                    (
                        row["id"],
                        row["source_type"],
                        row["source_ref"],
                        row["chunk_kind"],
                        row["session_id"],
                        row["content_raw"],
                        row["distilled"],
                        row["embedding"],
                        row["ts_last_active"],
                        row.get("namespace", namespace),
                        json.dumps(row["metadata"], ensure_ascii=False),
                    )
                    for row in rows
                ],
            )
            if table_rows:
                await conn.executemany(
                    f"""
                    INSERT INTO "{schema}".doc_rows
                      (namespace, document_id, row_index, data)
                    VALUES ($1, $2, $3, $4::jsonb)
                    """,
                    [
                        (
                            namespace,
                            document_id,
                            row["row_index"],
                            json.dumps(row["data"], ensure_ascii=False),
                        )
                        for row in table_rows
                    ],
                )
