"""Storage for ingested documents: their chunks in memory_chunks and table rows in doc_rows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.serve.common.job_store import TERMINAL_STATUSES


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


async def reset_interrupted_uploads() -> None:
    """Fail interrupted document jobs whose spool file is gone; return the rest to the queued stage."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f'''SELECT job_id, spool_path FROM "{PG_SCHEMA}".jobs
            WHERE kind = 'document' AND status <> ALL($1::text[])''',
            list(TERMINAL_STATUSES),
        )
        missing = [row["job_id"] for row in rows if not Path(row["spool_path"]).is_file()]
        async with conn.transaction():
            if missing:
                await conn.execute(
                    f'''UPDATE "{PG_SCHEMA}".jobs
                    SET status = 'failed', stage = 'done',
                        error = 'document spool file is missing during startup recovery',
                        updated_at = now()
                    WHERE job_id = ANY($1::text[])''',
                    missing,
                )
            await conn.execute(
                f'''UPDATE "{PG_SCHEMA}".jobs SET stage = 'queued'
                WHERE kind = 'document' AND status <> ALL($1::text[])''',
                list(TERMINAL_STATUSES),
            )


async def document_spool_rows() -> list[dict[str, Any]]:
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f'''SELECT spool_path, status FROM "{PG_SCHEMA}".jobs
            WHERE kind = 'document' AND spool_path IS NOT NULL'''
        )
    return [dict(row) for row in rows]


async def prune_spool(spool_root: Path) -> None:
    """Remove terminal and unreferenced spool files during startup."""
    rows = await document_spool_rows()
    active = {row["spool_path"] for row in rows if row["status"] not in TERMINAL_STATUSES}
    terminal = {row["spool_path"] for row in rows if row["status"] in TERMINAL_STATUSES}
    for name in terminal:
        Path(name).unlink(missing_ok=True)
    if spool_root.exists():
        for path in spool_root.iterdir():
            if path.is_file() and str(path) not in active:
                path.unlink(missing_ok=True)
