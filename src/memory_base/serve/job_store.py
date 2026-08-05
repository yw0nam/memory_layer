"""Durable Postgres admission, dispatch, recovery, and retention for API jobs."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once

INGEST_BACKLOG_PER_KEY = int(os.getenv("INGEST_BACKLOG_PER_KEY", "500"))
INGEST_BACKLOG_MAX = int(os.getenv("INGEST_BACKLOG_MAX", "2000"))
INGEST_MAX_CONCURRENT_JOBS = int(os.getenv("INGEST_MAX_CONCURRENT_JOBS", "2"))
REPO_MAX_QUEUED = int(os.getenv("REPO_MAX_QUEUED", "10"))
JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", str(7 * 24 * 60 * 60)))
INGEST_SPOOL = Path(os.getenv("INGEST_SPOOL", "/data/ingest-spool"))
WORKER_IDLE_SECONDS = 1.0
TERMINAL_STATUSES = frozenset({"succeeded", "no_op", "failed"})


class BacklogFullError(RuntimeError):
    """A durable backlog admission cap is reached."""


def _iso_time(value: float | datetime) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


@asynccontextmanager
async def _connection(connection=None):
    if connection is not None:
        yield connection
        return
    async with db.acquire() as acquired:
        await ensure_schema_once(acquired)
        yield acquired


def _row_to_job(row: Any):
    values = dict(row)
    if values["kind"] == "document":
        from memory_base.serve.ingest_api import IngestJob

        return IngestJob.from_row(values)
    from memory_base.serve.repos import RepoJob

    return RepoJob.from_row(values)


async def _prune_rows(connection) -> None:
    await connection.execute(
        f'''DELETE FROM "{PG_SCHEMA}".jobs
        WHERE status = ANY($1::text[])
          AND updated_at < now() - ($2 * interval '1 second')''',
        list(TERMINAL_STATUSES),
        JOB_RETENTION_SECONDS,
    )


async def admit_document(
    *,
    job_id: str,
    key_id: str,
    key_label: str,
    namespace: str,
    document_id: str,
    origin: str | None,
    mode: str,
    filename: str,
    spool_path: str,
    tags: list[str],
    connection=None,
):
    """Insert a document job atomically with the per-key and global cap checks."""
    async with _connection(connection) as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext('memory-jobs-admission'))")
            await _prune_rows(conn)
            per_key = await conn.fetchval(
                f'''SELECT count(*) FROM "{PG_SCHEMA}".jobs
                WHERE kind = 'document' AND key_id = $1
                  AND status <> ALL($2::text[])''',
                key_id,
                list(TERMINAL_STATUSES),
            )
            if per_key >= INGEST_BACKLOG_PER_KEY:
                raise BacklogFullError(
                    f"document per-key backlog limit reached ({INGEST_BACKLOG_PER_KEY})"
                )
            global_count = await conn.fetchval(
                f'''SELECT count(*) FROM "{PG_SCHEMA}".jobs
                WHERE kind = 'document' AND status <> ALL($1::text[])''',
                list(TERMINAL_STATUSES),
            )
            if global_count >= INGEST_BACKLOG_MAX:
                raise BacklogFullError(
                    f"document global backlog limit reached ({INGEST_BACKLOG_MAX})"
                )
            row = await conn.fetchrow(
                f'''INSERT INTO "{PG_SCHEMA}".jobs
                (job_id, kind, status, key_id, key_label, namespace, document_id,
                 origin, mode, filename, spool_path, stage, tags)
                VALUES ($1, 'document', 'queued', $2, $3, $4, $5, $6, $7, $8, $9, 'queued', $10)
                RETURNING *''',
                job_id,
                key_id,
                key_label,
                namespace,
                document_id,
                origin,
                mode,
                filename,
                spool_path,
                tags,
            )
    return _row_to_job(row)


async def admit_repo(
    *,
    job_id: str,
    key_id: str,
    key_label: str,
    name: str,
    action: str,
    url: str | None,
    branch: str | None,
    connection=None,
):
    """Insert a repo job atomically with the shared admission lock and repo cap."""
    async with _connection(connection) as conn:
        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext('memory-jobs-admission'))")
            await _prune_rows(conn)
            count = await conn.fetchval(
                f'''SELECT count(*) FROM "{PG_SCHEMA}".jobs
                WHERE kind = 'repo' AND status <> ALL($1::text[])''',
                list(TERMINAL_STATUSES),
            )
            if count >= REPO_MAX_QUEUED:
                raise BacklogFullError("repo job queue is full")
            row = await conn.fetchrow(
                f'''INSERT INTO "{PG_SCHEMA}".jobs
                (job_id, kind, status, key_id, key_label, name, action, url, branch)
                VALUES ($1, 'repo', 'queued', $2, $3, $4, $5, $6, $7)
                RETURNING *''',
                job_id,
                key_id,
                key_label,
                name,
                action,
                url,
                branch,
            )
    return _row_to_job(row)


DOCUMENT_CLAIM_SQL = f'''
WITH eligible AS (
  SELECT queued.job_id, queued.key_id, queued.created_at
  FROM "{PG_SCHEMA}".jobs queued
  WHERE queued.kind = 'document' AND queued.status = 'queued'
    AND NOT EXISTS (
      SELECT 1 FROM "{PG_SCHEMA}".jobs active
      WHERE active.kind = 'document' AND active.status = 'running'
        AND active.namespace = queued.namespace
        AND active.document_id = queued.document_id
    )
), key_order AS (
  SELECT eligible.key_id, min(eligible.created_at) AS oldest,
    (SELECT count(*) FROM "{PG_SCHEMA}".jobs active
     WHERE active.kind = 'document' AND active.status = 'running'
       AND active.key_id = eligible.key_id) AS active_count
  FROM eligible
  GROUP BY eligible.key_id
), chosen_key AS (
  SELECT key_id FROM key_order ORDER BY active_count, oldest, key_id LIMIT 1
), candidate AS (
  SELECT jobs.job_id
  FROM "{PG_SCHEMA}".jobs jobs
  JOIN chosen_key ON chosen_key.key_id = jobs.key_id
  WHERE jobs.job_id IN (SELECT job_id FROM eligible)
  ORDER BY jobs.created_at, jobs.job_id
  LIMIT 1
  FOR UPDATE OF jobs SKIP LOCKED
)
UPDATE "{PG_SCHEMA}".jobs jobs
SET status = 'running', updated_at = now(), stage = 'converting', error = NULL,
    chunks_total = 0, chunks_done = 0, chunks_dropped = 0, rows_written = 0,
    enrichment_retries = 0, content_hash = NULL
FROM candidate
WHERE jobs.job_id = candidate.job_id
RETURNING jobs.*
'''

REPO_CLAIM_SQL = f'''
WITH candidate AS (
  SELECT queued.job_id
  FROM "{PG_SCHEMA}".jobs queued
  WHERE queued.kind = 'repo' AND queued.status = 'queued'
    AND NOT EXISTS (
      SELECT 1 FROM "{PG_SCHEMA}".jobs active
      WHERE active.kind = 'repo' AND active.status = 'running'
    )
  ORDER BY queued.created_at, queued.job_id
  LIMIT 1
  FOR UPDATE OF queued SKIP LOCKED
)
UPDATE "{PG_SCHEMA}".jobs jobs
SET status = 'running', updated_at = now(), error = NULL
FROM candidate
WHERE jobs.job_id = candidate.job_id
RETURNING jobs.*
'''


async def claim_job(kind: str, *, connection=None):
    """Claim one job atomically in the kind-specific fair order."""
    if kind not in {"document", "repo"}:
        raise ValueError(f"unsupported job kind: {kind}")
    async with _connection(connection) as conn:
        async with conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext('memory-jobs-claim-' || $1))", kind
            )
            row = await conn.fetchrow(DOCUMENT_CLAIM_SQL if kind == "document" else REPO_CLAIM_SQL)
    return _row_to_job(row) if row else None


async def update_document_progress(job) -> None:
    """Persist the current document stage and attempt counters."""
    if not job.key_id:
        return
    async with _connection() as conn:
        await conn.execute(
            f'''UPDATE "{PG_SCHEMA}".jobs
            SET stage = $2, chunks_total = $3, chunks_done = $4, chunks_dropped = $5,
                rows_written = $6, enrichment_retries = $7, content_hash = $8,
                updated_at = now()
            WHERE job_id = $1''',
            job.job_id,
            job.stage,
            job.chunks_total,
            job.chunks_done,
            job.chunks_dropped,
            job.rows_written,
            job.enrichment_retries,
            job.content_hash,
        )


async def mark_terminal(job, status: str, error: str | None = None) -> None:
    """Persist a terminal result before any document spool cleanup."""
    stage = "done" if getattr(job, "kind", None) == "document" else None
    async with _connection() as conn:
        await conn.execute(
            f'''UPDATE "{PG_SCHEMA}".jobs
            SET status = $2, stage = COALESCE($3, stage), error = $4, updated_at = now(),
                chunks_total = $5, chunks_done = $6, chunks_dropped = $7,
                rows_written = $8, enrichment_retries = $9, content_hash = $10
            WHERE job_id = $1''',
            job.job_id,
            status,
            stage,
            error,
            getattr(job, "chunks_total", 0),
            getattr(job, "chunks_done", 0),
            getattr(job, "chunks_dropped", 0),
            getattr(job, "rows_written", 0),
            getattr(job, "enrichment_retries", 0),
            getattr(job, "content_hash", None),
        )
    job.status = status
    job.error = error
    if stage:
        job.stage = stage


async def get_job(job_id: str, *, kind: str):
    async with _connection() as conn:
        row = await conn.fetchrow(
            f'SELECT * FROM "{PG_SCHEMA}".jobs WHERE job_id = $1 AND kind = $2', job_id, kind
        )
    return _row_to_job(row) if row else None


async def list_document_jobs(
    *, namespaces: list[str] | None, origin: str | None, status: str | None
) -> list[Any]:
    async with _connection() as conn:
        rows = await conn.fetch(
            f'''SELECT * FROM "{PG_SCHEMA}".jobs
            WHERE kind = 'document'
              AND ($1::text[] IS NULL OR namespace = ANY($1::text[]))
              AND ($2::text IS NULL OR origin = $2)
              AND ($3::text IS NULL OR status = $3)
            ORDER BY created_at DESC, job_id DESC
            LIMIT 200''',
            namespaces,
            origin,
            status,
        )
    return [_row_to_job(row) for row in rows]


async def document_spool_rows() -> list[dict[str, Any]]:
    async with _connection() as conn:
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


async def recover_and_prune(spool_root: Path = INGEST_SPOOL) -> None:
    """Requeue interrupted work, fail missing uploads, and apply startup retention."""
    async with _connection() as conn:
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
                f'''UPDATE "{PG_SCHEMA}".jobs
                SET status = 'queued', stage = CASE WHEN kind = 'document' THEN 'queued' ELSE stage END,
                    updated_at = now()
                WHERE status <> ALL($1::text[]) AND NOT (job_id = ANY($2::text[]))''',
                list(TERMINAL_STATUSES),
                missing,
            )
            await _prune_rows(conn)
    await prune_spool(spool_root)


async def initialize() -> None:
    INGEST_SPOOL.mkdir(parents=True, exist_ok=True)
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
    await recover_and_prune(INGEST_SPOOL)


async def _run_claimed(job) -> None:
    if job.kind == "document":
        from memory_base.serve.ingest_api import run_document_job

        try:
            await run_document_job(job)
            await mark_terminal(job, job.status)
        except Exception as exc:
            await mark_terminal(job, "failed", str(exc) or type(exc).__name__)
        Path(job.spool_path).unlink(missing_ok=True)
        return

    from memory_base.serve import repos

    try:
        destination = repos.CACHE_ROOT / job.name
        if job.action == "ingest":
            await repos._run_ingest_job(job.url, destination, job.branch)
        else:
            await repos._run_remove_job(destination)
        await mark_terminal(job, "succeeded")
    except Exception as exc:
        await mark_terminal(job, "failed", str(exc) or type(exc).__name__)


async def worker_loop(
    kind: str,
    *,
    stop: asyncio.Event | None = None,
    claim: Callable[[str], Awaitable[Any | None]] | None = None,
) -> None:
    """Claim and run jobs forever; transient claim faults never stop the worker."""
    stop = stop or asyncio.Event()
    claim = claim or (lambda selected_kind: claim_job(selected_kind))
    while not stop.is_set():
        try:
            job = await claim(kind)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("{} worker claim failed: {}", kind, exc)
            await asyncio.sleep(WORKER_IDLE_SECONDS)
            continue
        if job is None:
            await asyncio.sleep(WORKER_IDLE_SECONDS)
            continue
        try:
            await _run_claimed(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("{} worker execution persistence failed: {}", kind, exc)
            await asyncio.sleep(WORKER_IDLE_SECONDS)


def start_workers() -> list[asyncio.Task[None]]:
    return [
        *(asyncio.create_task(worker_loop("document")) for _ in range(INGEST_MAX_CONCURRENT_JOBS)),
        asyncio.create_task(worker_loop("repo")),
    ]


async def stop_workers(tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
