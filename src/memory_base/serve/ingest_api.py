"""Bounded asynchronous document-ingestion REST orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

import asyncpg
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.adapters.document import (
    EXTRACTED_MAX_CHARS,
    Chunk,
    DocumentError,
    UnsupportedDocumentError,
    build_csv_card,
    chunk_markdown,
    convert_to_markdown,
    default_document_id,
    extension_for,
    map_csv_card_row,
    map_document_rows,
    normalize_document_id,
    read_csv_sample,
)
from memory_base.common import DB_URL, PG_SCHEMA, VllmEmbedder, embed_text
from memory_base.ingest.enrich import EnrichmentError, atomize_and_tag, summarize_and_tag
from memory_base.schema import ensure_schema

INGEST_MAX_BYTES = int(os.getenv("INGEST_MAX_BYTES", str(25 * 1024 * 1024)))
INGEST_MAX_QUEUED = int(os.getenv("INGEST_MAX_QUEUED", "10"))
INGEST_MAX_CONCURRENT_JOBS = int(os.getenv("INGEST_MAX_CONCURRENT_JOBS", "2"))
MAX_ACCEPTED_CHUNKS = 2_000
MAX_TOTAL_ROWS = 5_000
JOB_TTL_SECONDS = 24 * 60 * 60
MAX_COMPLETED_JOBS = 100
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "no_op"})


@dataclass
class IngestJob:
    job_id: str
    document_id: str
    status: str = "queued"
    stage: str = "queued"
    chunks_total: int = 0
    chunks_done: int = 0
    chunks_dropped: int = 0
    rows_written: int = 0
    enrichment_retries: int = 0
    content_hash: str | None = None
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def touch(self, *, status: str | None = None, stage: str | None = None) -> None:
        if status is not None:
            self.status = status
        if stage is not None:
            self.stage = stage
        self.updated_at = time.time()

    def response(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = _iso_time(self.created_at)
        payload["updated_at"] = _iso_time(self.updated_at)
        return payload


def _iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


class JobRegistry:
    """Retain bounded job state and run jobs under a concurrency semaphore."""

    def __init__(
        self,
        *,
        max_queued: int = INGEST_MAX_QUEUED,
        max_concurrent: int = INGEST_MAX_CONCURRENT_JOBS,
        ttl_seconds: int = JOB_TTL_SECONDS,
        max_completed: int = MAX_COMPLETED_JOBS,
    ) -> None:
        self.max_queued = max_queued
        self.ttl_seconds = ttl_seconds
        self.max_completed = max_completed
        self.jobs: OrderedDict[str, IngestJob] = OrderedDict()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task[None]] = set()

    def cleanup(self, now: float | None = None) -> None:
        current = time.time() if now is None else now
        expired = [
            job_id
            for job_id, job in self.jobs.items()
            if job.status in TERMINAL_STATUSES and current - job.updated_at >= self.ttl_seconds
        ]
        for job_id in expired:
            self.jobs.pop(job_id, None)

        completed = [job_id for job_id, job in self.jobs.items() if job.status in TERMINAL_STATUSES]
        for job_id in completed[: max(0, len(completed) - self.max_completed)]:
            self.jobs.pop(job_id, None)

    def queued_count(self) -> int:
        return sum(job.status == "queued" for job in self.jobs.values())

    def has_capacity(self) -> bool:
        self.cleanup()
        return self.queued_count() < self.max_queued

    def create(self, document_id: str) -> IngestJob:
        self.cleanup()
        if not self.has_capacity():
            raise OverflowError("document ingest queue is full")
        now = time.time()
        job = IngestJob(
            job_id=uuid.uuid4().hex,
            document_id=document_id,
            created_at=now,
            updated_at=now,
        )
        self.jobs[job.job_id] = job
        return job

    def start(self, job: IngestJob, runner: Callable[[], Awaitable[None]]) -> None:
        async def run() -> None:
            async with self._semaphore:
                job.touch(status="running", stage="converting")
                try:
                    await runner()
                except Exception as exc:
                    job.error = str(exc) or type(exc).__name__
                    job.touch(status="failed", stage="done")
                finally:
                    self.cleanup()

        task = asyncio.create_task(run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def get(self, job_id: str) -> IngestJob | None:
        self.cleanup()
        return self.jobs.get(job_id)


registry = JobRegistry()
_document_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


async def _existing_content_hash(document_id: str) -> str | None:
    conn = await asyncpg.connect(DB_URL)
    try:
        await ensure_schema(conn)
        return await conn.fetchval(
            f"""
            SELECT metadata->>'content_hash'
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'document' AND source_ref = $1
            LIMIT 1
            """,
            document_id,
        )
    finally:
        await conn.close()


async def replace_document_rows(document_id: str, rows: Sequence[dict[str, Any]]) -> None:
    """Replace one document's rows in a single transaction."""
    conn = await asyncpg.connect(DB_URL)
    try:
        await ensure_schema(conn)
        async with conn.transaction():
            await conn.execute(
                f"""
                DELETE FROM "{PG_SCHEMA}".memory_chunks
                WHERE source_type = 'document' AND source_ref = $1
                """,
                document_id,
            )
            await conn.executemany(
                f"""
                INSERT INTO "{PG_SCHEMA}".memory_chunks
                  (id, source_type, source_ref, chunk_kind, session_id, content_raw,
                   distilled, embedding, ts_last_active, idf_score, metadata)
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
                        row["idf_score"],
                        json.dumps(row["metadata"], ensure_ascii=False),
                    )
                    for row in rows
                ],
            )
    finally:
        await conn.close()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_context(chunk: Chunk) -> str:
    if not chunk.heading_path:
        return "This is a chunk from an uploaded document."
    return "Document heading path: " + " > ".join(chunk.heading_path)


async def _enrich_chunks(job: IngestJob, chunks: Sequence[Chunk]) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(4)

    def retried() -> None:
        job.enrichment_retries += 1
        job.touch()

    async def enrich(chunk: Chunk) -> dict[str, Any]:
        try:
            result = await atomize_and_tag(
                chunk.text,
                _chunk_context(chunk),
                semaphore=semaphore,
                on_retry=retried,
            )
        except EnrichmentError as exc:
            raise EnrichmentError(f"chunk {chunk.ordinal}: {exc}") from exc
        job.chunks_done += 1
        job.touch()
        return result

    return list(await asyncio.gather(*(enrich(chunk) for chunk in chunks)))


async def _embed_rows(rows: Sequence[dict[str, Any]]) -> None:
    embedder = VllmEmbedder()
    for row in rows:
        row["embedding"] = await embed_text(embedder, row.pop("embedding_text"))


async def run_document_job(
    job: IngestJob,
    upload_path: Path,
    filename: str,
    mode: str,
    origin: str | None,
) -> None:
    """Run one document pipeline and atomically publish its completed rows."""
    try:
        async with _document_locks[job.document_id]:
            content_hash = _file_hash(upload_path)
            job.content_hash = content_hash
            job.touch()
            if mode == "upsert" and await _existing_content_hash(job.document_id) == content_hash:
                job.touch(status="no_op", stage="done")
                return

            extension = extension_for(filename)
            now = time.time()
            if extension == ".csv":
                job.touch(stage="chunking")
                sample = read_csv_sample(upload_path)
                job.chunks_total = 1
                job.touch(stage="enriching")
                semaphore = asyncio.Semaphore(4)

                def retried() -> None:
                    job.enrichment_retries += 1
                    job.touch()

                async def summarize(text: str, context: str) -> dict[str, Any]:
                    try:
                        return await summarize_and_tag(
                            text,
                            context,
                            semaphore=semaphore,
                            on_retry=retried,
                        )
                    except EnrichmentError as exc:
                        raise EnrichmentError(f"CSV card 0: {exc}") from exc

                card = await build_csv_card(sample, summarize)
                job.chunks_done = 1
                rows = [
                    map_csv_card_row(
                        card,
                        sample,
                        filename=filename,
                        document_id=job.document_id,
                        content_hash=content_hash,
                        origin=origin,
                        timestamp=now,
                    )
                ]
            else:
                conversion = await convert_to_markdown(upload_path)
                if len(conversion.text) > EXTRACTED_MAX_CHARS:
                    raise DocumentError("extracted text exceeds 2000000 chars")
                job.touch(stage="chunking")
                chunking = chunk_markdown(conversion.text)
                chunks = chunking.chunks
                job.chunks_total = len(chunks)
                job.chunks_dropped = chunking.dropped
                if not chunks:
                    raise DocumentError("document produced zero accepted chunks")
                if len(chunks) > MAX_ACCEPTED_CHUNKS:
                    raise DocumentError("document exceeds 2000 accepted chunks")
                job.touch(stage="enriching")
                enrichments = await _enrich_chunks(job, chunks)
                rows = map_document_rows(
                    chunks,
                    enrichments,
                    filename=filename,
                    document_id=job.document_id,
                    content_hash=content_hash,
                    format_name=extension.removeprefix("."),
                    converter=conversion.converter,
                    origin=origin,
                    timestamp=now,
                )

            if not rows:
                raise DocumentError("document produced zero accepted rows")
            if len(rows) > MAX_TOTAL_ROWS:
                raise DocumentError("document exceeds 5000 total rows")
            job.touch(stage="embedding")
            await _embed_rows(rows)
            job.touch(stage="writing")
            await replace_document_rows(job.document_id, rows)
            job.rows_written = len(rows)
            job.touch(status="succeeded", stage="done")
    finally:
        upload_path.unlink(missing_ok=True)


async def _copy_upload(upload: UploadFile, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix="memory-base-upload-", suffix=suffix)
    os.close(descriptor)
    path = Path(name)
    size = 0
    try:
        with path.open("wb") as destination:
            while block := await upload.read(1024 * 1024):
                size += len(block)
                if size > INGEST_MAX_BYTES:
                    raise OverflowError(f"file exceeds {INGEST_MAX_BYTES} bytes")
                destination.write(block)
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


async def ingest_document_route(request: Request) -> JSONResponse:
    """Validate and enqueue a multipart document upload."""
    try:
        form = await request.form()
    except Exception as exc:
        return _error(f"malformed multipart form: {exc}", 400)
    upload = form.get("file")
    if not isinstance(upload, UploadFile) or not upload.filename:
        return _error("file is required", 400)

    filename = PurePath(upload.filename.replace("\\", "/")).name
    try:
        extension = extension_for(filename)
    except UnsupportedDocumentError as exc:
        await upload.close()
        return _error(str(exc), 415)

    raw_document_id = form.get("document_id")
    try:
        if raw_document_id is None or raw_document_id == "":
            document_id = default_document_id(filename)
        elif isinstance(raw_document_id, str):
            document_id = normalize_document_id(raw_document_id)
        else:
            raise DocumentError("document_id must be a string")
    except DocumentError as exc:
        await upload.close()
        return _error(str(exc), 400)

    mode = form.get("mode", "upsert")
    if mode not in {"upsert", "force"}:
        await upload.close()
        return _error("mode must be one of ('upsert', 'force')", 400)
    origin_value = form.get("origin")
    if origin_value is not None and not isinstance(origin_value, str):
        await upload.close()
        return _error("origin must be a string", 400)
    if not registry.has_capacity():
        await upload.close()
        return _error("document ingest queue is full", 429)

    try:
        upload_path = await _copy_upload(upload, extension)
    except OverflowError as exc:
        return _error(str(exc), 413)
    finally:
        await upload.close()

    try:
        job = registry.create(document_id)
    except OverflowError as exc:
        upload_path.unlink(missing_ok=True)
        return _error(str(exc), 429)
    registry.start(
        job,
        lambda: run_document_job(
            job,
            upload_path,
            filename,
            str(mode),
            origin_value,
        ),
    )
    return JSONResponse(
        {
            "job_id": job.job_id,
            "status": job.status,
            "status_url": f"/ingest/jobs/{job.job_id}",
        },
        status_code=202,
    )


async def ingest_job_route(request: Request) -> JSONResponse:
    """Return retained ingestion job state."""
    job = registry.get(request.path_params["job_id"])
    if job is None:
        return _error("ingest job not found", 404)
    return JSONResponse(job.response())
