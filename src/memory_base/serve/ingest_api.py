"""Durable asynchronous document-ingestion REST orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePath
from typing import Any

from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.adapters.document import (
    EXTRACTED_MAX_CHARS,
    DocumentError,
    UnsupportedDocumentError,
    build_csv_card,
    chunk_markdown,
    convert_to_markdown,
    extension_for,
    map_csv_card_row,
    map_document_rows,
    normalize_document_id,
    read_csv_sample,
)
from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, VllmEmbedder, embed_text
from memory_base.core.schema import ensure_schema_once
from memory_base.ingest.enrich import EnrichmentError, summarize_and_tag
from memory_base.serve import job_store
from memory_base.serve import namespaces
from memory_base.serve.job_store import _iso_time
from memory_base.serve.notes import normalize_tags

INGEST_MAX_BYTES = int(os.getenv("INGEST_MAX_BYTES", str(25 * 1024 * 1024)))
INGEST_SPOOL = job_store.INGEST_SPOOL
MAX_ACCEPTED_CHUNKS = 2_000
MAX_TOTAL_ROWS = 5_000


@dataclass
class IngestJob:
    job_id: str
    document_id: str
    namespace: str = "default"
    origin: str | None = None
    mode: str = "upsert"
    filename: str = ""
    spool_path: str = ""
    key_id: str = ""
    key_label: str = ""
    tags: list[str] = field(default_factory=list)
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

    @property
    def kind(self) -> str:
        return "document"

    @classmethod
    def for_document(cls, **values):
        now = time.time()
        values.setdefault("created_at", now)
        values.setdefault("updated_at", now)
        return cls(**values)

    @classmethod
    def from_row(cls, row: dict[str, Any]):
        return cls(
            job_id=row["job_id"],
            document_id=row["document_id"],
            namespace=row["namespace"],
            origin=row["origin"],
            mode=row["mode"],
            filename=row["filename"],
            spool_path=row["spool_path"],
            key_id=row["key_id"],
            key_label=row["key_label"],
            tags=list(row["tags"]),
            status=row["status"],
            stage=row["stage"],
            chunks_total=row["chunks_total"],
            chunks_done=row["chunks_done"],
            chunks_dropped=row["chunks_dropped"],
            rows_written=row["rows_written"],
            enrichment_retries=row["enrichment_retries"],
            content_hash=row["content_hash"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def response(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "namespace",
                "origin",
                "mode",
                "filename",
                "spool_path",
                "key_id",
                "key_label",
                "tags",
            }
        }
        payload["created_at"] = _iso_time(self.created_at)
        payload["updated_at"] = _iso_time(self.updated_at)
        return payload


def _error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


async def _existing_content_hash(document_id: str, namespace: str = "default") -> str | None:
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        return await conn.fetchval(
            f"""
            SELECT metadata->>'content_hash'
            FROM "{PG_SCHEMA}".memory_chunks
            WHERE source_type = 'document' AND source_ref = $1 AND namespace = $2
            LIMIT 1
            """,
            document_id,
            namespace,
        )


async def replace_document_rows(
    document_id: str, rows: Sequence[dict[str, Any]], namespace: str = "default"
) -> None:
    """Replace one document's rows, within one namespace, in a single transaction."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await conn.execute(
                f"""
                DELETE FROM "{PG_SCHEMA}".memory_chunks
                WHERE source_type = 'document' AND source_ref = $1 AND namespace = $2
                """,
                document_id,
                namespace,
            )
            await conn.executemany(
                f"""
                INSERT INTO "{PG_SCHEMA}".memory_chunks
                  (id, source_type, source_ref, chunk_kind, session_id, content_raw,
                   distilled, embedding, ts_last_active, idf_score, namespace, metadata)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::halfvec,$9,$10,$11,$12::jsonb)
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
                        row.get("namespace", namespace),
                        json.dumps(row["metadata"], ensure_ascii=False),
                    )
                    for row in rows
                ],
            )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def _embed_rows(rows: Sequence[dict[str, Any]]) -> None:
    embedder = VllmEmbedder()
    for row in rows:
        row["embedding"] = await embed_text(embedder, row.pop("embedding_text"))


async def _csv_rows(
    job: IngestJob,
    upload_path: Path,
    filename: str,
    content_hash: str,
    origin: str | None,
    now: float,
    namespace: str = "default",
) -> list[dict[str, Any]]:
    job.touch(stage="chunking")
    await job_store.update_document_progress(job)
    sample = read_csv_sample(upload_path)
    job.chunks_total = 1
    job.touch(stage="enriching")
    await job_store.update_document_progress(job)
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
    return [
        map_csv_card_row(
            card,
            sample,
            filename=filename,
            document_id=job.document_id,
            content_hash=content_hash,
            origin=origin,
            timestamp=now,
            namespace=namespace,
        )
    ]


async def _markdown_rows(
    job: IngestJob,
    upload_path: Path,
    filename: str,
    extension: str,
    content_hash: str,
    origin: str | None,
    now: float,
    namespace: str = "default",
) -> list[dict[str, Any]]:
    conversion = await convert_to_markdown(upload_path)
    if len(conversion.text) > EXTRACTED_MAX_CHARS:
        raise DocumentError("extracted text exceeds 2000000 chars")
    job.touch(stage="chunking")
    await job_store.update_document_progress(job)
    chunking = chunk_markdown(conversion.text)
    chunks = chunking.chunks
    job.chunks_total = len(chunks)
    job.chunks_dropped = chunking.dropped
    if not chunks:
        raise DocumentError("document produced zero accepted chunks")
    if len(chunks) > MAX_ACCEPTED_CHUNKS:
        raise DocumentError("document exceeds 2000 accepted chunks")
    job.chunks_done = len(chunks)
    await job_store.update_document_progress(job)
    return map_document_rows(
        chunks,
        tags=job.tags,
        filename=filename,
        document_id=job.document_id,
        content_hash=content_hash,
        format_name=extension.removeprefix("."),
        converter=conversion.converter,
        origin=origin,
        timestamp=now,
        namespace=namespace,
    )


async def run_document_job(
    job: IngestJob,
    upload_path: Path | None = None,
    filename: str | None = None,
    mode: str | None = None,
    origin: str | None = None,
    namespace: str | None = None,
) -> None:
    """Run one document pipeline and atomically publish its completed rows."""
    upload_path = upload_path or Path(job.spool_path)
    filename = filename or job.filename
    mode = mode or job.mode
    origin = job.origin if origin is None else origin
    namespace = namespace or job.namespace
    if not await namespaces.namespace_exists(namespace):
        raise RuntimeError(f"namespace was deleted before document job started: {namespace}")
    content_hash = _file_hash(upload_path)
    job.content_hash = content_hash
    await job_store.update_document_progress(job)
    if mode == "upsert":
        existing_hash = await _existing_content_hash(job.document_id, namespace)
        if existing_hash == content_hash:
            job.touch(status="no_op", stage="done")
            return

    extension = extension_for(filename)
    now = time.time()
    if extension == ".csv":
        rows = await _csv_rows(job, upload_path, filename, content_hash, origin, now, namespace)
    else:
        rows = await _markdown_rows(
            job,
            upload_path,
            filename,
            extension,
            content_hash,
            origin,
            now,
            namespace,
        )

    if not rows:
        raise DocumentError("document produced zero accepted rows")
    if len(rows) > MAX_TOTAL_ROWS:
        raise DocumentError("document exceeds 5000 total rows")
    job.touch(stage="embedding")
    await job_store.update_document_progress(job)
    await _embed_rows(rows)
    job.touch(stage="writing")
    await job_store.update_document_progress(job)
    await replace_document_rows(job.document_id, rows, namespace)
    job.rows_written = len(rows)
    job.touch(status="succeeded", stage="done")


async def _copy_upload(upload: UploadFile, suffix: str) -> Path:
    INGEST_SPOOL.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix="memory-base-upload-", suffix=suffix, dir=INGEST_SPOOL
    )
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
    """Validate and enqueue a multipart document upload; omitted namespace lands in key.home."""
    key = request.state.key
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
            document_id = normalize_document_id(filename)
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

    raw_tags = form.getlist("tags")
    if any(not isinstance(tag, str) or not tag.strip() for tag in raw_tags):
        await upload.close()
        return _error("tags must be non-empty strings", 400)
    tags = normalize_tags(list(raw_tags))

    namespace_value = form.get("namespace")
    if namespace_value is None or namespace_value == "":
        namespace = key.home
    elif isinstance(namespace_value, str):
        namespace = namespace_value
    else:
        await upload.close()
        return _error("namespace must be a string", 400)
    if not key.permits(namespace):
        await upload.close()
        return _error(f"namespace {namespace!r} is outside the caller's allowed set", 403)
    if not await namespaces.namespace_exists(namespace):
        await upload.close()
        return _error(f"unregistered namespace: {namespace}", 400)

    try:
        upload_path = await _copy_upload(upload, extension)
    except OverflowError as exc:
        return _error(str(exc), 413)
    finally:
        await upload.close()

    try:
        job = await job_store.admit_document(
            job_id=uuid.uuid4().hex,
            key_id=key.key_id,
            key_label=key.label,
            namespace=namespace,
            document_id=document_id,
            origin=origin_value,
            mode=str(mode),
            filename=filename,
            spool_path=str(upload_path),
            tags=tags,
        )
    except job_store.BacklogFullError as exc:
        upload_path.unlink(missing_ok=True)
        return _error(str(exc), 429)
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise
    return JSONResponse(
        {
            "job_id": job.job_id,
            "status": job.status,
            "status_url": f"/ingest/jobs/{job.job_id}",
        },
        status_code=202,
    )


async def ingest_job_route(request: Request) -> JSONResponse:
    """Return durable ingestion job state."""
    job = await job_store.get_job(request.path_params["job_id"], kind="document")
    if job is None:
        return _error("ingest job not found", 404)
    return JSONResponse(job.response())


async def ingest_jobs_route(request: Request) -> JSONResponse:
    """List document jobs newest first within the caller's namespace scope."""
    key = request.state.key
    jobs = await job_store.list_document_jobs(
        namespaces=None if key.is_admin else sorted(key.allowed),
        origin=request.query_params.get("origin"),
        status=request.query_params.get("status"),
    )
    return JSONResponse({"jobs": [job.response() for job in jobs]})
