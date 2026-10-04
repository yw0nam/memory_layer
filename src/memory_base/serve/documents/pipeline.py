"""The document pipeline: convert, chunk or summarize, embed, and publish one document job."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from memory_base.adapters.document import (
    EXTRACTED_MAX_CHARS,
    ConversionResult,
    CredentialDocumentError,
    CSVSample,
    DocumentError,
    build_csv_card,
    chunk_markdown,
    convert_to_markdown,
    csv_text,
    extension_for,
    map_csv_card_row,
    map_csv_table_rows,
    map_document_rows,
    read_csv_sample,
)
from memory_base.core.config import VllmEmbedder, embed_text
from memory_base.core.secrets import find_secret
from memory_base.ingest.enrich import EnrichmentError, summarize_and_tag
from memory_base.serve import namespaces
from memory_base.serve.common import job_store
from memory_base.serve.common.job_store import IngestJob
from memory_base.serve.documents import store

INGEST_SPOOL = Path(os.getenv("INGEST_SPOOL", "/data/ingest-spool"))
INGEST_MAX_CONCURRENT_JOBS = int(os.getenv("INGEST_MAX_CONCURRENT_JOBS", "2"))
MAX_ACCEPTED_CHUNKS = 2_000
MAX_TOTAL_ROWS = 5_000


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
    sample: CSVSample,
    filename: str,
    content_hash: str,
    origin: str | None,
    now: float,
    namespace: str = "default",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    job.touch(stage="chunking")
    await job_store.update_document_progress(job)
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
    return (
        [
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
        ],
        map_csv_table_rows(sample),
    )


async def _markdown_rows(
    job: IngestJob,
    conversion: ConversionResult,
    filename: str,
    extension: str,
    content_hash: str,
    origin: str | None,
    now: float,
    namespace: str = "default",
) -> list[dict[str, Any]]:
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
    schema: str | None = None,
) -> None:
    """Run one document pipeline and atomically publish its completed rows.

    schema overrides PG_SCHEMA for this call; only the eval harness passes it.
    """
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
    extension = extension_for(filename)
    if extension == ".csv":
        sample = read_csv_sample(upload_path)
        text = csv_text(sample)
    else:
        conversion = await convert_to_markdown(upload_path)
        if len(conversion.text) > EXTRACTED_MAX_CHARS:
            raise DocumentError("extracted text exceeds 2000000 chars")
        text = conversion.text
    secret_type = find_secret(text)
    if secret_type is not None:
        raise CredentialDocumentError(secret_type)
    if mode == "upsert":
        existing = await store.existing_document_state(job.document_id, namespace, schema=schema)
        if existing is not None:
            existing_hash, table_rows_loaded = existing
        else:
            existing_hash, table_rows_loaded = None, False
        if existing_hash == content_hash and (extension != ".csv" or table_rows_loaded):
            job.touch(status="no_op", stage="done")
            return

    now = time.time()
    if extension == ".csv":
        rows, table_rows = await _csv_rows(
            job, sample, filename, content_hash, origin, now, namespace
        )
    else:
        rows = await _markdown_rows(
            job,
            conversion,
            filename,
            extension,
            content_hash,
            origin,
            now,
            namespace,
        )
        table_rows = []

    if not rows:
        raise DocumentError("document produced zero accepted rows")
    if len(rows) > MAX_TOTAL_ROWS:
        raise DocumentError("document exceeds 5000 total rows")
    _, existing_created_by = await store.existing_document_owner(
        job.document_id, namespace, schema=schema
    )
    created_by = existing_created_by or job.key_label
    for row in rows:
        row.setdefault("metadata", {})["created_by"] = created_by
    job.touch(stage="embedding")
    await job_store.update_document_progress(job)
    await _embed_rows(rows)
    job.touch(stage="writing")
    await job_store.update_document_progress(job)
    await store.replace_document_rows(
        job.document_id,
        rows,
        namespace,
        schema=schema,
        table_rows=table_rows,
    )
    job.rows_written = len(rows)
    job.touch(status="succeeded", stage="done")


async def run_claimed(job: IngestJob) -> None:
    """Run a claimed job, persist its terminal state, then drop its spool file."""
    try:
        await run_document_job(job)
        await job_store.mark_terminal(job, job.status)
    except Exception as exc:
        await job_store.mark_terminal(job, "failed", str(exc) or type(exc).__name__)
    Path(job.spool_path).unlink(missing_ok=True)


async def start() -> list[asyncio.Task[None]]:
    """Recover interrupted document jobs and spool files, then start the document workers."""
    INGEST_SPOOL.mkdir(parents=True, exist_ok=True)
    await job_store.recover_and_prune("document")
    await job_store.prune_spool(INGEST_SPOOL)
    return [
        asyncio.create_task(job_store.worker_loop("document", run_claimed))
        for _ in range(INGEST_MAX_CONCURRENT_JOBS)
    ]


async def stop(workers: list[asyncio.Task[None]]) -> None:
    await job_store.stop_workers(workers)
