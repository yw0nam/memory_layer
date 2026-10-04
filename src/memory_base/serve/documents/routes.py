"""REST routes for documents: upload and enqueue, job status, and removal, scoped by the caller's key."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path, PurePath

from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.adapters.document import (
    DocumentError,
    UnsupportedDocumentError,
    extension_for,
    normalize_document_id,
)
from memory_base.core.secrets import find_secret
from memory_base.retrieval.search import normalize_tags
from memory_base.serve import namespaces
from memory_base.serve.common import job_store
from memory_base.serve.common.http import error
from memory_base.serve.documents import pipeline, store

INGEST_MAX_BYTES = int(os.getenv("INGEST_MAX_BYTES", str(25 * 1024 * 1024)))


async def _copy_upload(upload: UploadFile, suffix: str) -> Path:
    pipeline.INGEST_SPOOL.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix="memory-base-upload-", suffix=suffix, dir=pipeline.INGEST_SPOOL
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


async def ingest_route(request: Request) -> JSONResponse:
    """Validate and enqueue a multipart document upload; omitted namespace lands in key.home."""
    key = request.state.key
    try:
        form = await request.form()
    except Exception as exc:
        return error(f"malformed multipart form: {exc}", 400)
    upload = form.get("file")
    if not isinstance(upload, UploadFile) or not upload.filename:
        return error("file is required", 400)

    filename = PurePath(upload.filename.replace("\\", "/")).name
    try:
        extension = extension_for(filename)
    except UnsupportedDocumentError as exc:
        await upload.close()
        return error(str(exc), 415)

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
        return error(str(exc), 400)

    mode = form.get("mode", "upsert")
    if mode not in {"upsert", "force"}:
        await upload.close()
        return error("mode must be one of ('upsert', 'force')", 400)
    origin_value = form.get("origin")
    if origin_value is not None and not isinstance(origin_value, str):
        await upload.close()
        return error("origin must be a string", 400)

    raw_tags = form.getlist("tags")
    if any(not isinstance(tag, str) or not tag.strip() for tag in raw_tags):
        await upload.close()
        return error("tags must be non-empty strings", 400)
    secret_type = find_secret(
        "\n".join([upload.filename, raw_document_id or "", origin_value or "", *raw_tags])
    )
    if secret_type is not None:
        await upload.close()
        return error(
            f"upload metadata contains a credential ({secret_type}); remove it and upload again",
            400,
        )
    tags = normalize_tags(list(raw_tags), allow_empty=True)

    namespace_value = form.get("namespace")
    if namespace_value is None or namespace_value == "":
        namespace = key.home
    elif isinstance(namespace_value, str):
        namespace = namespace_value
    else:
        await upload.close()
        return error("namespace must be a string", 400)
    if not key.permits(namespace):
        await upload.close()
        return error(f"namespace {namespace!r} is outside the caller's allowed set", 403)
    if not await namespaces.namespace_exists(namespace):
        await upload.close()
        return error(f"unregistered namespace: {namespace}", 400)

    exists, existing_created_by = await store.existing_document_owner(document_id, namespace)
    if exists and not key.is_admin and existing_created_by != key.label:
        await upload.close()
        return error("only the document owner or an admin can overwrite this document", 403)

    try:
        upload_path = await _copy_upload(upload, extension)
    except OverflowError as exc:
        return error(str(exc), 413)
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
        return error(str(exc), 429)
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


async def job_route(request: Request) -> JSONResponse:
    """Return durable ingestion job state."""
    job = await job_store.get_job(request.path_params["job_id"], kind="document")
    if job is None:
        return error("ingest job not found", 404)
    return JSONResponse(job.response())


async def remove_route(request: Request) -> JSONResponse:
    """Delete a document's chunks in one namespace; restricted to its creator or an admin.

    `namespace` is a query parameter defaulting to the caller's home namespace.
    """
    key = request.state.key
    document_id = request.path_params["document_id"]
    namespace = request.query_params.get("namespace") or key.home
    if not key.permits(namespace):
        return error(f"namespace {namespace!r} is outside the caller's allowed set", 403)
    exists, created_by = await store.existing_document_owner(document_id, namespace)
    if not exists:
        return error("document not found", 404)
    if not key.is_admin and created_by != key.label:
        return error("only the document owner or an admin can delete this document", 403)
    deleted = await store.delete_document_rows(document_id, namespace)
    return JSONResponse({"document_id": document_id, "namespace": namespace, "deleted": deleted})


async def jobs_route(request: Request) -> JSONResponse:
    """List document jobs newest first within the caller's namespace scope."""
    key = request.state.key
    jobs = await job_store.list_document_jobs(
        namespaces=None if key.is_admin else sorted(key.allowed),
        origin=request.query_params.get("origin"),
        status=request.query_params.get("status"),
    )
    return JSONResponse({"jobs": [job.response() for job in jobs]})
