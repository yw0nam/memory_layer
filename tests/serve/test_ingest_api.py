"""Unit tests for document ingestion jobs, REST validation, and atomic storage."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from memory_base.adapters import document
from memory_base.serve import api, ingest_api


def _post(path, **kwargs):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.post(path, **kwargs)

    return asyncio.run(request())


class AcceptingRegistry:
    def __init__(self):
        self.job = None
        self.runner = None

    def has_capacity(self):
        return True

    def create(self, document_id):
        now = time.time()
        self.job = ingest_api.IngestJob("job-1", document_id, created_at=now, updated_at=now)
        return self.job

    def start(self, job, runner):
        self.runner = runner


@pytest.mark.parametrize("extension", [".doc", ".xls", ".ppt", ".json", ""])
def test_rest_rejects_unsupported_extensions_with_415(extension):
    response = _post(
        "/ingest/document",
        files={"file": (f"legacy{extension}", b"content")},
    )
    assert response.status_code == 415
    assert set(response.json()) == {"error"}


@pytest.mark.parametrize("document_id", ["bad:id", "-bad", "has space", "a" * 122])
def test_rest_rejects_invalid_document_id_with_400(document_id):
    response = _post(
        "/ingest/document",
        data={"document_id": document_id},
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 400
    assert set(response.json()) == {"error"}


def test_rest_normalizes_document_id_and_returns_202(monkeypatch):
    fake = AcceptingRegistry()
    monkeypatch.setattr(ingest_api, "registry", fake)
    response = _post(
        "/ingest/document",
        data={"document_id": "Folder/Guide.MD", "origin": "local:item"},
        files={"file": ("GUIDE.MD", b"content")},
    )
    assert response.status_code == 202
    assert response.json() == {
        "job_id": "job-1",
        "status": "queued",
        "status_url": "/ingest/jobs/job-1",
    }
    assert fake.job.document_id == "guide.md"
    assert fake.runner is not None


def test_rest_returns_429_when_queue_is_full(monkeypatch):
    fake = AcceptingRegistry()
    fake.has_capacity = lambda: False
    monkeypatch.setattr(ingest_api, "registry", fake)
    response = _post(
        "/ingest/document",
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 429
    assert response.json() == {"error": "document ingest queue is full"}


def test_rest_returns_413_when_upload_exceeds_limit(monkeypatch):
    fake = AcceptingRegistry()
    monkeypatch.setattr(ingest_api, "registry", fake)
    monkeypatch.setattr(ingest_api, "INGEST_MAX_BYTES", 5)
    response = _post(
        "/ingest/document",
        files={"file": ("guide.md", b"123456")},
    )
    assert response.status_code == 413
    assert set(response.json()) == {"error"}


def test_job_registry_queue_bound_ttl_and_completed_eviction():
    registry = ingest_api.JobRegistry(
        max_queued=1,
        max_concurrent=1,
        ttl_seconds=10,
        max_completed=2,
    )
    first = registry.create("first.md")
    with pytest.raises(OverflowError):
        registry.create("second.md")

    first.status = "succeeded"
    first.updated_at = 100
    assert asyncio.run(registry.get(first.job_id)) is None

    now = time.time()
    for index in range(3):
        job = registry.create(f"{index}.md")
        job.status = "succeeded"
        job.updated_at = now + index
    registry.cleanup(now + 3)
    assert len(registry.jobs) == 2
    assert [job.document_id for job in registry.jobs.values()] == ["1.md", "2.md"]


def test_job_response_has_exact_status_schema():
    now = time.time()
    job = ingest_api.IngestJob("id", "doc.md", created_at=now, updated_at=now)
    assert set(job.response()) == {
        "job_id",
        "document_id",
        "status",
        "stage",
        "chunks_total",
        "chunks_done",
        "chunks_dropped",
        "rows_written",
        "enrichment_retries",
        "content_hash",
        "error",
        "created_at",
        "updated_at",
    }


class FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class FakeConnection:
    def __init__(self):
        self.calls = []

    def transaction(self):
        return FakeTransaction()

    async def execute(self, query, *args):
        self.calls.append(("execute", query, args))

    async def executemany(self, query, args):
        self.calls.append(("executemany", query, args))


def test_atomic_replacement_deletes_only_document_rows_then_inserts(monkeypatch):
    connection = FakeConnection()

    @asynccontextmanager
    async def acquire():
        yield connection

    monkeypatch.setattr(ingest_api.db, "acquire", acquire)
    row = {
        "id": "doc:guide.md:0",
        "source_type": "document",
        "source_ref": "guide.md",
        "chunk_kind": "doc",
        "session_id": "guide.md",
        "content_raw": "body",
        "distilled": None,
        "embedding": "[0]",
        "ts_last_active": 1.0,
        "idf_score": None,
        "metadata": {"content_hash": "hash"},
    }
    asyncio.run(ingest_api.replace_document_rows("guide.md", [row]))
    delete = next(call for call in connection.calls if "DELETE FROM" in call[1])
    insert = next(call for call in connection.calls if call[0] == "executemany")
    assert "source_type = 'document' AND source_ref = $1" in delete[1]
    assert delete[2] == ("guide.md",)
    assert insert[0] == "executemany"
    assert json.loads(insert[2][0][-1]) == {"content_hash": "hash"}


def _job(document_id="guide.md"):
    now = time.time()
    return ingest_api.IngestJob(
        "job", document_id, status="running", created_at=now, updated_at=now
    )


def test_identical_hash_is_no_op_without_conversion_or_write(monkeypatch, tmp_path):
    upload = tmp_path / "guide.md"
    upload.write_text("same content")
    expected_hash = ingest_api._file_hash(upload)
    called = []

    async def existing(document_id):
        return expected_hash

    async def forbidden(*args, **kwargs):
        called.append(True)

    monkeypatch.setattr(ingest_api, "_existing_content_hash", existing)
    monkeypatch.setattr(ingest_api, "convert_to_markdown", forbidden)
    monkeypatch.setattr(ingest_api, "replace_document_rows", forbidden)
    job = _job()
    asyncio.run(ingest_api.run_document_job(job, upload, "guide.md", "upsert", None))
    assert job.status == "no_op"
    assert job.stage == "done"
    assert called == []
    assert not upload.exists()


def test_csv_persistent_enrichment_failure_names_card_and_never_writes(monkeypatch, tmp_path):
    upload = tmp_path / "data.csv"
    upload.write_text("name,value\none,1\n")
    writes = []

    async def no_existing(document_id):
        return None

    async def fail(*args, **kwargs):
        raise ingest_api.EnrichmentError("enrichment failed after retry")

    async def write(*args, **kwargs):
        writes.append(True)

    monkeypatch.setattr(ingest_api, "_existing_content_hash", no_existing)
    monkeypatch.setattr(ingest_api, "summarize_and_tag", fail)
    monkeypatch.setattr(ingest_api, "replace_document_rows", write)
    with pytest.raises(ingest_api.EnrichmentError, match="CSV card 0"):
        asyncio.run(
            ingest_api.run_document_job(_job("data.csv"), upload, "data.csv", "force", None)
        )
    assert writes == []
    assert not upload.exists()


def test_chunk_persistent_enrichment_failure_names_ordinal_and_never_writes(monkeypatch, tmp_path):
    upload = tmp_path / "guide.md"
    upload.write_text("source")
    writes = []

    async def no_existing(document_id):
        return None

    async def converted(path):
        return document.ConversionResult("Useful document prose. " * 20, "markitdown:0.1.2")

    async def fail(*args, **kwargs):
        raise ingest_api.EnrichmentError("enrichment failed after retry")

    async def write(*args, **kwargs):
        writes.append(True)

    monkeypatch.setattr(ingest_api, "_existing_content_hash", no_existing)
    monkeypatch.setattr(ingest_api, "convert_to_markdown", converted)
    monkeypatch.setattr(ingest_api, "atomize_and_tag", fail)
    monkeypatch.setattr(ingest_api, "replace_document_rows", write)
    with pytest.raises(ingest_api.EnrichmentError, match="chunk 0"):
        asyncio.run(ingest_api.run_document_job(_job(), upload, "guide.md", "force", None))
    assert writes == []
    assert not upload.exists()


def test_zero_accepted_chunks_fails_before_enrichment_and_write(monkeypatch, tmp_path):
    upload = tmp_path / "empty.md"
    upload.write_text("content")
    writes = []

    async def no_existing(document_id):
        return None

    async def converted(path):
        return document.ConversionResult("123" * 100, "markitdown:0.1.6")

    async def write(*args, **kwargs):
        writes.append(True)

    monkeypatch.setattr(ingest_api, "_existing_content_hash", no_existing)
    monkeypatch.setattr(ingest_api, "convert_to_markdown", converted)
    monkeypatch.setattr(ingest_api, "replace_document_rows", write)
    with pytest.raises(document.DocumentError, match="zero accepted chunks"):
        asyncio.run(
            ingest_api.run_document_job(_job("empty.md"), upload, "empty.md", "force", None)
        )
    assert writes == []


def test_oversized_fast_worker_output_is_statted_before_read_and_removed(monkeypatch, tmp_path):
    input_path = tmp_path / "guide.md"
    input_path.write_text("input")
    output = {}

    class FakeProcess:
        returncode = 0

        async def wait(self):
            return 0

        def kill(self):
            self.returncode = -9

    async def spawn(*args):
        output_path = Path(args[-1])
        output["path"] = output_path
        output_path.write_bytes(b"x" * (document.CONVERSION_MAX_BYTES + 1))
        return FakeProcess()

    monkeypatch.setattr(document.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(document.ConversionError, match="exceeds 16 MB"):
        asyncio.run(document.convert_to_markdown(input_path))
    assert not output["path"].exists()
