"""Unit tests for document ingestion jobs, REST validation, and atomic storage."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

import asyncpg

from memory_base.adapters import document
from memory_base.adapters.document import Chunk, map_document_rows
from memory_base.core.config import PG_SCHEMA, VllmEmbedder, db_url, embed_text
from memory_base.ingest import enrich
from memory_base.serve import api, ingest_api, namespaces


@pytest.fixture(autouse=True)
def _default_namespace_registered(monkeypatch):
    """POST /ingest/document checks namespace registration; assume 'default' exists."""

    async def fake_namespace_exists(name):
        return name == "default"

    monkeypatch.setattr(ingest_api.namespaces, "namespace_exists", fake_namespace_exists)


def _post(path, **kwargs):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": "test-key"},
        ) as client:
            return await client.post(path, **kwargs)

    return asyncio.run(request())


class AcceptingBacklog:
    def __init__(self):
        self.job = None
        self.kwargs = None

    async def admit(self, **kwargs):
        self.kwargs = kwargs
        now = time.time()
        self.job = ingest_api.IngestJob(
            job_id="job-1",
            document_id=kwargs["document_id"],
            namespace=kwargs["namespace"],
            origin=kwargs["origin"],
            mode=kwargs["mode"],
            filename=kwargs["filename"],
            spool_path=kwargs["spool_path"],
            key_id=kwargs["key_id"],
            key_label=kwargs["key_label"],
            tags=kwargs["tags"],
            created_at=now,
            updated_at=now,
        )
        return self.job


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
    fake = AcceptingBacklog()
    monkeypatch.setattr(ingest_api.job_store, "admit_document", fake.admit)
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
    assert fake.kwargs["filename"] == "GUIDE.MD"
    Path(fake.kwargs["spool_path"]).unlink(missing_ok=True)


def test_rest_returns_distinct_429_when_per_key_backlog_is_full(monkeypatch):
    async def full(**kwargs):
        raise ingest_api.job_store.BacklogFullError("document per-key backlog limit reached")

    monkeypatch.setattr(ingest_api.job_store, "admit_document", full)
    response = _post(
        "/ingest/document",
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 429
    assert response.json() == {"error": "document per-key backlog limit reached"}


def test_rest_returns_413_when_upload_exceeds_limit(monkeypatch):
    monkeypatch.setattr(ingest_api, "INGEST_MAX_BYTES", 5)
    response = _post(
        "/ingest/document",
        files={"file": ("guide.md", b"123456")},
    )
    assert response.status_code == 413
    assert set(response.json()) == {"error"}


# ---- namespace validation ---------------------------------------------------


def test_rest_ingest_omitted_namespace_uses_default(monkeypatch):
    fake = AcceptingBacklog()
    monkeypatch.setattr(ingest_api.job_store, "admit_document", fake.admit)
    response = _post(
        "/ingest/document",
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 202
    assert fake.kwargs["namespace"] == "default"
    Path(fake.kwargs["spool_path"]).unlink(missing_ok=True)


def test_rest_ingest_unregistered_namespace_400(monkeypatch):
    async def fake_namespace_exists(name):
        return False

    monkeypatch.setattr(ingest_api.namespaces, "namespace_exists", fake_namespace_exists)
    response = _post(
        "/ingest/document",
        data={"namespace": "ghost"},
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 400
    assert "unregistered namespace" in response.json()["error"]


def test_rest_ingest_registered_namespace_is_persisted(monkeypatch):
    fake = AcceptingBacklog()
    monkeypatch.setattr(ingest_api.job_store, "admit_document", fake.admit)

    async def fake_namespace_exists(name):
        return name == "team-a"

    monkeypatch.setattr(ingest_api.namespaces, "namespace_exists", fake_namespace_exists)
    response = _post(
        "/ingest/document",
        data={"namespace": "team-a"},
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 202
    assert fake.kwargs["namespace"] == "team-a"
    Path(fake.kwargs["spool_path"]).unlink(missing_ok=True)


def test_job_response_has_exact_status_schema():
    now = time.time()
    job = ingest_api.IngestJob.for_document(
        job_id="id",
        document_id="doc.md",
        namespace="default",
        origin=None,
        mode="force",
        filename="doc.md",
        spool_path="/spool/id.md",
        key_id="hash",
        key_label="label",
        created_at=now,
        updated_at=now,
    )
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
    asyncio.run(ingest_api.replace_document_rows("guide.md", [row], "default"))
    delete = next(call for call in connection.calls if "DELETE FROM" in call[1])
    insert = next(call for call in connection.calls if call[0] == "executemany")
    assert "source_type = 'document' AND source_ref = $1 AND namespace = $2" in delete[1]
    assert delete[2] == ("guide.md", "default")
    assert insert[0] == "executemany"
    assert insert[2][0][-2] == "default"
    assert json.loads(insert[2][0][-1]) == {"content_hash": "hash"}


def _job(document_id="guide.md", tags=None):
    now = time.time()
    return ingest_api.IngestJob(
        "job",
        document_id,
        status="running",
        tags=list(tags or []),
        created_at=now,
        updated_at=now,
    )


def test_identical_hash_is_no_op_without_conversion_or_write(monkeypatch, tmp_path):
    upload = tmp_path / "guide.md"
    upload.write_text("same content")
    expected_hash = ingest_api._file_hash(upload)
    called = []

    async def existing(document_id, namespace="default", schema=None):
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


def test_force_mode_never_consults_existing_content_hash(monkeypatch, tmp_path):
    """force mode must skip the DB lookup entirely (not just ignore its result):
    a DB-unreachable environment must not hang/crash a force-mode job."""
    upload = tmp_path / "guide.md"
    upload.write_text("force mode content")

    async def forbidden_existing_hash(*args, **kwargs):
        raise AssertionError("force mode must not call _existing_content_hash")

    async def fake_markdown_rows(*args):
        return [{"id": "row-1"}]

    async def no_embed(rows):
        return None

    async def no_write(document_id, rows, namespace="default", schema=None):
        return None

    monkeypatch.setattr(ingest_api, "_existing_content_hash", forbidden_existing_hash)
    monkeypatch.setattr(ingest_api, "_markdown_rows", fake_markdown_rows)
    monkeypatch.setattr(ingest_api, "_embed_rows", no_embed)
    monkeypatch.setattr(ingest_api, "replace_document_rows", no_write)
    job = _job()
    asyncio.run(ingest_api.run_document_job(job, upload, "guide.md", "force", None))
    assert job.status == "succeeded"
    assert job.stage == "done"
    assert upload.exists()


def test_csv_persistent_enrichment_failure_names_card_and_never_writes(monkeypatch, tmp_path):
    upload = tmp_path / "data.csv"
    upload.write_text("name,value\none,1\n")
    writes = []

    async def no_existing(document_id, namespace="default", schema=None):
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
    assert upload.exists()


def test_markdown_ingest_calls_no_llm_and_writes_doc_rows_with_caller_tags(monkeypatch, tmp_path):
    upload = tmp_path / "guide.md"
    upload.write_text("source")
    written = []

    async def no_existing(document_id, namespace="default", schema=None):
        return None

    async def converted(path):
        return document.ConversionResult("Useful document prose. " * 40, "markitdown:0.1.2")

    def forbidden_llm():
        raise AssertionError("document ingest must not call the LLM")

    async def embed(rows):
        for row in rows:
            row["embedding"] = "[0]"
            row.pop("embedding_text")

    async def write(document_id, rows, namespace="default", schema=None):
        written.extend(rows)

    monkeypatch.setattr(ingest_api, "_existing_content_hash", no_existing)
    monkeypatch.setattr(ingest_api, "convert_to_markdown", converted)
    monkeypatch.setattr(enrich, "llm_client", forbidden_llm)
    monkeypatch.setattr(ingest_api, "_embed_rows", embed)
    monkeypatch.setattr(ingest_api, "replace_document_rows", write)
    job = _job(tags=["zx bank", "policy"])
    asyncio.run(ingest_api.run_document_job(job, upload, "guide.md", "force", None))
    assert job.status == "succeeded"
    assert written
    assert {row["chunk_kind"] for row in written} == {"doc"}
    assert all(row["metadata"]["tags"] == ["zx bank", "policy"] for row in written)
    assert job.chunks_done == job.chunks_total == len(written)


def test_rest_repeated_tags_are_normalized_and_persisted_on_the_job(monkeypatch):
    fake = AcceptingBacklog()
    monkeypatch.setattr(ingest_api.job_store, "admit_document", fake.admit)
    response = _post(
        "/ingest/document",
        data={"tags": ["  ZX Bank ", "policy", "zx bank"]},
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 202
    assert fake.kwargs["tags"] == ["zx bank", "policy"]
    assert fake.job.tags == ["zx bank", "policy"]
    Path(fake.kwargs["spool_path"]).unlink(missing_ok=True)


def test_rest_omitted_tags_default_to_empty_list(monkeypatch):
    fake = AcceptingBacklog()
    monkeypatch.setattr(ingest_api.job_store, "admit_document", fake.admit)
    response = _post("/ingest/document", files={"file": ("guide.md", b"content")})
    assert response.status_code == 202
    assert fake.kwargs["tags"] == []
    Path(fake.kwargs["spool_path"]).unlink(missing_ok=True)


def test_rest_rejects_blank_tag_with_400():
    response = _post(
        "/ingest/document",
        data={"tags": ["policy", "   "]},
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 400
    assert set(response.json()) == {"error"}


def test_rest_rejects_non_string_tag_with_400():
    response = _post(
        "/ingest/document",
        files=[("file", ("guide.md", b"content")), ("tags", ("tag.md", b"not a string"))],
    )
    assert response.status_code == 400
    assert set(response.json()) == {"error"}


def test_zero_accepted_chunks_fails_before_enrichment_and_write(monkeypatch, tmp_path):
    upload = tmp_path / "empty.md"
    upload.write_text("content")
    writes = []

    async def no_existing(document_id, namespace="default", schema=None):
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


# ---- integration: same document_id across namespaces must not collide -----


def _db_reachable() -> bool:
    async def _check() -> None:
        conn = await asyncpg.connect(db_url(), timeout=5)
        await conn.close()

    try:
        asyncio.run(_check())
        return True
    except Exception:
        return False


_DB = _db_reachable()
requires_db = pytest.mark.skipif(not _DB, reason="DB is not configured or not reachable")


async def _delete_document_rows(document_id: str) -> None:
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".memory_chunks '
            "WHERE source_type = 'document' AND source_ref = $1",
            document_id,
        )
    finally:
        await conn.close()


@pytest.mark.integration
@requires_db
def test_same_document_id_two_namespaces_no_pk_collision():
    """Regression: ingesting the same document_id into a second namespace must
    not crash with a UniqueViolation on the memory_chunks primary key, since
    replace_document_rows deletes/inserts scoped by (source_ref, namespace)
    but ids used to be namespace-independent (issue #81 review)."""
    document_id = "zzz-namespace-collision-test.md"
    chunks = [Chunk("Namespace collision regression check content.", (), 0)]

    async def scenario():
        await namespaces.create_namespace("zzz-collision-ns")
        try:
            embedding = await embed_text(VllmEmbedder(), "namespace collision check")
            for ns in ("default", "zzz-collision-ns"):
                rows = map_document_rows(
                    chunks,
                    tags=[],
                    filename=document_id,
                    document_id=document_id,
                    content_hash="hash",
                    format_name="md",
                    converter="test",
                    origin=None,
                    timestamp=1.0,
                    namespace=ns,
                )
                for row in rows:
                    row["embedding"] = embedding
                # must not raise asyncpg.UniqueViolationError
                await ingest_api.replace_document_rows(document_id, rows, ns)

            conn = await asyncpg.connect(db_url())
            try:
                count = await conn.fetchval(
                    f'SELECT count(*) FROM "{PG_SCHEMA}".memory_chunks '
                    "WHERE source_type = 'document' AND source_ref = $1",
                    document_id,
                )
            finally:
                await conn.close()
            assert count == 2
        finally:
            await _delete_document_rows(document_id)
            await namespaces.delete_namespace("zzz-collision-ns")

    asyncio.run(scenario())
