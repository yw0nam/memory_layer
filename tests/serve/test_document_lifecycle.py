"""Unit tests for document re-ingest and DELETE /ingest/documents/{document_id}."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from memory_base.serve import api
from memory_base.serve.access import namespaces
from memory_base.serve.common import job_store
from memory_base.serve.documents import pipeline as document_pipeline
from memory_base.serve.documents import store as document_store


@pytest.fixture(autouse=True)
def _default_namespace_registered(monkeypatch):
    """POST /ingest/document and run_document_job check namespace registration."""

    async def fake_namespace_exists(name):
        return name in {"default", "team-a"}

    monkeypatch.setattr(namespaces, "namespace_exists", fake_namespace_exists)


def _post(path, api_key="test-key", **kwargs):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": api_key},
        ) as client:
            return await client.post(path, **kwargs)

    return asyncio.run(request())


def _delete(path, api_key="test-key"):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": api_key},
        ) as client:
            return await client.delete(path)

    return asyncio.run(request())


class AcceptingBacklog:
    def __init__(self):
        self.job = None
        self.kwargs = None

    async def admit(self, **kwargs):
        self.kwargs = kwargs
        now = time.time()
        self.job = job_store.IngestJob(
            job_id="job-1",
            document_id=kwargs["document_id"],
            namespace=kwargs["namespace"],
            origin=kwargs["origin"],
            mode=kwargs["mode"],
            filename=kwargs["filename"],
            spool_path=kwargs["spool_path"],
            key_id=kwargs["key_id"],
            tags=kwargs["tags"],
            created_at=now,
            updated_at=now,
        )
        return self.job


def _job(document_id="guide.md"):
    now = time.time()
    return job_store.IngestJob(
        "job",
        document_id,
        status="running",
        created_at=now,
        updated_at=now,
    )


# ---- CSV card write-time bound ---------------------------------------------


def test_run_document_job_rejects_oversized_csv_summary(monkeypatch, tmp_path):
    from memory_base.adapters import document

    upload = tmp_path / "data.csv"
    upload.write_text("name,value\none,1\n")

    async def oversized_summary(text, context, *, semaphore=None, on_retry=None):
        return {"summary": "s" * (document.HARD_CHUNK_CHARS + 1), "tags": ["overflow"]}

    embed_calls = []
    write_calls = []

    async def embed(rows):
        embed_calls.append(rows)

    async def write(document_id, rows, namespace="default", schema=None, table_rows=()):
        write_calls.append(rows)

    monkeypatch.setattr(document_pipeline, "summarize_and_tag", oversized_summary)
    monkeypatch.setattr(document_pipeline, "_embed_rows", embed)
    monkeypatch.setattr(document_store, "replace_document_rows", write)

    job = _job(document_id="data.csv")
    with pytest.raises(document.DocumentError, match=str(document.HARD_CHUNK_CHARS)):
        asyncio.run(document_pipeline.run_document_job(job, upload, "data.csv", "force", None))
    assert not embed_calls
    assert not write_calls


# ---- POST /ingest/document and DELETE /ingest/documents/{document_id} -----------


def test_any_key_overwrites_and_deletes_a_document(monkeypatch):
    fake = AcceptingBacklog()
    monkeypatch.setattr(job_store, "admit_document", fake.admit)
    deleted_calls = []

    async def exists(document_id, namespace="default"):
        return document_id == "guide.md"

    async def delete_rows(document_id, namespace="default"):
        deleted_calls.append((document_id, namespace))
        return 3

    monkeypatch.setattr(document_store, "document_exists", exists)
    monkeypatch.setattr(document_store, "delete_document_rows", delete_rows)

    response = _post(
        "/ingest/document",
        data={"document_id": "guide.md", "mode": "force"},
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 202
    Path(fake.kwargs["spool_path"]).unlink(missing_ok=True)

    response = _delete("/ingest/documents/guide.md?namespace=team-a")
    assert response.status_code == 200
    assert response.json() == {"document_id": "guide.md", "namespace": "team-a", "deleted": 3}
    assert deleted_calls == [("guide.md", "team-a")]
    assert _delete("/ingest/documents/ghost.md").status_code == 404


# ---- integration: DELETE scoped to one namespace ---------------------------


class FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class FakeConnection:
    def __init__(self, rows_by_key):
        self.rows_by_key = rows_by_key
        self.table_rows_by_key = {
            ("guide.md", "default"),
            ("guide.md", "team-a"),
        }
        self.deletes = []

    def transaction(self):
        return FakeTransaction()

    async def fetchval(self, query, *args):
        document_id, namespace = args
        return (document_id, namespace) in self.rows_by_key

    async def execute(self, query, *args):
        if not args:
            return None  # schema bootstrap call from ensure_schema_once
        if ".doc_rows" in query:
            namespace, document_id = args
            key = (document_id, namespace)
            existed = key in self.table_rows_by_key
            self.table_rows_by_key.discard(key)
            self.deletes.append(("table", key))
            return f"DELETE {1 if existed else 0}"
        document_id, namespace = args
        key = (document_id, namespace)
        existed = key in self.rows_by_key
        self.rows_by_key.pop(key, None)
        self.deletes.append(("card", key))
        return f"DELETE {1 if existed else 0}"


def test_delete_document_rows_scoped_to_one_namespace(monkeypatch):
    connection = FakeConnection({("guide.md", "default"): "alice", ("guide.md", "team-a"): "alice"})

    @asynccontextmanager
    async def acquire():
        yield connection

    monkeypatch.setattr(document_store.db, "acquire", acquire)
    deleted = asyncio.run(document_store.delete_document_rows("guide.md", "default"))
    assert deleted == 1
    assert ("guide.md", "team-a") in connection.rows_by_key
    assert ("guide.md", "default") not in connection.rows_by_key
    assert ("guide.md", "team-a") in connection.table_rows_by_key
    assert ("guide.md", "default") not in connection.table_rows_by_key
    assert connection.deletes == [
        ("card", ("guide.md", "default")),
        ("table", ("guide.md", "default")),
    ]
