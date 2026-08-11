"""Unit tests for document ownership and deletion: created_by stamping, the
overwrite gate on re-ingest, and DELETE /ingest/documents/{document_id}."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from memory_base.serve import api, auth, ingest_api


def _identity(label, *, is_admin=False):
    return auth.KeyIdentity(
        key_id=f"{label}-hash", label=label, home="default", is_admin=is_admin, allowed=frozenset()
    )


def _auth_map(monkeypatch, mapping):
    async def fake_authenticate_request(plaintext_key):
        return mapping.get(plaintext_key)

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)


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


def _job(document_id="guide.md", key_label="test"):
    now = time.time()
    return ingest_api.IngestJob(
        "job",
        document_id,
        status="running",
        key_label=key_label,
        created_at=now,
        updated_at=now,
    )


# ---- created_by stamping on ingest -----------------------------------------


def test_ingest_stamps_created_by_with_key_label(monkeypatch, tmp_path):
    upload = tmp_path / "guide.md"
    upload.write_text("source")
    written = []

    async def no_owner(document_id, namespace="default", schema=None):
        return False, None

    async def converted(path):
        from memory_base.adapters import document

        return document.ConversionResult("Useful document prose. " * 40, "markitdown:0.1.2")

    async def embed(rows):
        for row in rows:
            row["embedding"] = "[0]"
            row.pop("embedding_text")

    async def write(document_id, rows, namespace="default", schema=None):
        written.extend(rows)

    monkeypatch.setattr(ingest_api, "_existing_document_owner", no_owner)
    monkeypatch.setattr(ingest_api, "convert_to_markdown", converted)
    monkeypatch.setattr(ingest_api, "_embed_rows", embed)
    monkeypatch.setattr(ingest_api, "replace_document_rows", write)
    job = _job(key_label="alice")
    asyncio.run(ingest_api.run_document_job(job, upload, "guide.md", "force", None))
    assert written
    assert all(row["metadata"]["created_by"] == "alice" for row in written)


def test_reingest_preserves_original_created_by(monkeypatch, tmp_path):
    upload = tmp_path / "guide.md"
    upload.write_text("source")
    written = []

    async def existing_owner(document_id, namespace="default", schema=None):
        return True, "alice"

    async def converted(path):
        from memory_base.adapters import document

        return document.ConversionResult("Useful document prose. " * 40, "markitdown:0.1.2")

    async def embed(rows):
        for row in rows:
            row["embedding"] = "[0]"
            row.pop("embedding_text")

    async def write(document_id, rows, namespace="default", schema=None):
        written.extend(rows)

    monkeypatch.setattr(ingest_api, "_existing_document_owner", existing_owner)
    monkeypatch.setattr(ingest_api, "convert_to_markdown", converted)
    monkeypatch.setattr(ingest_api, "_embed_rows", embed)
    monkeypatch.setattr(ingest_api, "replace_document_rows", write)
    job = _job(key_label="root")  # an admin re-ingesting alice's document
    asyncio.run(ingest_api.run_document_job(job, upload, "guide.md", "force", None))
    assert written
    assert all(row["metadata"]["created_by"] == "alice" for row in written)


# ---- REST overwrite gate on POST /ingest/document --------------------------


@pytest.mark.parametrize("mode", ["upsert", "force"])
def test_reingest_by_other_non_admin_key_gets_403(monkeypatch, mode):
    async def existing_owner(document_id, namespace="default", schema=None):
        return True, "alice"

    monkeypatch.setattr(ingest_api, "_existing_document_owner", existing_owner)
    _auth_map(monkeypatch, {"intruder-key": _identity("bob")})
    fake = AcceptingBacklog()
    monkeypatch.setattr(ingest_api.job_store, "admit_document", fake.admit)

    response = _post(
        "/ingest/document",
        api_key="intruder-key",
        data={"document_id": "guide.md", "mode": mode},
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 403
    assert fake.job is None


def test_reingest_by_creator_succeeds(monkeypatch):
    async def existing_owner(document_id, namespace="default", schema=None):
        return True, "alice"

    monkeypatch.setattr(ingest_api, "_existing_document_owner", existing_owner)
    _auth_map(monkeypatch, {"alice-key": _identity("alice")})
    fake = AcceptingBacklog()
    monkeypatch.setattr(ingest_api.job_store, "admit_document", fake.admit)

    response = _post(
        "/ingest/document",
        api_key="alice-key",
        data={"document_id": "guide.md"},
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 202
    Path(fake.kwargs["spool_path"]).unlink(missing_ok=True)


def test_reingest_by_admin_succeeds(monkeypatch):
    async def existing_owner(document_id, namespace="default", schema=None):
        return True, "alice"

    monkeypatch.setattr(ingest_api, "_existing_document_owner", existing_owner)
    _auth_map(monkeypatch, {"admin-key": _identity("root", is_admin=True)})
    fake = AcceptingBacklog()
    monkeypatch.setattr(ingest_api.job_store, "admit_document", fake.admit)

    response = _post(
        "/ingest/document",
        api_key="admin-key",
        data={"document_id": "guide.md"},
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 202
    Path(fake.kwargs["spool_path"]).unlink(missing_ok=True)


def test_reingest_of_ownerless_document_by_non_admin_gets_403(monkeypatch):
    async def existing_owner(document_id, namespace="default", schema=None):
        return True, None

    monkeypatch.setattr(ingest_api, "_existing_document_owner", existing_owner)
    _auth_map(monkeypatch, {"member-key": _identity("member")})
    fake = AcceptingBacklog()
    monkeypatch.setattr(ingest_api.job_store, "admit_document", fake.admit)

    response = _post(
        "/ingest/document",
        api_key="member-key",
        data={"document_id": "guide.md"},
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 403
    assert fake.job is None


def test_ingest_of_new_document_id_is_unaffected_by_gate(monkeypatch):
    async def no_owner(document_id, namespace="default", schema=None):
        return False, None

    monkeypatch.setattr(ingest_api, "_existing_document_owner", no_owner)
    _auth_map(monkeypatch, {"member-key": _identity("member")})
    fake = AcceptingBacklog()
    monkeypatch.setattr(ingest_api.job_store, "admit_document", fake.admit)

    response = _post(
        "/ingest/document",
        api_key="member-key",
        data={"document_id": "new-doc.md"},
        files={"file": ("new-doc.md", b"content")},
    )
    assert response.status_code == 202
    Path(fake.kwargs["spool_path"]).unlink(missing_ok=True)


# ---- DELETE /ingest/documents/{document_id} --------------------------------


def test_delete_by_creator_removes_chunks(monkeypatch):
    async def existing_owner(document_id, namespace="default", schema=None):
        return True, "alice"

    deleted_calls = []

    async def delete_rows(document_id, namespace="default", schema=None):
        deleted_calls.append((document_id, namespace))
        return 3

    monkeypatch.setattr(ingest_api, "_existing_document_owner", existing_owner)
    monkeypatch.setattr(ingest_api, "delete_document_rows", delete_rows)
    _auth_map(monkeypatch, {"alice-key": _identity("alice")})

    response = _delete("/ingest/documents/guide.md", api_key="alice-key")
    assert response.status_code == 200
    assert response.json() == {"document_id": "guide.md", "namespace": "default", "deleted": 3}
    assert deleted_calls == [("guide.md", "default")]


def test_delete_by_admin_removes_chunks(monkeypatch):
    async def existing_owner(document_id, namespace="default", schema=None):
        return True, "alice"

    async def delete_rows(document_id, namespace="default", schema=None):
        return 1

    monkeypatch.setattr(ingest_api, "_existing_document_owner", existing_owner)
    monkeypatch.setattr(ingest_api, "delete_document_rows", delete_rows)
    _auth_map(monkeypatch, {"admin-key": _identity("root", is_admin=True)})

    response = _delete("/ingest/documents/guide.md", api_key="admin-key")
    assert response.status_code == 200


def test_delete_by_other_non_admin_key_gets_403(monkeypatch):
    async def existing_owner(document_id, namespace="default", schema=None):
        return True, "alice"

    called = []

    async def delete_rows(document_id, namespace="default", schema=None):
        called.append(True)
        return 1

    monkeypatch.setattr(ingest_api, "_existing_document_owner", existing_owner)
    monkeypatch.setattr(ingest_api, "delete_document_rows", delete_rows)
    _auth_map(monkeypatch, {"intruder-key": _identity("bob")})

    response = _delete("/ingest/documents/guide.md", api_key="intruder-key")
    assert response.status_code == 403
    assert called == []


def test_delete_of_ownerless_document_by_non_admin_gets_403(monkeypatch):
    async def existing_owner(document_id, namespace="default", schema=None):
        return True, None

    monkeypatch.setattr(ingest_api, "_existing_document_owner", existing_owner)
    _auth_map(monkeypatch, {"member-key": _identity("member")})

    response = _delete("/ingest/documents/guide.md", api_key="member-key")
    assert response.status_code == 403


def test_delete_missing_document_gets_404(monkeypatch):
    async def no_owner(document_id, namespace="default", schema=None):
        return False, None

    monkeypatch.setattr(ingest_api, "_existing_document_owner", no_owner)

    response = _delete("/ingest/documents/ghost.md")
    assert response.status_code == 404


def test_delete_respects_namespace_query_param(monkeypatch):
    captured = []

    async def existing_owner(document_id, namespace="default", schema=None):
        captured.append(("owner", document_id, namespace))
        return True, "alice"

    async def delete_rows(document_id, namespace="default", schema=None):
        captured.append(("delete", document_id, namespace))
        return 2

    monkeypatch.setattr(ingest_api, "_existing_document_owner", existing_owner)
    monkeypatch.setattr(ingest_api, "delete_document_rows", delete_rows)
    _auth_map(monkeypatch, {"alice-key": _identity("alice")})

    response = _delete("/ingest/documents/guide.md?namespace=team-a", api_key="alice-key")
    assert response.status_code == 200
    assert response.json()["namespace"] == "team-a"
    assert captured == [("owner", "guide.md", "team-a"), ("delete", "guide.md", "team-a")]


def test_delete_rejects_namespace_outside_callers_allowed_set(monkeypatch):
    _auth_map(monkeypatch, {"member-key": _identity("member")})

    response = _delete(
        "/ingest/documents/guide.md?namespace=other-team", api_key="member-key"
    )
    assert response.status_code == 403


# ---- integration: DELETE scoped to one namespace ---------------------------


class FakeConnection:
    def __init__(self, rows_by_key):
        self.rows_by_key = rows_by_key
        self.deletes = []

    async def fetchrow(self, query, *args):
        document_id, namespace = args
        row = self.rows_by_key.get((document_id, namespace))
        return {"created_by": row} if row is not None else None

    async def execute(self, query, *args):
        if not args:
            return None  # schema bootstrap call from ensure_schema_once
        document_id, namespace = args
        key = (document_id, namespace)
        existed = key in self.rows_by_key
        self.rows_by_key.pop(key, None)
        self.deletes.append(key)
        return f"DELETE {1 if existed else 0}"


def test_delete_document_rows_scoped_to_one_namespace(monkeypatch):
    connection = FakeConnection({("guide.md", "default"): "alice", ("guide.md", "team-a"): "alice"})

    @asynccontextmanager
    async def acquire():
        yield connection

    monkeypatch.setattr(ingest_api.db, "acquire", acquire)
    deleted = asyncio.run(ingest_api.delete_document_rows("guide.md", "default"))
    assert deleted == 1
    assert ("guide.md", "team-a") in connection.rows_by_key
    assert ("guide.md", "default") not in connection.rows_by_key
