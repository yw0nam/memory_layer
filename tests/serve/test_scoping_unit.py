"""Unit tests for non-admin REST namespace enforcement."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from memory_base.serve import api, auth, ingest_api

HOME = "alice-notes"
ALLOWED = {"default", HOME}


def _non_admin_client(monkeypatch):
    identity = auth.KeyIdentity(
        key_id="alice-hash",
        label="alice",
        home=HOME,
        is_admin=False,
        allowed=frozenset(ALLOWED),
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "member-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)
    return TestClient(api.app, headers={"X-API-Key": "member-key"})


def test_search_rejects_namespace_outside_allowed_set(monkeypatch):
    client = _non_admin_client(monkeypatch)
    response = client.post("/search", json={"query": "hello", "namespaces": ["team-b"]})
    assert response.status_code == 403
    assert "error" in response.json()


def test_search_omitted_namespaces_uses_full_allowed_set(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(api, "search", fake_search)
    client = _non_admin_client(monkeypatch)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert captured["namespaces"] == sorted(ALLOWED)


def test_deep_search_rejects_namespace_outside_allowed_set(monkeypatch):
    client = _non_admin_client(monkeypatch)
    response = client.post("/search/deep", json={"query": "hello", "namespaces": ["team-b"]})
    assert response.status_code == 403
    assert "error" in response.json()


def test_save_memory_omitted_namespace_lands_in_key_home(monkeypatch):
    captured = {}

    async def fake_save_note(content, kind="note", tags=None, supersedes=None, namespace="default"):
        captured["namespace"] = namespace
        return {"id": "note:x", "kind": kind, "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(api, "save_note", fake_save_note)
    client = _non_admin_client(monkeypatch)
    response = client.post("/save_memory", json={"content": "distilled note text"})
    assert response.status_code == 200
    assert captured["namespace"] == HOME


def test_save_memory_rejects_namespace_outside_allowed_set(monkeypatch):
    client = _non_admin_client(monkeypatch)
    response = client.post(
        "/save_memory", json={"content": "distilled note text", "namespace": "team-b"}
    )
    assert response.status_code == 403
    assert "error" in response.json()


def test_ingest_document_omitted_namespace_lands_in_key_home(monkeypatch):
    captured = {}

    async def admit(**kwargs):
        captured.update(kwargs)
        return ingest_api.IngestJob.for_document(**kwargs)

    monkeypatch.setattr(ingest_api.job_store, "admit_document", admit)

    async def fake_namespace_exists(name):
        return name == HOME

    monkeypatch.setattr(ingest_api.namespaces, "namespace_exists", fake_namespace_exists)
    client = _non_admin_client(monkeypatch)
    response = client.post("/ingest/document", files={"file": ("guide.md", b"content")})
    assert response.status_code == 202
    assert captured["namespace"] == HOME
    Path(captured["spool_path"]).unlink(missing_ok=True)


def test_ingest_document_rejects_namespace_outside_allowed_set(monkeypatch):
    client = _non_admin_client(monkeypatch)
    response = client.post(
        "/ingest/document",
        data={"namespace": "team-b"},
        files={"file": ("guide.md", b"content")},
    )
    assert response.status_code == 403
    assert "error" in response.json()
