"""One kind of key: any valid key may call every route; the server grants nothing else."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.access import auth
from memory_base.serve.consolidation import verdicts
from memory_base.serve.facts import store as facts_store
from memory_base.serve.notes import curation
from memory_base.serve.notes import store as notes_store
from memory_base.serve.turns import store as turns_store


@pytest.fixture()
def client(monkeypatch):
    identity = auth.KeyIdentity(key_id="plain-key-hash", label="plain")

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "plain-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)
    return TestClient(api.app, headers={"X-API-Key": "plain-key", "X-Author": "claude-code"})


def test_a_plain_key_dry_runs_a_fact_write(client, monkeypatch):
    async def fake_apply(ops, author, dry_run):
        return [{"op": "add", "dry_run": dry_run, "author": author}]

    monkeypatch.setattr(facts_store, "apply", fake_apply)
    op = {"op": "add", "entity": "natsume", "attribute": "city", "value_text": "Busan",
          "note_id": "note:a"}  # fmt: skip
    response = client.post("/admin/facts", json={"ops": [op], "dry_run": True})
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["author"] == "claude-code"


def test_a_plain_key_lists_consolidation_actions(client, monkeypatch):
    async def fake_list_actions(**filters):
        return {"actions": []}

    monkeypatch.setattr(verdicts, "list_actions", fake_list_actions)
    assert client.get("/admin/consolidate/actions").status_code == 200


def test_a_plain_key_moves_notes(client, monkeypatch):
    async def fake_move_notes(ids, namespace):
        return {"moved": [], "skipped": []}

    monkeypatch.setattr(curation, "move_notes", fake_move_notes)
    response = client.post(
        "/admin/notes/move", json={"ids": ["note:aaaaaaaaaaaaaaaa"], "namespace": "default"}
    )
    assert response.status_code == 200, response.text


def test_a_plain_key_deletes_a_session(client, monkeypatch):
    async def fake_delete_session(origin, session_id):
        return 3

    monkeypatch.setattr(turns_store, "delete_session", fake_delete_session)
    response = client.delete("/admin/sessions/claude_code/s1")
    assert response.status_code == 200
    assert response.json() == {"deleted": 3}


def test_a_request_without_a_valid_key_is_refused(client):
    assert TestClient(api.app).get("/notes").status_code == 401
    assert TestClient(api.app, headers={"X-API-Key": "revoked"}).get("/notes").status_code == 401


def test_a_request_without_an_author_header_is_refused(client):
    response = TestClient(api.app, headers={"X-API-Key": "plain-key"}).get("/notes")
    assert response.status_code == 400
    assert "X-Author" in response.json()["error"]


@pytest.fixture()
def saved(monkeypatch):
    calls = []

    async def fake_save_note(content, **fields):
        calls.append(fields)
        return {"id": "note:a"}

    monkeypatch.setattr(notes_store, "save_note", fake_save_note)
    return calls


@pytest.mark.parametrize("author", ["Bad Author!", "-lead", "x" * 41, "all"])
def test_an_author_header_that_is_not_a_slug_is_refused(client, saved, author):
    response = client.post(
        "/save_memory",
        json={"content": "a note"},
        headers={"X-Author": author},
    )
    assert response.status_code == 400
    assert saved == []


def test_any_author_slug_is_accepted_and_a_write_lands_in_default(client, saved):
    response = client.post(
        "/save_memory", json={"content": "a note"}, headers={"X-Author": "new-agent"}
    )
    assert response.status_code == 200, response.text
    assert saved[0]["author"] == "new-agent"
    assert saved[0]["namespace"] == "default"


def test_a_note_list_without_a_namespace_covers_every_namespace(client, monkeypatch):
    seen = {}

    async def fake_list_notes(**filters):
        seen.update(filters)
        return []

    monkeypatch.setattr(notes_store, "list_notes", fake_list_notes)
    assert client.get("/notes").status_code == 200
    assert seen["namespaces"] is None


def test_a_document_upload_without_a_namespace_lands_in_default(client, monkeypatch):
    from pathlib import Path

    from memory_base.serve.access import namespaces
    from memory_base.serve.common import job_store

    captured = {}

    async def admit(**kwargs):
        captured.update(kwargs)
        return job_store.IngestJob(**kwargs)

    async def namespace_exists(name):
        return name == "default"

    monkeypatch.setattr(job_store, "admit_document", admit)
    monkeypatch.setattr(namespaces, "namespace_exists", namespace_exists)
    response = client.post("/ingest/document", files={"file": ("guide.md", b"content")})
    assert response.status_code == 202, response.text
    assert captured["namespace"] == "default"
    Path(captured["spool_path"]).unlink(missing_ok=True)
