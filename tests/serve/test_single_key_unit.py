"""One kind of key: any valid key may call every route; the server grants nothing else."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.access import auth
from memory_base.serve.notes import store as notes_store


@pytest.fixture()
def client(monkeypatch):
    identity = auth.KeyIdentity(key_id="plain-key-hash", label="plain")

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "plain-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)
    return TestClient(api.app, headers={"X-API-Key": "plain-key", "X-Author": "claude-code"})


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
