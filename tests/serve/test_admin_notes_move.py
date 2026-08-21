"""Contract tests for POST /admin/notes/move (red-first).

Route-layer tests mirror tests/serve/test_admin_api.py (monkeypatch
``admin.move_notes`` directly) and tests/serve/test_namespaces_api.py's
``_non_admin_client`` helper for the admin-only gate. Unit tests for
``admin.move_notes`` itself mirror the FakeConnection pattern in
tests/serve/test_rest_notes.py: no DB/network involved.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
from starlette.testclient import TestClient

from memory_base.serve import admin, api, auth, namespaces

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


def _non_admin_client(monkeypatch, label="member", allowed=frozenset({"default"})):
    identity = auth.KeyIdentity(
        key_id=f"{label}-hash", label=label, home="default", is_admin=False, allowed=allowed
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "member-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)
    return TestClient(api.app, headers={"X-API-Key": "member-key"})


# ---- route: delegates and validates ----------------------------------------


def test_move_notes_delegates_to_admin(monkeypatch):
    captured = {}

    async def fake_move_notes(ids, target_namespace):
        captured["ids"] = ids
        captured["target_namespace"] = target_namespace
        return {
            "moved": [{"old": "note:aaaaaaaaaaaaaaaa", "new": "note:team-a:aaaaaaaaaaaaaaaa"}],
            "skipped": [],
        }

    monkeypatch.setattr(admin, "move_notes", fake_move_notes)
    response = client.post(
        "/admin/notes/move", json={"ids": ["note:aaaaaaaaaaaaaaaa"], "namespace": "team-a"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "moved": [{"old": "note:aaaaaaaaaaaaaaaa", "new": "note:team-a:aaaaaaaaaaaaaaaa"}],
        "skipped": [],
    }
    assert captured == {"ids": ["note:aaaaaaaaaaaaaaaa"], "target_namespace": "team-a"}


def test_move_notes_missing_ids_400():
    response = client.post("/admin/notes/move", json={"namespace": "team-a"})
    assert response.status_code == 400
    assert "error" in response.json()


def test_move_notes_empty_ids_400():
    response = client.post("/admin/notes/move", json={"ids": [], "namespace": "team-a"})
    assert response.status_code == 400
    assert "error" in response.json()


@pytest.mark.parametrize("namespace", ["", "   ", 123, [], None])
def test_move_notes_malformed_target_namespace_400(namespace):
    response = client.post(
        "/admin/notes/move", json={"ids": ["note:aaaaaaaaaaaaaaaa"], "namespace": namespace}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "namespace must be a non-empty string"


def test_move_notes_missing_target_namespace_400():
    response = client.post("/admin/notes/move", json={"ids": ["note:aaaaaaaaaaaaaaaa"]})
    assert response.status_code == 400


def test_move_notes_unregistered_target_namespace_400(monkeypatch):
    async def fake_move_notes(ids, target_namespace):
        raise namespaces.NamespaceError(f"unregistered namespace: {target_namespace}")

    monkeypatch.setattr(admin, "move_notes", fake_move_notes)
    response = client.post(
        "/admin/notes/move", json={"ids": ["note:aaaaaaaaaaaaaaaa"], "namespace": "ghost"}
    )
    assert response.status_code == 400
    assert "unregistered namespace" in response.json()["error"]


def test_move_notes_malformed_json_400():
    response = client.post(
        "/admin/notes/move",
        content=b"{not valid json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert "error" in response.json()


def test_move_notes_non_admin_403(monkeypatch):
    async def fail_if_called(ids, target_namespace):
        raise AssertionError("admin.move_notes must not be called for a non-admin caller")

    monkeypatch.setattr(admin, "move_notes", fail_if_called)
    member_client = _non_admin_client(monkeypatch)
    response = member_client.post(
        "/admin/notes/move", json={"ids": ["note:aaaaaaaaaaaaaaaa"], "namespace": "team-a"}
    )
    assert response.status_code == 403
    assert "error" in response.json()


# ---- admin.move_notes: id rewriting, collisions, unit-level (no DB) --------


class FakeTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class FakeConnection:
    def __init__(
        self, existing_notes=frozenset(), registered_namespaces=frozenset(), collisions=frozenset()
    ):
        self.existing_notes = set(existing_notes)
        self.registered_namespaces = set(registered_namespaces)
        self.collisions = set(collisions)
        self.updates: list[tuple] = []

    def transaction(self):
        return FakeTransaction()

    async def fetch(self, query, *args):
        if "source_type = 'agent_note'" in query:
            (ids,) = args
            return [{"id": i} for i in ids if i in self.existing_notes]
        return []

    async def fetchval(self, query, *args):
        if "namespaces" in query:
            (name,) = args
            return name in self.registered_namespaces
        (new_id,) = args
        return new_id in self.collisions

    async def execute(self, query, *args):
        if "UPDATE" in query:
            self.updates.append(args)
        return "UPDATE 1"


async def _noop(conn):
    return None


def _patch_admin_deps(monkeypatch, conn):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(admin.db, "acquire", acquire)
    monkeypatch.setattr(admin, "ensure_schema_once", _noop)


DEFAULT_ID = "note:aaaaaaaaaaaaaaaa"
PRIVATE_ID = "note:team-a:bbbbbbbbbbbbbbbb"


def test_move_notes_rewrites_id_default_to_private(monkeypatch):
    conn = FakeConnection(existing_notes={DEFAULT_ID}, registered_namespaces={"team-a"})
    _patch_admin_deps(monkeypatch, conn)
    result = asyncio.run(admin.move_notes([DEFAULT_ID], "team-a"))
    assert result == {
        "moved": [{"old": DEFAULT_ID, "new": "note:team-a:aaaaaaaaaaaaaaaa"}],
        "skipped": [],
    }
    assert conn.updates == [("note:team-a:aaaaaaaaaaaaaaaa", "team-a", DEFAULT_ID)]


def test_move_notes_rewrites_id_private_to_default(monkeypatch):
    conn = FakeConnection(existing_notes={PRIVATE_ID}, registered_namespaces={"default"})
    _patch_admin_deps(monkeypatch, conn)
    result = asyncio.run(admin.move_notes([PRIVATE_ID], "default"))
    assert result == {
        "moved": [{"old": PRIVATE_ID, "new": "note:bbbbbbbbbbbbbbbb"}],
        "skipped": [],
    }


def test_move_notes_unregistered_target_namespace_raises(monkeypatch):
    conn = FakeConnection(existing_notes={DEFAULT_ID}, registered_namespaces=set())
    _patch_admin_deps(monkeypatch, conn)
    with pytest.raises(namespaces.NamespaceError, match="unregistered namespace"):
        asyncio.run(admin.move_notes([DEFAULT_ID], "ghost"))
    assert conn.updates == []


def test_move_notes_skips_id_collision_in_target(monkeypatch):
    target_id = "note:team-a:aaaaaaaaaaaaaaaa"
    conn = FakeConnection(
        existing_notes={DEFAULT_ID},
        registered_namespaces={"team-a"},
        collisions={target_id},
    )
    _patch_admin_deps(monkeypatch, conn)
    result = asyncio.run(admin.move_notes([DEFAULT_ID], "team-a"))
    assert result == {"moved": [], "skipped": [DEFAULT_ID]}
    assert conn.updates == []


def test_move_notes_skips_ids_that_are_not_agent_notes(monkeypatch):
    conn = FakeConnection(existing_notes=set(), registered_namespaces={"team-a"})
    _patch_admin_deps(monkeypatch, conn)
    result = asyncio.run(admin.move_notes(["note:ghost000000000"], "team-a"))
    assert result == {"moved": [], "skipped": ["note:ghost000000000"]}


def test_move_notes_mixed_batch_reports_both_moved_and_skipped(monkeypatch):
    ok_id = "note:cccccccccccccccc"
    missing_id = "note:dddddddddddddddd"
    conn = FakeConnection(existing_notes={ok_id}, registered_namespaces={"team-a"})
    _patch_admin_deps(monkeypatch, conn)
    result = asyncio.run(admin.move_notes([ok_id, missing_id], "team-a"))
    assert result == {
        "moved": [{"old": ok_id, "new": "note:team-a:cccccccccccccccc"}],
        "skipped": [missing_id],
    }
