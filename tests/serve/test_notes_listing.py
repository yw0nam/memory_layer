"""GET /notes: deterministic note listing without a search query.

Route tests monkeypatch ``api.notes.list_notes`` (parse/validate/delegate only);
``notes.list_notes`` itself is exercised against a fake connection, so no DB,
no network, and no embedding call anywhere on this read path.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api, auth, notes
from memory_base.serve.http import TEXT_LIMIT

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

AUG_12 = datetime(2026, 8, 12, tzinfo=timezone.utc).timestamp()


def _capture_list_notes(monkeypatch, rows=None):
    captured = {}

    async def fake_list_notes(**kwargs):
        captured.update(kwargs)
        return rows or []

    monkeypatch.setattr(api.notes, "list_notes", fake_list_notes)
    return captured


# ---- route: parse/validate/delegate ----------------------------------------


def test_list_notes_defaults(monkeypatch):
    captured = _capture_list_notes(monkeypatch)
    response = client.get("/notes")
    assert response.status_code == 200
    assert response.json() == []
    assert captured["tags"] is None
    assert captured["kind"] is None
    assert captured["since"] is None
    assert captured["until"] is None
    assert captured["include_archived"] is False
    assert captured["limit"] == 50
    assert captured["namespaces"] is None  # stubbed identity is admin


def test_list_notes_forwards_every_filter(monkeypatch):
    captured = _capture_list_notes(monkeypatch)
    response = client.get(
        "/notes?tags=infra&tags=db&kind=decision"
        "&since=2026-08-01&until=2026-08-12&limit=5&include_archived=true"
    )
    assert response.status_code == 200
    assert captured["tags"] == ["infra", "db"]
    assert captured["kind"] == "decision"
    assert captured["since"] == "2026-08-01"
    assert captured["until"] == "2026-08-12"
    assert captured["limit"] == 5
    assert captured["include_archived"] is True


def test_list_notes_invalid_limit_400():
    response = client.get("/notes?limit=lots")
    assert response.status_code == 400
    assert "limit" in response.json()["error"]


def test_list_notes_invalid_include_archived_400():
    response = client.get("/notes?include_archived=maybe")
    assert response.status_code == 400
    assert "include_archived" in response.json()["error"]


def test_list_notes_value_error_maps_to_400(monkeypatch):
    async def fake_list_notes(**kwargs):
        raise ValueError("since must be earlier than until")

    monkeypatch.setattr(api.notes, "list_notes", fake_list_notes)
    response = client.get("/notes?since=2026-08-13&until=2026-08-12")
    assert response.status_code == 400
    assert "earlier" in response.json()["error"]


# ---- route: namespace scoping -----------------------------------------------

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


def test_list_notes_omitted_namespace_uses_full_allowed_set(monkeypatch):
    captured = _capture_list_notes(monkeypatch)
    response = _non_admin_client(monkeypatch).get("/notes")
    assert response.status_code == 200
    assert captured["namespaces"] == sorted(ALLOWED)


def test_list_notes_explicit_namespace_narrows_the_scope(monkeypatch):
    captured = _capture_list_notes(monkeypatch)
    response = _non_admin_client(monkeypatch).get(f"/notes?namespace={HOME}")
    assert response.status_code == 200
    assert captured["namespaces"] == [HOME]


def test_list_notes_rejects_namespace_outside_allowed_set(monkeypatch):
    _capture_list_notes(monkeypatch)
    response = _non_admin_client(monkeypatch).get("/notes?namespace=team-b")
    assert response.status_code == 403
    assert "error" in response.json()


# ---- notes.list_notes: SQL shape and row mapping ----------------------------


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows
        self.query: str | None = None
        self.args: tuple | None = None

    async def fetch(self, query, *args):
        self.query = query
        self.args = args
        return self.rows


def _patch_conn(monkeypatch, conn):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(notes.db, "acquire", acquire)


def _row(**overrides):
    row = {
        "id": "note:abc",
        "kind": "note",
        "text": "hello",
        "metadata": {"tags": ["infra"]},
        "ts_last_active": AUG_12,
        "namespace": "default",
        "archived_at": None,
    }
    row.update(overrides)
    return row


def test_list_notes_queries_agent_notes_newest_first(monkeypatch):
    conn = FakeConnection([])
    _patch_conn(monkeypatch, conn)
    asyncio.run(notes.list_notes())
    assert "source_type = 'agent_note'" in conn.query
    assert "ORDER BY ts_last_active DESC" in conn.query
    assert conn.args[0] == 50  # limit rides as $1, matching the predicate offset


def test_list_notes_maps_rows_to_response_shape(monkeypatch):
    conn = FakeConnection([_row()])
    _patch_conn(monkeypatch, conn)
    rows = asyncio.run(notes.list_notes())
    assert rows == [
        {
            "id": "note:abc",
            "kind": "note",
            "text": "hello",
            "tags": ["infra"],
            "namespace": "default",
            "date": "2026-08-12",
        }
    ]


def test_list_notes_marks_archived_rows(monkeypatch):
    conn = FakeConnection([_row(archived_at=AUG_12)])
    _patch_conn(monkeypatch, conn)
    rows = asyncio.run(notes.list_notes(include_archived=True))
    assert rows[0]["archived"] is True


def test_list_notes_truncates_text(monkeypatch):
    conn = FakeConnection([_row(text="x" * (TEXT_LIMIT + 100))])
    _patch_conn(monkeypatch, conn)
    rows = asyncio.run(notes.list_notes())
    assert len(rows[0]["text"]) == TEXT_LIMIT


def test_list_notes_forwards_filters_into_predicates(monkeypatch):
    conn = FakeConnection([])
    _patch_conn(monkeypatch, conn)
    asyncio.run(
        notes.list_notes(
            tags=["Infra "],
            kind="decision",
            namespaces=["team-a"],
            since="2026-08-01",
            until="2026-08-12",
            limit=5,
        )
    )
    assert conn.args[0] == 5
    assert "archived_at IS NULL" in conn.query
    assert ["infra"] in conn.args  # tags normalized like the search filter
    assert ["team-a"] in conn.args


@pytest.mark.parametrize("limit", [0, -1, 201, "many", True])
def test_list_notes_rejects_bad_limit(monkeypatch, limit):
    _patch_conn(monkeypatch, FakeConnection([]))
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(notes.list_notes(limit=limit))


def test_list_notes_rejects_unknown_kind(monkeypatch):
    _patch_conn(monkeypatch, FakeConnection([]))
    with pytest.raises(ValueError, match="kind"):
        asyncio.run(notes.list_notes(kind="reminder"))
