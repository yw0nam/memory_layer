"""Unit tests for /admin/* REST endpoints and POST /search's include_archived.

No DB, no network: the note routes call ``curation.<fn>(...)`` at request
time, so every curation function is monkeypatched here directly on the
``curation`` module, matching the convention already used for
``search_routes.search``/``store.save_note`` in tests/test_rest_api.py.

Endpoint contract pinned by these tests:

- ``POST /admin/notes/delete {"ids": [...], "confirm": bool}``
  - confirm missing/false (dry-run): calls
    ``curation.notes_by_ids(ids)``; response
    ``{"rows": <that list>}``; ``curation.delete_notes`` is NOT called.
  - confirm true: calls ``curation.delete_notes(ids) -> int``;
    response ``{"deleted": <count>}``.
  - an id that is not an agent note -> 400 naming it in ``missing``.
  - missing/empty ``ids``, or malformed JSON -> 400.
- ``POST /admin/archive {"ids": [...], "confirm": bool}``
  - confirm missing/false (dry-run): response ``{"notes_to_archive": <rows>}``;
    ``curation.archive_rows`` is NOT called.
  - confirm true: response ``{"archived": <count>}``.
  - an id that is not an active note -> 400 naming it in ``missing``; missing ``ids`` -> 400.
- ``POST /search`` gains optional ``"include_archived"`` (default False),
  forwarded to ``memory_base.retrieval.search.search`` as the keyword
  ``include_archived``.

"""

from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.notes import curation
from memory_base.serve.search import routes as search_routes

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})


# ---- POST /admin/notes/delete ----------------------------------------------


def test_admin_notes_delete_dry_run_by_default(monkeypatch):
    calls = {"notes_by_ids": None, "delete_notes": None}
    rows = [{"id": "note:a", "kind": "agent_note"}]

    async def fake_notes_by_ids(
        ids,
    ):
        calls["notes_by_ids"] = ids
        return rows

    async def fake_delete_notes(
        ids,
    ):
        calls["delete_notes"] = ids
        return 999

    monkeypatch.setattr(curation, "notes_by_ids", fake_notes_by_ids)
    monkeypatch.setattr(curation, "delete_notes", fake_delete_notes)
    response = client.post("/admin/notes/delete", json={"ids": ["note:a"]})
    assert response.status_code == 200
    assert response.json() == {"rows": rows}
    assert calls["notes_by_ids"] == ["note:a"]
    assert calls["delete_notes"] is None


def test_admin_notes_delete_confirm_true_calls_delete_notes(monkeypatch):
    calls = {}

    async def fake_notes_by_ids(
        ids,
    ):
        return [{"id": i} for i in ids]

    async def fake_delete_notes(
        ids,
    ):
        calls["ids"] = ids
        return len(ids)

    monkeypatch.setattr(curation, "notes_by_ids", fake_notes_by_ids)
    monkeypatch.setattr(curation, "delete_notes", fake_delete_notes)
    response = client.post(
        "/admin/notes/delete", json={"ids": ["note:a", "note:b"], "confirm": True}
    )
    assert response.status_code == 200
    assert response.json() == {"deleted": 2}
    assert calls["ids"] == ["note:a", "note:b"]


@pytest.mark.parametrize("extra", [{}, {"confirm": True}])
def test_admin_notes_delete_names_ids_it_cannot_find(monkeypatch, extra):
    calls = {"delete_notes": None}

    async def fake_notes_by_ids(ids):
        return [{"id": "note:a"}]

    async def fake_delete_notes(ids):
        calls["delete_notes"] = ids
        return len(ids)

    monkeypatch.setattr(curation, "notes_by_ids", fake_notes_by_ids)
    monkeypatch.setattr(curation, "delete_notes", fake_delete_notes)
    response = client.post("/admin/notes/delete", json={"ids": ["note:a", "note:gone"], **extra})
    assert response.status_code == 400
    assert response.json()["missing"] == ["note:gone"]
    assert calls["delete_notes"] is None


def test_admin_notes_delete_empty_ids_400():
    response = client.post("/admin/notes/delete", json={"ids": []})
    assert response.status_code == 400
    assert "error" in response.json()


def test_admin_notes_delete_malformed_json_400():
    response = client.post(
        "/admin/notes/delete",
        content=b"{not valid json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert "error" in response.json()


# ---- POST /admin/archive -----------------------------------------------------


def test_admin_archive_with_ids_previews_those_rows(monkeypatch):
    calls = {"rows_by_ids": None, "archive_rows": None}
    rows = [{"id": "note:a", "kind": "note", "archived_at": None, "archived_by": None}]

    async def fake_rows_by_ids(ids, notes_only=False):
        assert notes_only
        calls["rows_by_ids"] = ids
        return rows

    async def fake_archive_rows(ids, now, archived_by=None):
        calls["archive_rows"] = ids
        return len(ids)

    monkeypatch.setattr(curation, "rows_by_ids", fake_rows_by_ids)
    monkeypatch.setattr(curation, "archive_rows", fake_archive_rows)
    response = client.post("/admin/archive", json={"ids": ["note:a"]})
    assert response.status_code == 200
    assert response.json() == {"notes_to_archive": rows}
    assert calls["rows_by_ids"] == ["note:a"]
    assert calls["archive_rows"] is None


def test_admin_archive_with_ids_confirm_stamps_the_header_author(monkeypatch):
    captured = {}

    async def fake_rows_by_ids(ids, notes_only=False):
        return [{"id": i} for i in ids]

    async def fake_archive_rows(ids, now, archived_by=None):
        captured["ids"] = ids
        captured["archived_by"] = archived_by
        return len(ids)

    monkeypatch.setattr(curation, "rows_by_ids", fake_rows_by_ids)
    monkeypatch.setattr(curation, "archive_rows", fake_archive_rows)
    response = client.post(
        "/admin/archive",
        json={"ids": ["note:a", "note:b"], "confirm": True},
        headers={"X-Author": "natsume"},
    )
    assert response.status_code == 200
    assert response.json() == {"archived": 2}
    assert captured == {"ids": ["note:a", "note:b"], "archived_by": "natsume"}


def test_admin_archive_with_ids_names_the_ids_not_in_scope(monkeypatch):
    calls = {"archive_rows": None}

    async def fake_rows_by_ids(ids, notes_only=False):
        return [{"id": "note:a"}]

    async def fake_archive_rows(ids, now, archived_by=None):
        calls["archive_rows"] = ids
        return len(ids)

    monkeypatch.setattr(curation, "rows_by_ids", fake_rows_by_ids)
    monkeypatch.setattr(curation, "archive_rows", fake_archive_rows)
    response = client.post(
        "/admin/archive",
        json={"ids": ["note:a", "note:x", "doc:y:0"], "confirm": True},
    )
    assert response.status_code == 400
    assert response.json() == {
        "error": "2 of 3 ids are not active notes: doc:y:0, note:x",
        "missing": ["doc:y:0", "note:x"],
    }
    assert calls["archive_rows"] is None


def test_admin_archive_rejects_a_body_author():
    response = client.post("/admin/archive", json={"ids": ["note:a"], "author": "natsume"})
    assert response.status_code == 400
    assert response.json()["error"] == "author comes from the X-Author header"


def test_admin_archive_malformed_ids_400():
    response = client.post("/admin/archive", json={"ids": []})
    assert response.status_code == 400
    assert "error" in response.json()


def test_admin_archive_refuses_a_body_without_ids():
    response = client.post("/admin/archive", json={"confirm": True})
    assert response.status_code == 400
    assert response.json() == {"error": "ids must be a non-empty list"}


# ---- admin SQL: archived_by is written, cleared, and reported ---------------


class _NoTransaction:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class RecordingConnection:
    def __init__(self, rows=None, status="UPDATE 1"):
        self.rows = rows or []
        self.status = status
        self.queries: list[tuple] = []

    def transaction(self):
        return _NoTransaction()

    async def execute(self, query, *args):
        if "pg_advisory" not in query:
            self.queries.append((query, args))
        return self.status

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        return self.rows


def _patch_admin_conn(monkeypatch, conn):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(curation.db, "acquire", acquire)


def test_archive_rows_stamps_archived_by(monkeypatch):
    conn = RecordingConnection()
    _patch_admin_conn(monkeypatch, conn)
    asyncio.run(curation.archive_rows(["note:a"], 1.0, archived_by="natsume"))
    query, args = conn.queries[0]
    assert "jsonb_build_object('archived_by'" in query
    assert args[-1] == "natsume"


def test_rows_by_ids_reports_archived_by(monkeypatch):
    conn = RecordingConnection(rows=[{"id": "note:a", "archived_by": "natsume"}])
    _patch_admin_conn(monkeypatch, conn)
    rows = asyncio.run(curation.rows_by_ids(["note:a"]))
    query, _ = conn.queries[0]
    assert "metadata->>'archived_by'" in query
    assert rows[0]["archived_by"] == "natsume"


# ---- POST /search include_archived ------------------------------------------


def test_search_include_archived_true_reaches_search(monkeypatch):
    captured = {}

    async def fake_search(query, source="all", include_archived=False, **options):
        captured["include_archived"] = include_archived
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello", "include_archived": True})
    assert response.status_code == 200
    assert captured["include_archived"] is True


def test_search_forwards_the_author_filter(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post(
        "/search", json={"query": "hello", "source": "memory", "author": "natsume"}
    )
    assert response.status_code == 200
    assert captured["authors"] == ["natsume"]


def test_search_omitted_author_is_not_forwarded(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert "authors" not in captured


def test_search_null_author_400():
    response = client.post("/search", json={"query": "hello", "author": None})
    assert response.status_code == 400
    assert "error" in response.json()


def test_search_include_archived_defaults_to_false(monkeypatch):
    captured = {}

    async def fake_search(query, source="all", include_archived=False, **options):
        captured["include_archived"] = include_archived
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert captured["include_archived"] is False
