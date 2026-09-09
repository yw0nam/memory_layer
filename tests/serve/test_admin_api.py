"""Unit tests for /admin/* REST endpoints and POST /search's include_archived.

No DB, no network: api.py does
``from memory_base.serve import admin`` and calls ``admin.<fn>(...)`` at
request time, so every admin function is monkeypatched here directly on the
``admin`` module (not on ``api``), matching the convention already used for
``api.search``/``api.save_note`` in tests/test_rest_api.py.

Endpoint contract pinned by these tests:

- ``GET /admin/notes?older_than_days=N`` (default 90) -> calls
  ``admin.list_old_notes(N, namespaces=<caller's scope>)``, response body is
  that list verbatim. Non-integer ``older_than_days`` -> 400.
- ``POST /admin/notes/delete {"ids": [...], "confirm": bool}``
  - confirm missing/false (dry-run): calls
    ``admin.notes_by_ids(ids, namespaces=<scope>)``; response
    ``{"rows": <that list>}``; ``admin.delete_notes`` is NOT called.
  - confirm true: calls ``admin.delete_notes(ids, namespaces=<scope>) -> int``;
    response ``{"deleted": <count>}``.
  - missing/empty ``ids``, or malformed JSON -> 400.
- ``GET /admin/duplicates?threshold=0.9&kind=&limit=50`` -> calls
  ``admin.find_duplicates(threshold, kind, limit, namespaces=<scope>)``;
  response ``{"pairs": <that list>}``. Non-numeric ``threshold`` -> 400.
- ``POST /admin/archive {"confirm": bool}``
  - confirm missing/false (dry-run): calls
    ``admin.archive_candidates(now, namespaces=<scope>)``; response
    ``{"candidates": <that list>}``; ``admin.archive_rows`` is NOT called.
  - confirm true: calls ``admin.archive_candidates(now, namespaces=<scope>)``
    to get candidates, then
    ``admin.archive_rows(<candidate ids>, now, namespaces=<scope>) -> int``;
    response ``{"archived": <count>}``.
- ``POST /admin/restore {"ids": [...], "confirm": bool}``
  - confirm missing/false (dry-run): calls
    ``admin.rows_by_ids(ids, namespaces=<scope>)``; response
    ``{"rows": <that list>}``; ``admin.restore_rows`` is NOT called.
  - confirm true: calls ``admin.restore_rows(ids, namespaces=<scope>) -> int``;
    response ``{"restored": <count>}``.
  - missing ``ids`` -> 400.
- ``POST /search`` gains optional ``"include_archived"`` (default False),
  forwarded to ``memory_base.retrieval.search.search`` as the keyword
  ``include_archived``.

``<scope>`` is ``None`` for an admin key (unfiltered) or the caller's
sorted allowed-namespace list otherwise; the fixed ``test-key`` used by the
client here stubs to an admin identity (see tests/serve/conftest.py), so
every call below observes ``namespaces=None``.
"""

from __future__ import annotations

import asyncio

from starlette.testclient import TestClient

from memory_base.serve import admin, api

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


# ---- GET /admin/notes ------------------------------------------------------


def test_admin_notes_defaults_older_than_days_to_90(monkeypatch):
    captured = {}
    rows = [{"id": "note:a", "hit_count": 0, "last_hit_at": None}]

    async def fake_list_old_notes(older_than_days, namespaces=None):
        captured["older_than_days"] = older_than_days
        return rows

    monkeypatch.setattr(admin, "list_old_notes", fake_list_old_notes)
    response = client.get("/admin/notes")
    assert response.status_code == 200
    assert captured["older_than_days"] == 90
    assert response.json() == rows


def test_admin_notes_custom_older_than_days_reaches_admin(monkeypatch):
    captured = {}

    async def fake_list_old_notes(older_than_days, namespaces=None):
        captured["older_than_days"] = older_than_days
        return []

    monkeypatch.setattr(admin, "list_old_notes", fake_list_old_notes)
    response = client.get("/admin/notes", params={"older_than_days": "30"})
    assert response.status_code == 200
    assert captured["older_than_days"] == 30


def test_admin_notes_non_integer_older_than_days_400():
    response = client.get("/admin/notes", params={"older_than_days": "soon"})
    assert response.status_code == 400
    assert "error" in response.json()


# ---- POST /admin/notes/delete ----------------------------------------------


def test_admin_notes_delete_dry_run_by_default(monkeypatch):
    calls = {"notes_by_ids": None, "delete_notes": None}
    rows = [{"id": "note:a", "kind": "agent_note"}]

    async def fake_notes_by_ids(ids, namespaces=None):
        calls["notes_by_ids"] = ids
        return rows

    async def fake_delete_notes(ids, namespaces=None):
        calls["delete_notes"] = ids
        return 999

    monkeypatch.setattr(admin, "notes_by_ids", fake_notes_by_ids)
    monkeypatch.setattr(admin, "delete_notes", fake_delete_notes)
    response = client.post("/admin/notes/delete", json={"ids": ["note:a"]})
    assert response.status_code == 200
    assert response.json() == {"rows": rows}
    assert calls["notes_by_ids"] == ["note:a"]
    assert calls["delete_notes"] is None


def test_admin_notes_delete_confirm_false_is_also_dry_run(monkeypatch):
    calls = {"delete_notes": None}

    async def fake_notes_by_ids(ids, namespaces=None):
        return [{"id": i} for i in ids]

    async def fake_delete_notes(ids, namespaces=None):
        calls["delete_notes"] = ids

    monkeypatch.setattr(admin, "notes_by_ids", fake_notes_by_ids)
    monkeypatch.setattr(admin, "delete_notes", fake_delete_notes)
    response = client.post("/admin/notes/delete", json={"ids": ["note:a"], "confirm": False})
    assert response.status_code == 200
    assert calls["delete_notes"] is None


def test_admin_notes_delete_confirm_true_calls_delete_notes(monkeypatch):
    calls = {}

    async def fake_notes_by_ids(ids, namespaces=None):
        return [{"id": i} for i in ids]

    async def fake_delete_notes(ids, namespaces=None):
        calls["ids"] = ids
        return len(ids)

    monkeypatch.setattr(admin, "notes_by_ids", fake_notes_by_ids)
    monkeypatch.setattr(admin, "delete_notes", fake_delete_notes)
    response = client.post(
        "/admin/notes/delete", json={"ids": ["note:a", "note:b"], "confirm": True}
    )
    assert response.status_code == 200
    assert response.json() == {"deleted": 2}
    assert calls["ids"] == ["note:a", "note:b"]


def test_admin_notes_delete_missing_ids_400():
    response = client.post("/admin/notes/delete", json={})
    assert response.status_code == 400
    assert "error" in response.json()


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


# ---- GET /admin/duplicates --------------------------------------------------


def test_admin_duplicates_defaults(monkeypatch):
    captured = {}
    pairs = [{"a": {"id": "x"}, "b": {"id": "y"}, "score": 0.95}]

    async def fake_find_duplicates(threshold, kind, limit, namespaces=None):
        captured["threshold"] = threshold
        captured["kind"] = kind
        captured["limit"] = limit
        return pairs

    monkeypatch.setattr(admin, "find_duplicates", fake_find_duplicates)
    response = client.get("/admin/duplicates")
    assert response.status_code == 200
    assert captured == {"threshold": 0.9, "kind": None, "limit": 50}
    assert response.json() == {"pairs": pairs}


def test_admin_duplicates_custom_params_reach_admin(monkeypatch):
    captured = {}

    async def fake_find_duplicates(threshold, kind, limit, namespaces=None):
        captured["threshold"] = threshold
        captured["kind"] = kind
        captured["limit"] = limit
        return []

    monkeypatch.setattr(admin, "find_duplicates", fake_find_duplicates)
    response = client.get(
        "/admin/duplicates", params={"threshold": "0.8", "kind": "agent_note", "limit": "5"}
    )
    assert response.status_code == 200
    assert captured == {"threshold": 0.8, "kind": "agent_note", "limit": 5}


def test_admin_duplicates_non_numeric_threshold_400():
    response = client.get("/admin/duplicates", params={"threshold": "high"})
    assert response.status_code == 400
    assert "error" in response.json()


def test_duplicate_pairs_carry_each_sides_author(monkeypatch):
    from contextlib import asynccontextmanager

    row = {
        "a_id": "note:a",
        "a_kind": "note",
        "a_text": "one",
        "a_author": "natsume",
        "b_id": "note:b",
        "b_kind": "note",
        "b_text": "two",
        "b_author": "claude-code",
        "score": 0.97,
    }

    class FakeConnection:
        def __init__(self):
            self.query = None

        async def fetch(self, query, *args):
            self.query = query
            return [row]

    conn = FakeConnection()

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(admin.db, "acquire", acquire)
    pairs = asyncio.run(admin.find_duplicates(0.9, None, 10))
    assert "metadata->>'author'" in conn.query
    assert pairs[0]["a"]["author"] == "natsume"
    assert pairs[0]["b"]["author"] == "claude-code"


# ---- POST /admin/archive -----------------------------------------------------


def test_admin_archive_dry_run_by_default(monkeypatch):
    calls = {"archive_candidates": 0, "archive_rows": None}
    candidates = [{"id": "note:old", "kind": "agent_note", "hit_count": 0, "last_hit_at": None}]

    async def fake_archive_candidates(now, namespaces=None):
        calls["archive_candidates"] += 1
        assert isinstance(now, float)
        return candidates

    async def fake_archive_rows(ids, now, namespaces=None, archived_by=None):
        calls["archive_rows"] = (ids, now)
        return len(ids)

    monkeypatch.setattr(admin, "archive_candidates", fake_archive_candidates)
    monkeypatch.setattr(admin, "archive_rows", fake_archive_rows)
    response = client.post("/admin/archive", json={})
    assert response.status_code == 200
    assert response.json() == {"candidates": candidates}
    assert calls["archive_candidates"] == 1
    assert calls["archive_rows"] is None


def test_admin_archive_confirm_true_archives_exactly_candidate_ids(monkeypatch):
    calls = {}
    candidates = [
        {"id": "note:old1", "kind": "agent_note", "hit_count": 0, "last_hit_at": None},
        {"id": "note:old2", "kind": "history", "hit_count": 2, "last_hit_at": 123.0},
    ]

    async def fake_archive_candidates(now, namespaces=None):
        return candidates

    async def fake_archive_rows(ids, now, namespaces=None, archived_by=None):
        calls["ids"] = ids
        return len(ids)

    monkeypatch.setattr(admin, "archive_candidates", fake_archive_candidates)
    monkeypatch.setattr(admin, "archive_rows", fake_archive_rows)
    response = client.post("/admin/archive", json={"confirm": True})
    assert response.status_code == 200
    assert response.json() == {"archived": 2}
    assert calls["ids"] == ["note:old1", "note:old2"]


def test_admin_archive_with_ids_previews_those_rows(monkeypatch):
    calls = {"rows_by_ids": None, "archive_rows": None, "archive_candidates": 0}
    rows = [{"id": "note:a", "kind": "note", "archived_at": None, "archived_by": None}]

    async def fake_rows_by_ids(ids, namespaces=None):
        calls["rows_by_ids"] = ids
        return rows

    async def fake_archive_candidates(now, namespaces=None):
        calls["archive_candidates"] += 1
        return []

    async def fake_archive_rows(ids, now, namespaces=None, archived_by=None):
        calls["archive_rows"] = ids
        return len(ids)

    monkeypatch.setattr(admin, "rows_by_ids", fake_rows_by_ids)
    monkeypatch.setattr(admin, "archive_candidates", fake_archive_candidates)
    monkeypatch.setattr(admin, "archive_rows", fake_archive_rows)
    response = client.post("/admin/archive", json={"ids": ["note:a"], "author": "natsume"})
    assert response.status_code == 200
    assert response.json() == {"rows": rows}
    assert calls["rows_by_ids"] == ["note:a"]
    assert calls["archive_rows"] is None
    assert calls["archive_candidates"] == 0


def test_admin_archive_with_ids_confirm_stamps_the_author(monkeypatch):
    captured = {}

    async def fake_archive_rows(ids, now, namespaces=None, archived_by=None):
        captured["ids"] = ids
        captured["archived_by"] = archived_by
        return len(ids)

    monkeypatch.setattr(admin, "archive_rows", fake_archive_rows)
    response = client.post(
        "/admin/archive",
        json={"ids": ["note:a", "note:b"], "author": "natsume", "confirm": True},
    )
    assert response.status_code == 200
    assert response.json() == {"archived": 2}
    assert captured == {"ids": ["note:a", "note:b"], "archived_by": "natsume"}


def test_admin_archive_with_ids_requires_an_author():
    response = client.post("/admin/archive", json={"ids": ["note:a"]})
    assert response.status_code == 400
    assert response.json()["error"] == "author is required"


def test_admin_archive_author_outside_the_allowlist_403():
    response = client.post("/admin/archive", json={"ids": ["note:a"], "author": "mallory"})
    assert response.status_code == 403
    assert response.json()["error"] == "author 'mallory' is not permitted for this key"


def test_admin_archive_malformed_ids_400():
    response = client.post("/admin/archive", json={"ids": [], "author": "natsume"})
    assert response.status_code == 400
    assert "error" in response.json()


def test_admin_archive_cold_candidates_stamp_a_given_author(monkeypatch):
    captured = {}
    candidates = [{"id": "note:old", "kind": "note", "hit_count": 0, "last_hit_at": None}]

    async def fake_archive_candidates(now, namespaces=None):
        return candidates

    async def fake_archive_rows(ids, now, namespaces=None, archived_by=None):
        captured["archived_by"] = archived_by
        return len(ids)

    monkeypatch.setattr(admin, "archive_candidates", fake_archive_candidates)
    monkeypatch.setattr(admin, "archive_rows", fake_archive_rows)
    response = client.post("/admin/archive", json={"author": "natsume", "confirm": True})
    assert response.status_code == 200
    assert captured["archived_by"] == "natsume"


def test_admin_archive_without_ids_or_author_stamps_nothing(monkeypatch):
    captured = {}
    candidates = [{"id": "note:old", "kind": "note", "hit_count": 0, "last_hit_at": None}]

    async def fake_archive_candidates(now, namespaces=None):
        return candidates

    async def fake_archive_rows(ids, now, namespaces=None, archived_by=None):
        captured["archived_by"] = archived_by
        return len(ids)

    monkeypatch.setattr(admin, "archive_candidates", fake_archive_candidates)
    monkeypatch.setattr(admin, "archive_rows", fake_archive_rows)
    response = client.post("/admin/archive", json={"confirm": True})
    assert response.status_code == 200
    assert captured["archived_by"] is None


# ---- admin SQL: archived_by is written, cleared, and reported ---------------


class RecordingConnection:
    def __init__(self, rows=None, status="UPDATE 1"):
        self.rows = rows or []
        self.status = status
        self.queries: list[tuple] = []

    async def execute(self, query, *args):
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

    monkeypatch.setattr(admin.db, "acquire", acquire)


def test_archive_rows_stamps_archived_by(monkeypatch):
    conn = RecordingConnection()
    _patch_admin_conn(monkeypatch, conn)
    asyncio.run(admin.archive_rows(["note:a"], 1.0, archived_by="natsume"))
    query, args = conn.queries[0]
    assert "jsonb_build_object('archived_by'" in query
    assert args[-1] == "natsume"


def test_archive_rows_without_an_author_writes_no_archived_by(monkeypatch):
    conn = RecordingConnection()
    _patch_admin_conn(monkeypatch, conn)
    asyncio.run(admin.archive_rows(["note:a"], 1.0))
    query, _ = conn.queries[0]
    assert "archived_by" not in query


def test_restore_rows_clears_archived_by(monkeypatch):
    conn = RecordingConnection()
    _patch_admin_conn(monkeypatch, conn)
    asyncio.run(admin.restore_rows(["note:a"]))
    query, _ = conn.queries[0]
    assert "metadata = metadata - 'archived_by'" in query


def test_rows_by_ids_reports_archived_by(monkeypatch):
    conn = RecordingConnection(rows=[{"id": "note:a", "archived_by": "natsume"}])
    _patch_admin_conn(monkeypatch, conn)
    rows = asyncio.run(admin.rows_by_ids(["note:a"]))
    query, _ = conn.queries[0]
    assert "metadata->>'archived_by'" in query
    assert rows[0]["archived_by"] == "natsume"


# ---- POST /admin/restore -----------------------------------------------------


def test_admin_restore_dry_run_by_default(monkeypatch):
    calls = {"restore_rows": None}
    rows = [{"id": "note:old", "kind": "agent_note"}]

    async def fake_rows_by_ids(ids, namespaces=None):
        return rows

    async def fake_restore_rows(ids, namespaces=None):
        calls["restore_rows"] = ids
        return len(ids)

    monkeypatch.setattr(admin, "rows_by_ids", fake_rows_by_ids)
    monkeypatch.setattr(admin, "restore_rows", fake_restore_rows)
    response = client.post("/admin/restore", json={"ids": ["note:old"]})
    assert response.status_code == 200
    assert response.json() == {"rows": rows}
    assert calls["restore_rows"] is None


def test_admin_restore_confirm_true_calls_restore_rows(monkeypatch):
    calls = {}

    async def fake_rows_by_ids(ids, namespaces=None):
        return [{"id": i} for i in ids]

    async def fake_restore_rows(ids, namespaces=None):
        calls["ids"] = ids
        return len(ids)

    monkeypatch.setattr(admin, "rows_by_ids", fake_rows_by_ids)
    monkeypatch.setattr(admin, "restore_rows", fake_restore_rows)
    response = client.post("/admin/restore", json={"ids": ["a", "b"], "confirm": True})
    assert response.status_code == 200
    assert response.json() == {"restored": 2}
    assert calls["ids"] == ["a", "b"]


def test_admin_restore_missing_ids_400():
    response = client.post("/admin/restore", json={})
    assert response.status_code == 400
    assert "error" in response.json()


# ---- POST /search include_archived ------------------------------------------


def test_search_include_archived_true_reaches_search(monkeypatch):
    captured = {}

    async def fake_search(query, source="all", include_archived=False):
        captured["include_archived"] = include_archived
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello", "include_archived": True})
    assert response.status_code == 200
    assert captured["include_archived"] is True


def test_search_forwards_the_author_filter(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post(
        "/search", json={"query": "hello", "source": "memory", "author": "natsume"}
    )
    assert response.status_code == 200
    assert captured["author"] == "natsume"


def test_search_omitted_author_is_not_forwarded(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert "author" not in captured


def test_search_null_author_400():
    response = client.post("/search", json={"query": "hello", "author": None})
    assert response.status_code == 400
    assert "error" in response.json()


def test_search_include_archived_defaults_to_false(monkeypatch):
    captured = {}

    async def fake_search(query, source="all", include_archived=False):
        captured["include_archived"] = include_archived
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert captured["include_archived"] is False
