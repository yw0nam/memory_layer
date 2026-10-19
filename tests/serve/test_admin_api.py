"""Unit tests for /admin/* REST endpoints and POST /search's include_archived.

No DB, no network: the note routes call ``curation.<fn>(...)`` at request
time, so every curation function is monkeypatched here directly on the
``curation`` module, matching the convention already used for
``search_routes.search``/``store.save_note`` in tests/test_rest_api.py.

Endpoint contract pinned by these tests:

- ``GET /admin/notes?older_than_days=N`` (default 90) -> calls
  ``curation.list_old_notes(N)``, response body is
  that list verbatim. Non-integer ``older_than_days`` -> 400.
- ``POST /admin/notes/delete {"ids": [...], "confirm": bool}``
  - confirm missing/false (dry-run): calls
    ``curation.notes_by_ids(ids)``; response
    ``{"rows": <that list>}``; ``curation.delete_notes`` is NOT called.
  - confirm true: calls ``curation.delete_notes(ids) -> int``;
    response ``{"deleted": <count>}``.
  - missing/empty ``ids``, or malformed JSON -> 400.
- ``GET /admin/duplicates?threshold=0.75&kind=&limit=50`` -> calls
  ``curation.find_duplicates(threshold, kind, limit)``;
  response ``{"pairs": <that list>}``. Non-numeric ``threshold`` -> 400.
- ``POST /admin/archive {"confirm": bool}``
  - confirm missing/false (dry-run): calls
    ``curation.archive_candidates(now)``; response
    ``{"notes_to_archive": <that list>}``; ``curation.archive_rows`` is NOT called.
  - confirm true: archives the note candidates; response ``{"archived": <count>}``.
  - with ``ids``: only the named rows are touched.
- ``POST /admin/restore {"ids": [...], "confirm": bool}``
  - confirm missing/false (dry-run): calls
    ``curation.rows_by_ids(ids)``; response
    ``{"rows": <that list>}``; ``curation.restore_rows`` is NOT called.
  - confirm true: calls ``curation.restore_rows(ids) -> int``;
    response ``{"restored": <count>}``.
  - missing ``ids`` -> 400.
- ``POST /search`` gains optional ``"include_archived"`` (default False),
  forwarded to ``memory_base.retrieval.search.search`` as the keyword
  ``include_archived``.

"""

from __future__ import annotations

import asyncio

from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.notes import curation
from memory_base.serve.search import routes as search_routes

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})


# ---- GET /admin/notes ------------------------------------------------------


def test_admin_notes_defaults_older_than_days_to_90(monkeypatch):
    captured = {}
    rows = [{"id": "note:a", "hit_count": 0, "last_hit_at": None}]

    async def fake_list_old_notes(older_than_days, author=None, visible_to=None):
        captured["older_than_days"] = older_than_days
        return rows

    monkeypatch.setattr(curation, "list_old_notes", fake_list_old_notes)
    response = client.get("/admin/notes")
    assert response.status_code == 200
    assert captured["older_than_days"] == 90
    assert response.json() == rows


def test_admin_notes_custom_older_than_days_reaches_admin(monkeypatch):
    captured = {}

    async def fake_list_old_notes(older_than_days, author=None, visible_to=None):
        captured["older_than_days"] = older_than_days
        return []

    monkeypatch.setattr(curation, "list_old_notes", fake_list_old_notes)
    response = client.get("/admin/notes", params={"older_than_days": "30"})
    assert response.status_code == 200
    assert captured["older_than_days"] == 30


def test_admin_notes_non_integer_older_than_days_400():
    response = client.get("/admin/notes", params={"older_than_days": "soon"})
    assert response.status_code == 400
    assert response.json()["error"] == "older_than_days must be an integer"


def test_admin_duplicates_non_integer_limit_400():
    response = client.get("/admin/duplicates", params={"limit": "lots"})
    assert response.status_code == 400
    assert response.json()["error"] == "limit must be an integer"


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

    async def fake_find_duplicates(threshold, kind, limit, author=None, visible_to=None):
        captured.update(
            threshold=threshold, kind=kind, limit=limit, author=author, visible_to=visible_to
        )
        return []

    monkeypatch.setattr(curation, "find_duplicates", fake_find_duplicates)
    response = client.get("/admin/duplicates")
    assert response.status_code == 200
    assert captured == {
        "threshold": 0.75,
        "kind": None,
        "limit": 50,
        "author": None,
        "visible_to": "claude-code",
    }


def test_admin_duplicates_custom_params_reach_admin(monkeypatch):
    captured = {}

    async def fake_find_duplicates(threshold, kind, limit, author=None, visible_to=None):
        captured.update(
            threshold=threshold, kind=kind, limit=limit, author=author, visible_to=visible_to
        )
        return []

    monkeypatch.setattr(curation, "find_duplicates", fake_find_duplicates)
    response = client.get(
        "/admin/duplicates", params={"threshold": "0.8", "kind": "work", "limit": "5"}
    )
    assert response.status_code == 200
    assert captured == {
        "threshold": 0.8,
        "kind": "work",
        "limit": 5,
        "author": None,
        "visible_to": "claude-code",
    }


def test_admin_duplicates_invalid_author_400():
    response = client.get("/admin/duplicates", params={"author": "Not A Slug"})
    assert response.status_code == 400
    assert response.json()["error"] == 'author must be an author slug or "all"'


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

    monkeypatch.setattr(curation.db, "acquire", acquire)
    pairs = asyncio.run(curation.find_duplicates(0.9, None, 10))
    assert "metadata->>'author'" in conn.query
    assert pairs[0]["a"]["author"] == "natsume"
    assert pairs[0]["b"]["author"] == "claude-code"


def test_duplicates_scope_both_sides(monkeypatch):
    from contextlib import asynccontextmanager

    class FakeConnection:
        def __init__(self):
            self.query = None
            self.args = None

        async def fetch(self, query, *args):
            self.query = query
            self.args = args
            return []

    conn = FakeConnection()

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(curation.db, "acquire", acquire)
    asyncio.run(curation.find_duplicates(0.9, None, 10, visible_to="natsume"))
    lateral_at = conn.query.index("CROSS JOIN LATERAL")
    inner = conn.query[lateral_at:]
    assert "a.metadata @> '{\"shared\": true}'::jsonb" in conn.query
    assert "candidate.metadata @> '{\"shared\": true}'::jsonb" in inner
    assert inner.index("candidate.metadata @>") < inner.index("LIMIT")
    assert "natsume" in conn.args


# ---- POST /admin/archive -----------------------------------------------------


def test_admin_archive_dry_run_by_default(monkeypatch):
    calls = {"archive_candidates": 0, "archive_rows": None}
    candidates = [{"id": "note:old", "kind": "agent_note", "hit_count": 0, "last_hit_at": None}]

    async def fake_archive_candidates(now, author=None, visible_to=None):
        calls["archive_candidates"] += 1
        assert isinstance(now, float)
        return candidates

    async def fake_archive_rows(ids, now, archived_by=None):
        calls["archive_rows"] = (ids, now)
        return len(ids)

    monkeypatch.setattr(curation, "archive_candidates", fake_archive_candidates)
    monkeypatch.setattr(curation, "archive_rows", fake_archive_rows)
    response = client.post("/admin/archive", json={})
    assert response.status_code == 200
    assert response.json() == {"notes_to_archive": candidates}
    assert calls["archive_candidates"] == 1
    assert calls["archive_rows"] is None


def test_admin_archive_confirm_archives_the_cold_notes(monkeypatch):
    calls = {}
    candidates = [
        {"id": "note:old1", "kind": "agent_note", "hit_count": 0, "last_hit_at": None},
        {"id": "note:old2", "kind": "history", "hit_count": 2, "last_hit_at": 123.0},
    ]

    async def fake_archive_candidates(now, author=None, visible_to=None):
        return candidates

    async def fake_archive_rows(ids, now, archived_by=None):
        calls["ids"] = ids
        return len(ids)

    monkeypatch.setattr(curation, "archive_candidates", fake_archive_candidates)
    monkeypatch.setattr(curation, "archive_rows", fake_archive_rows)
    response = client.post("/admin/archive", json={"confirm": True})
    assert response.status_code == 200
    assert response.json() == {"archived": 2}
    assert calls["ids"] == ["note:old1", "note:old2"]


def test_admin_archive_with_ids_previews_those_rows(monkeypatch):
    calls = {"rows_by_ids": None, "archive_rows": None, "archive_candidates": 0}
    rows = [{"id": "note:a", "kind": "note", "archived_at": None, "archived_by": None}]

    async def fake_rows_by_ids(ids, notes_only=False):
        assert notes_only
        calls["rows_by_ids"] = ids
        return rows

    async def fake_archive_candidates(now, author=None, visible_to=None):
        calls["archive_candidates"] += 1
        return []

    async def fake_archive_rows(ids, now, archived_by=None):
        calls["archive_rows"] = ids
        return len(ids)

    monkeypatch.setattr(curation, "rows_by_ids", fake_rows_by_ids)
    monkeypatch.setattr(curation, "archive_candidates", fake_archive_candidates)
    monkeypatch.setattr(curation, "archive_rows", fake_archive_rows)
    response = client.post("/admin/archive", json={"ids": ["note:a"]})
    assert response.status_code == 200
    assert response.json() == {"notes_to_archive": rows}
    assert calls["rows_by_ids"] == ["note:a"]
    assert calls["archive_rows"] is None
    assert calls["archive_candidates"] == 0


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


def test_admin_archive_filter_selection_passes_the_scope(monkeypatch):
    captured = {}

    async def fake_archive_candidates(now, author=None, visible_to=None):
        captured.update(author=author, visible_to=visible_to)
        return []

    monkeypatch.setattr(curation, "archive_candidates", fake_archive_candidates)
    assert client.post("/admin/archive", json={}).status_code == 200
    assert captured == {"author": None, "visible_to": "claude-code"}


def test_admin_archive_filter_selection_confirm_stamps_the_header_author(monkeypatch):
    captured = {}
    candidates = [{"id": "note:old", "kind": "note", "hit_count": 0, "last_hit_at": None}]

    async def fake_archive_candidates(now, author=None, visible_to=None):
        return candidates

    async def fake_archive_rows(ids, now, archived_by=None):
        captured["archived_by"] = archived_by
        return len(ids)

    monkeypatch.setattr(curation, "archive_candidates", fake_archive_candidates)
    monkeypatch.setattr(curation, "archive_rows", fake_archive_rows)
    response = client.post(
        "/admin/archive", json={"confirm": True}, headers={"X-Author": "natsume"}
    )
    assert response.status_code == 200
    assert captured["archived_by"] == "natsume"


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


def test_restore_rows_clears_the_lineage_of_the_archive(monkeypatch):
    conn = RecordingConnection()
    _patch_admin_conn(monkeypatch, conn)
    asyncio.run(curation.restore_rows(["note:a"]))
    query, _ = conn.queries[0]
    for field in ("archived_by", "replaced_by", "consolidated_into"):
        assert f"- '{field}'" in query


def test_rows_by_ids_reports_archived_by(monkeypatch):
    conn = RecordingConnection(rows=[{"id": "note:a", "archived_by": "natsume"}])
    _patch_admin_conn(monkeypatch, conn)
    rows = asyncio.run(curation.rows_by_ids(["note:a"]))
    query, _ = conn.queries[0]
    assert "metadata->>'archived_by'" in query
    assert rows[0]["archived_by"] == "natsume"


# ---- POST /admin/restore -----------------------------------------------------


def test_admin_restore_dry_run_by_default(monkeypatch):
    calls = {"restore_rows": None}
    rows = [{"id": "note:old", "kind": "agent_note"}]

    async def fake_rows_by_ids(
        ids,
    ):
        return rows

    async def fake_restore_rows(
        ids,
    ):
        calls["restore_rows"] = ids
        return len(ids)

    monkeypatch.setattr(curation, "rows_by_ids", fake_rows_by_ids)
    monkeypatch.setattr(curation, "restore_rows", fake_restore_rows)
    response = client.post("/admin/restore", json={"ids": ["note:old"]})
    assert response.status_code == 200
    assert response.json() == {"rows": rows}
    assert calls["restore_rows"] is None


def test_admin_restore_confirm_true_calls_restore_rows(monkeypatch):
    calls = {}

    async def fake_rows_by_ids(
        ids,
    ):
        return [{"id": i} for i in ids]

    async def fake_restore_rows(
        ids,
    ):
        calls["ids"] = ids
        return len(ids)

    monkeypatch.setattr(curation, "rows_by_ids", fake_rows_by_ids)
    monkeypatch.setattr(curation, "restore_rows", fake_restore_rows)
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
    assert captured["author"] == "natsume"


def test_search_omitted_author_is_not_forwarded(monkeypatch):
    captured = {}

    async def fake_search(query, **options):
        captured.update(options)
        return []

    monkeypatch.setattr(search_routes, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert "author" not in captured


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
