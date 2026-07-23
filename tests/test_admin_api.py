"""Unit tests for /admin/* REST endpoints and POST /search's include_archived.

No DB, no network: per docs/specs/impl_lifecycle.md, api.py does
``from memory_base.serve import admin`` and calls ``admin.<fn>(...)`` at
request time, so every admin function is monkeypatched here directly on the
``admin`` module (not on ``api``), matching the convention already used for
``api.search``/``api.save_note`` in tests/test_rest_api.py.

Endpoint contract pinned by these tests:

- ``GET /admin/notes?older_than_days=N`` (default 90) -> calls
  ``admin.list_old_notes(N)``, response body is that list verbatim.
  Non-integer ``older_than_days`` -> 400.
- ``POST /admin/notes/delete {"ids": [...], "confirm": bool}``
  - confirm missing/false (dry-run): calls ``admin.notes_by_ids(ids)``;
    response ``{"rows": <that list>}``; ``admin.delete_notes`` is NOT called.
  - confirm true: calls ``admin.delete_notes(ids) -> int``; response
    ``{"deleted": <count>}``.
  - missing/empty ``ids``, or malformed JSON -> 400.
- ``GET /admin/duplicates?threshold=0.9&kind=&limit=50`` -> calls
  ``admin.find_duplicates(threshold, kind, limit)``; response
  ``{"pairs": <that list>}``. Non-numeric ``threshold`` -> 400.
- ``POST /admin/archive {"confirm": bool}``
  - confirm missing/false (dry-run): calls ``admin.archive_candidates(now)``;
    response ``{"candidates": <that list>}``; ``admin.archive_rows`` is NOT
    called.
  - confirm true: calls ``admin.archive_candidates(now)`` to get candidates,
    then ``admin.archive_rows(<candidate ids>, now) -> int``; response
    ``{"archived": <count>}``.
- ``POST /admin/restore {"ids": [...], "confirm": bool}``
  - confirm missing/false (dry-run): calls ``admin.rows_by_ids(ids)``;
    response ``{"rows": <that list>}``; ``admin.restore_rows`` is NOT called.
  - confirm true: calls ``admin.restore_rows(ids) -> int``; response
    ``{"restored": <count>}``.
  - missing ``ids`` -> 400.
- ``POST /search`` gains optional ``"include_archived"`` (default False),
  forwarded to ``memory_base.retrieval.search.search`` as the keyword
  ``include_archived``.

Collection fails today: memory_base.serve.admin does not exist yet, and
api.py has no /admin/* routes or include_archived wiring.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from memory_base.serve import admin, api

client = TestClient(api.app)


# ---- GET /admin/notes ------------------------------------------------------


def test_admin_notes_defaults_older_than_days_to_90(monkeypatch):
    captured = {}
    rows = [{"id": "note:a", "hit_count": 0, "last_hit_at": None}]

    def fake_list_old_notes(older_than_days):
        captured["older_than_days"] = older_than_days
        return rows

    monkeypatch.setattr(admin, "list_old_notes", fake_list_old_notes)
    response = client.get("/admin/notes")
    assert response.status_code == 200
    assert captured["older_than_days"] == 90
    assert response.json() == rows


def test_admin_notes_custom_older_than_days_reaches_admin(monkeypatch):
    captured = {}

    def fake_list_old_notes(older_than_days):
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

    def fake_notes_by_ids(ids):
        calls["notes_by_ids"] = ids
        return rows

    def fake_delete_notes(ids):
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
    monkeypatch.setattr(admin, "notes_by_ids", lambda ids: [{"id": i} for i in ids])
    monkeypatch.setattr(admin, "delete_notes", lambda ids: calls.__setitem__("delete_notes", ids))
    response = client.post("/admin/notes/delete", json={"ids": ["note:a"], "confirm": False})
    assert response.status_code == 200
    assert calls["delete_notes"] is None


def test_admin_notes_delete_confirm_true_calls_delete_notes(monkeypatch):
    calls = {}

    def fake_notes_by_ids(ids):
        return [{"id": i} for i in ids]

    def fake_delete_notes(ids):
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

    def fake_find_duplicates(threshold, kind, limit):
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

    def fake_find_duplicates(threshold, kind, limit):
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


# ---- POST /admin/archive -----------------------------------------------------


def test_admin_archive_dry_run_by_default(monkeypatch):
    calls = {"archive_candidates": 0, "archive_rows": None}
    candidates = [{"id": "note:old", "kind": "agent_note", "hit_count": 0, "last_hit_at": None}]

    def fake_archive_candidates(now):
        calls["archive_candidates"] += 1
        assert isinstance(now, float)
        return candidates

    def fake_archive_rows(ids, now):
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

    def fake_archive_candidates(now):
        return candidates

    def fake_archive_rows(ids, now):
        calls["ids"] = ids
        return len(ids)

    monkeypatch.setattr(admin, "archive_candidates", fake_archive_candidates)
    monkeypatch.setattr(admin, "archive_rows", fake_archive_rows)
    response = client.post("/admin/archive", json={"confirm": True})
    assert response.status_code == 200
    assert response.json() == {"archived": 2}
    assert calls["ids"] == ["note:old1", "note:old2"]


# ---- POST /admin/restore -----------------------------------------------------


def test_admin_restore_dry_run_by_default(monkeypatch):
    calls = {"restore_rows": None}
    rows = [{"id": "note:old", "kind": "agent_note"}]

    def fake_rows_by_ids(ids):
        return rows

    def fake_restore_rows(ids):
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

    def fake_rows_by_ids(ids):
        return [{"id": i} for i in ids]

    def fake_restore_rows(ids):
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


def test_search_include_archived_defaults_to_false(monkeypatch):
    captured = {}

    async def fake_search(query, source="all", include_archived=False):
        captured["include_archived"] = include_archived
        return []

    monkeypatch.setattr(api, "search", fake_search)
    response = client.post("/search", json={"query": "hello"})
    assert response.status_code == 200
    assert captured["include_archived"] is False
