"""Unit tests for the turn lane REST routes and the store's pure helpers: no database."""

from __future__ import annotations

import uuid

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.turns import store

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})

AWS_KEY = "AKIA" + "Q" * 16


def _turn(**overrides):
    body = {
        "id": str(uuid.uuid4()),
        "origin": "claude_code",
        "session_id": "s-1",
        "role": "user",
        "content": "hello",
    }
    return {**body, **overrides}


@pytest.fixture
def appended(monkeypatch):
    rows = []

    async def fake_append(row):
        rows.append(row)
        return {"id": row["id"], "seq": 1, "stored": True}

    monkeypatch.setattr(store, "append", fake_append)
    return rows


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "not-a-uuid"},
        {"origin": "cursor"},
        {"role": "system"},
        {"content": "x" * 64_001},
        {"reasoning": "x" * 64_001},
        {"session_id": " "},
        {"repo": "github.com/o/r"},
        {"unknown": 1},
    ],
)
def test_a_malformed_turn_is_refused_and_nothing_is_stored(appended, overrides):
    response = client.post("/turns", json=_turn(**overrides))
    assert response.status_code == 400
    assert appended == []


def test_a_valid_turn_defaults_to_the_default_namespace(appended):
    response = client.post("/turns", json=_turn(repo="repo:github.com/o/r", reasoning="because"))
    assert response.status_code == 200
    assert response.json()["seq"] == 1
    assert appended[0]["namespace"] == "default"
    assert appended[0]["reasoning"] == "because"


@pytest.mark.parametrize("field", ["content", "reasoning"])
def test_a_credential_is_refused_before_the_database(field):
    response = client.post("/turns", json=_turn(**{field: f"key {AWS_KEY}"}))
    assert response.status_code == 400
    assert "credential" in response.json()["error"]


def test_an_id_holding_another_turn_is_409(monkeypatch):
    async def fake_append(row):
        raise store.Conflict("id already holds a different turn")

    monkeypatch.setattr(store, "append", fake_append)
    assert client.post("/turns", json=_turn()).status_code == 409


def test_sessions_pass_their_filters_and_cover_every_namespace_unless_narrowed(monkeypatch):
    captured = {}

    async def fake_list_sessions(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(store, "list_sessions", fake_list_sessions)
    response = client.get(
        "/sessions", params={"origin": "hermes", "repo": "repo:h/p", "since": "2026-01-01"}
    )
    assert response.status_code == 200
    assert captured["namespaces"] is None
    assert captured["origin"] == "hermes"
    assert captured["repo"] == "repo:h/p"
    assert captured["since"] is not None
    assert client.get("/sessions", params={"namespace": "personal"}).status_code == 200
    assert captured["namespaces"] == ["personal"]


def test_an_unknown_session_is_404(monkeypatch):
    async def fake_read_session(*args):
        raise store.NotFound("no such session")

    monkeypatch.setattr(store, "read_session", fake_read_session)
    assert client.get("/sessions/hermes/nope").status_code == 404


def test_a_search_query_under_two_characters_is_refused():
    assert client.get("/turns/search", params={"q": "a"}).status_code == 400


def test_a_session_delete_returns_the_count(monkeypatch):
    async def fake_delete_session(origin, session_id):
        return 3

    monkeypatch.setattr(store, "delete_session", fake_delete_session)
    assert client.delete("/admin/sessions/hermes/s-1").json() == {"deleted": 3}


def test_the_response_is_cut_at_the_character_cap_and_names_where_to_resume():
    rows = [{"seq": n, "content": "x" * 15_000, "reasoning": None} for n in range(1, 6)]
    kept, next_offset = store._cut(rows, 50)
    assert [row["seq"] for row in kept] == [1, 2]
    assert next_offset == 3


def test_the_response_is_cut_at_the_row_cap():
    rows = [{"seq": n, "content": "x", "reasoning": None} for n in range(1, 4)]
    kept, next_offset = store._cut(rows, 2)
    assert [row["seq"] for row in kept] == [1, 2]
    assert next_offset == 3


def test_one_oversized_row_is_still_returned():
    kept, next_offset = store._cut(
        [{"seq": 1, "content": "x" * 64_000, "reasoning": "y" * 64_000}], 50
    )
    assert len(kept) == 1
    assert next_offset is None


def test_a_substring_pattern_escapes_like_wildcards():
    assert store._like(r"50%_\x") == r"%50\%\_\\x%"


def test_a_window_centres_on_the_first_case_insensitive_match():
    content = "a" * 500 + "NEEDLE" + "b" * 500
    window = store._window(content, "needle")
    assert len(window) == store.WINDOW_CHARS
    assert "NEEDLE" in window


def test_an_id_stored_under_another_session_is_409(monkeypatch):
    import contextlib

    import asyncpg

    class Conn:
        async def execute(self, *args):
            return None

        async def fetchrow(self, sql, *args):
            if "INSERT" in sql:
                raise asyncpg.UniqueViolationError("duplicate key")
            return None

        async def fetchval(self, sql, *args):
            return None if "SELECT namespace" in sql else 1

        def transaction(self):
            return contextlib.AsyncExitStack()

    @contextlib.asynccontextmanager
    async def acquire():
        yield Conn()

    async def ensure(conn):
        return None

    monkeypatch.setattr(store.db, "acquire", acquire)
    monkeypatch.setattr(store, "ensure_schema_once", ensure)
    assert client.post("/turns", json=_turn()).status_code == 409
