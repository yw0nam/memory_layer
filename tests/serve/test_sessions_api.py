"""Unit tests for the session record REST endpoints (memory_base.serve.sessions.routes).

No DB: validation runs before storage, and the store functions are monkeypatched
where a route's delegation is under test.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.sessions import store

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})


def _event(**overrides):
    event = {
        "id": str(uuid.uuid4()),
        "session_id": "s1",
        "role": "user",
        "content": "hello",
        "tool_name": None,
        "tool_use_id": None,
        "event_at": "2026-05-01T10:00:00Z",
        "repo": "repo:github.com/o/r",
        "host": "mac",
        "cwd": "/work/r",
    }
    event.update(overrides)
    return event


def _unreachable(**kwargs):
    raise AssertionError("storage must not be reached")


@pytest.fixture
def no_store(monkeypatch):
    async def fake(*args, **kwargs):
        _unreachable()

    for name in ("store_events", "list_sessions", "read_session", "search", "delete_session"):
        monkeypatch.setattr(store, name, fake)


def test_write_delegates_the_header_author_and_parsed_events(monkeypatch):
    captured = {}

    async def fake_store(author, events):
        captured.update(author=author, events=events)
        return {"stored": 1, "known": 0, "conflicts": []}

    monkeypatch.setattr(store, "store_events", fake_store)
    event = _event(role="tool_call", tool_name="Bash", tool_use_id="tu1", content="")
    response = client.post(
        "/session_events", json={"events": [event]}, headers={"X-Author": "natsume"}
    )
    assert response.status_code == 200
    assert response.json() == {"stored": 1, "known": 0, "conflicts": []}
    assert captured["author"] == "natsume"
    [row] = captured["events"]
    assert row["id"] == event["id"] and row["role"] == "tool_call" and row["content"] == ""
    assert row["event_at"].isoformat() == "2026-05-01T10:00:00+00:00"
    assert row["repo"] == "repo:github.com/o/r"


def test_a_nul_character_is_dropped_before_storage(monkeypatch):
    captured = {}

    async def fake_store(author, events):
        captured["events"] = events
        return {"stored": 1, "known": 0, "conflicts": []}

    monkeypatch.setattr(store, "store_events", fake_store)
    client.post("/session_events", json={"events": [_event(content="a\u0000b")]})
    assert captured["events"][0]["content"] == "ab"


@pytest.mark.parametrize(
    "body",
    [
        {"events": [_event()], "author": "x"},
        {"events": [{**_event(), "author": "x"}]},
        {"events": []},
        {"events": [_event()] * (store.MAX_EVENTS + 1)},
        {"events": "nope"},
        {},
        [],
        {"events": [_event(), "x"]},
        {"events": [{k: v for k, v in _event().items() if k != "host"}]},
        {"events": [_event(id="not-a-uuid")]},
        {"events": [_event(role="system")]},
        {"events": [_event(session_id="")]},
        {"events": [_event(session_id="x" * 201)]},
        {"events": [_event(content="x" * (store.MAX_CONTENT_CHARS + 1))]},
        {"events": [_event(content=None)]},
        {"events": [_event(event_at="yesterday")]},
        {"events": [_event(repo="github.com/o/r")]},
        {"events": [_event(host="")]},
        {"events": [_event(cwd="x" * 4097)]},
        {"events": [_event(extra=1)]},
    ],
    ids=lambda body: json.dumps(body)[:60],
)
def test_a_bad_write_body_is_a_400_that_stores_nothing(no_store, body):
    response = client.post("/session_events", json=body)
    assert response.status_code == 400
    assert response.json()["error"]


def test_an_author_in_the_body_names_the_header(no_store):
    response = client.post("/session_events", json={"events": [_event()], "author": "x"})
    assert "X-Author" in response.json()["error"]


def test_a_body_over_the_byte_limit_is_a_400(no_store):
    blob = json.dumps({"events": [_event(content="x" * 60_000)] * 40})
    assert len(blob) > store.MAX_BODY_BYTES
    response = client.post(
        "/session_events", content=blob, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert "bytes" in response.json()["error"]


def test_a_non_json_body_is_a_400(no_store):
    assert client.post("/session_events", content=b"{").status_code == 400


def test_list_delegates_filters_and_the_header_author(monkeypatch):
    captured = {}

    async def fake_list(**kwargs):
        captured.update(kwargs)
        return [{"session_id": "s1"}]

    monkeypatch.setattr(store, "list_sessions", fake_list)
    response = client.get(
        "/sessions",
        params={"repo": "repo:github.com/o/r", "since": "2026-05-01", "limit": "5"},
        headers={"X-Author": "natsume"},
    )
    assert response.status_code == 200 and response.json() == [{"session_id": "s1"}]
    assert captured["author"] == "natsume"
    assert captured["repo"] == "repo:github.com/o/r"
    assert captured["since"] == datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
    assert captured["until"] is None and captured["limit"] == 5


@pytest.mark.parametrize(
    "params",
    [{"limit": "0"}, {"limit": "101"}, {"limit": "x"}, {"since": "nope"}, {"repo": "x"}],
)
def test_list_rejects_bad_filters(no_store, params):
    assert client.get("/sessions", params=params).status_code == 400


def test_read_delegates_and_an_unknown_session_is_a_404(monkeypatch):
    captured = {}

    async def fake_read(**kwargs):
        captured.update(kwargs)
        raise store.NotFound("no session 'a/b'")

    monkeypatch.setattr(store, "read_session", fake_read)
    response = client.get(
        "/sessions/a%2Fb", params={"from_seq": "3", "limit": "7", "role": "user", "contains": "ab"}
    )
    assert response.status_code == 404
    assert captured == {
        "author": "claude-code",
        "session_id": "a/b",
        "from_seq": 3,
        "limit": 7,
        "role": "user",
        "contains": "ab",
    }


@pytest.mark.parametrize(
    "params",
    [{"limit": "201"}, {"from_seq": "0"}, {"role": "system"}, {"contains": "a"}],
)
def test_read_rejects_bad_parameters(no_store, params):
    assert client.get("/sessions/s1", params=params).status_code == 400


def test_search_delegates_and_requires_two_characters(monkeypatch, no_store):
    assert client.get("/session_events/search", params={"q": "a"}).status_code == 400
    assert (
        client.get("/session_events/search", params={"q": "ab", "limit": "51"}).status_code == 400
    )
    assert client.get("/session_events/search", params={"q": "ab", "role": "x"}).status_code == 400
    captured = {}

    async def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(store, "search", fake_search)
    response = client.get(
        "/session_events/search", params={"q": "ab", "role": "assistant", "limit": "9"}
    )
    assert response.status_code == 200 and response.json() == []
    assert captured == {
        "author": "claude-code",
        "q": "ab",
        "repo": None,
        "role": "assistant",
        "limit": 9,
    }


def test_admin_delete_removes_the_callers_session(monkeypatch):
    captured = {}

    async def fake_delete(**kwargs):
        captured.update(kwargs)
        return 3

    monkeypatch.setattr(store, "delete_session", fake_delete)
    response = client.delete("/admin/sessions/s%201", headers={"X-Author": "natsume"})
    assert response.status_code == 200 and response.json() == {"deleted": 3}
    assert captured == {"author": "natsume", "session_id": "s 1"}
