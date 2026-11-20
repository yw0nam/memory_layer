"""Integration tests for the session record against the throwaway Postgres.

Covers what a faked connection cannot: batch insert with replay and conflict handling,
per-(author, session) seq numbering under concurrent batches, author isolation on every
read, the trigram substring search on Korean text, and the admin delete. Each test names
its sessions with a random tag and deletes its rows afterwards.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.core.schema import ensure_schema
from memory_base.serve import api
from memory_base.serve.sessions import store

pytestmark = pytest.mark.integration

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})
OTHER = {"X-Author": "natsume"}
START = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def tag():
    value = f"t{uuid.uuid4().hex[:10]}"
    yield value

    async def _cleanup():
        conn = await asyncpg.connect(db_url())
        try:
            await conn.execute(
                f'DELETE FROM "{PG_SCHEMA}".session_events WHERE session_id LIKE $1', f"{value}%"
            )
        finally:
            await conn.close()

    asyncio.run(_cleanup())


def _event(session, content, n=0, role="user", **overrides):
    event = {
        "id": str(uuid.uuid4()),
        "session_id": session,
        "role": role,
        "content": content,
        "tool_name": None,
        "tool_use_id": None,
        "event_at": (START + timedelta(minutes=n)).isoformat(),
        "repo": "repo:github.com/o/r",
        "host": "mac",
        "cwd": "/work/r",
    }
    event.update(overrides)
    return event


def _write(events, headers=None):
    response = client.post("/session_events", json={"events": events}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _read(session, headers=None, **params):
    return client.get(f"/sessions/{session}", params=params, headers=headers)


def _seqs(session, **params):
    return [(e["seq"], e["content"]) for e in _read(session, **params).json()["events"]]


def test_schema_holds_the_table_the_trigram_index_and_the_extension(tag):
    async def scenario():
        conn = await asyncpg.connect(db_url())
        try:
            await ensure_schema(conn)
            columns = {
                row["column_name"]
                for row in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = $1 AND table_name = 'session_events'",
                    PG_SCHEMA,
                )
            }
            trgm = await conn.fetchval("SELECT count(*) FROM pg_extension WHERE extname='pg_trgm'")
            gin = await conn.fetchval(
                "SELECT count(*) FROM pg_indexes WHERE schemaname = $1 "
                "AND tablename = 'session_events' AND indexdef ILIKE '%gin_trgm_ops%'",
                PG_SCHEMA,
            )
            return columns, trgm, gin
        finally:
            await conn.close()

    columns, trgm, gin = asyncio.run(scenario())
    assert columns == {
        "id", "author", "namespace", "session_id", "seq", "role", "content", "tool_name",
        "tool_use_id", "event_at", "repo", "host", "cwd", "created_at",
    }  # fmt: skip
    assert trgm == 1 and gin == 1


def test_a_batch_is_stored_in_order_a_replay_is_known_and_a_changed_payload_conflicts(tag):
    session = f"{tag}-a"
    events = [_event(session, f"m{n}", n) for n in range(3)]
    assert _write(events) == {"stored": 3, "known": 0, "conflicts": []}
    assert _seqs(session) == [(1, "m0"), (2, "m1"), (3, "m2")]
    assert _write(events) == {"stored": 0, "known": 3, "conflicts": []}
    changed = {**events[1], "content": "changed"}
    fresh = _event(session, "m3", 3)
    result = _write([events[0], changed, fresh])
    assert result == {"stored": 1, "known": 1, "conflicts": [events[1]["id"]]}
    assert _seqs(session) == [(1, "m0"), (2, "m1"), (3, "m2"), (4, "m3")]


def test_an_event_keeps_its_tool_fields_and_an_empty_tool_call_content(tag):
    session = f"{tag}-tool"
    call = _event(session, "", 0, role="tool_call", tool_name="Bash", tool_use_id="tu1")
    _write([call])
    [stored] = _read(session).json()["events"]
    assert stored["role"] == "tool_call" and stored["tool_name"] == "Bash"
    assert stored["content"] == "" and stored["event_at"].startswith("2026-05-01T10:00:00")


def test_concurrent_batches_to_one_session_get_distinct_contiguous_seqs(tag):
    session = f"{tag}-race"
    batches = [[_event(session, f"{name}{n}", n) for n in range(5)] for name in ("x", "y")]
    events = [
        [
            {**e, "event_at": datetime.fromisoformat(e["event_at"]), "id": uuid.UUID(e["id"])}
            for e in batch
        ]
        for batch in batches
    ]

    async def _run():
        await db.get_pool()
        try:
            return await asyncio.gather(*(store.store_events("claude-code", b) for b in events))
        finally:
            await db.close_pool()

    assert [r["stored"] for r in asyncio.run(_run())] == [5, 5]
    rows = _seqs(session, limit=200)
    assert [seq for seq, _ in rows] == list(range(1, 11))
    first = rows[0][1][0]
    assert [content[0] for _, content in rows] == [first] * 5 + [rows[5][1][0]] * 5
    assert first != rows[5][1][0]


def test_every_read_returns_only_the_callers_rows(tag):
    session = f"{tag}-iso"
    _write([_event(session, "스커트를 다시 입는다", 0)])
    assert _read(session, headers=OTHER).status_code == 404
    listed = client.get("/sessions", headers=OTHER).json()
    assert all(not row["session_id"].startswith(tag) for row in listed)
    found = client.get("/session_events/search", params={"q": "스커트"}, headers=OTHER).json()
    assert [h for h in found if h["session_id"].startswith(tag)] == []
    assert client.delete(f"/admin/sessions/{session}", headers=OTHER).json() == {"deleted": 0}
    assert _write([_event(session, "mine", 1)], headers=OTHER)["stored"] == 1
    assert _seqs(session, headers=OTHER) == [(1, "mine")]
    assert [c for _, c in _seqs(session)] == ["스커트를 다시 입는다"]
    assert client.delete(f"/admin/sessions/{session}", headers=OTHER).json() == {"deleted": 1}
    assert len(_seqs(session)) == 1


def test_list_sessions_titles_by_the_first_user_event_and_orders_newest_first(tag):
    older, newer = f"{tag}-old", f"{tag}-new"
    long_title = "w " * 100
    _write(
        [
            _event(older, "", 0, role="tool_call", tool_name="Bash"),
            _event(older, f"  first\n\n{long_title}", 1),
            _event(older, "answer", 2, role="assistant"),
            _event(newer, "later one", 60, repo=None),
        ]
    )
    rows = [
        r for r in client.get("/sessions", params={"limit": 100}).json() if tag in r["session_id"]
    ]
    assert [r["session_id"] for r in rows] == [newer, older]
    old = rows[1]
    assert old["title"] == " ".join(f"first {long_title}".split())[:120]
    assert (old["events"], old["repo"], old["host"], old["cwd"]) == (
        3,
        "repo:github.com/o/r",
        "mac",
        "/work/r",
    )
    assert old["started_at"].startswith("2026-05-01T10:00") and old["last_at"].startswith(
        "2026-05-01T10:02"
    )
    assert rows[0]["title"] == "later one" and rows[0]["repo"] is None
    by_repo = client.get("/sessions", params={"repo": "repo:github.com/o/r", "limit": 100}).json()
    assert [r["session_id"] for r in by_repo if tag in r["session_id"]] == [older]
    after = client.get("/sessions", params={"since": "2026-05-01T10:30:00Z", "limit": 100}).json()
    assert [r["session_id"] for r in after if tag in r["session_id"]] == [newer]


def test_read_session_pages_by_seq_and_filters_by_role(tag):
    session = f"{tag}-read"
    _write([_event(session, f"m{n}", n, role="assistant" if n % 2 else "user") for n in range(6)])
    assert [seq for seq, _ in _seqs(session, from_seq=3, limit=2)] == [3, 4]
    assert [seq for seq, _ in _seqs(session, role="assistant")] == [2, 4, 6]


def test_contains_returns_the_matches_with_one_neighbour_each_side(tag):
    session = f"{tag}-contains"
    _write([_event(session, "needle" if n == 4 else f"hay{n}", n) for n in range(9)])
    events = _read(session, contains="NEEDLE").json()["events"]
    assert [(e["seq"], e["match"]) for e in events] == [(4, False), (5, True), (6, False)]
    assert _read(session, contains="absent").json()["events"] == []
    only_user = _read(session, contains="needle", role="assistant").json()["events"]
    assert only_user == []


def test_search_finds_a_korean_word_with_a_particle_newest_first(tag):
    first, second = f"{tag}-k1", f"{tag}-k2"
    _write(
        [
            _event(first, "스커트를 다시 입어 보자", 0),
            _event(second, "그 스커트는 짧다", 5, role="assistant"),
            _event(second, "100% literal_underscore", 6),
        ]
    )
    hits = [
        h
        for h in client.get("/session_events/search", params={"q": "스커트", "limit": 50}).json()
        if h["session_id"].startswith(tag)
    ]
    assert [(h["session_id"], h["seq"], h["role"]) for h in hits] == [
        (second, 1, "assistant"),
        (first, 1, "user"),
    ]
    assert "스커트는" in hits[0]["window"] and hits[0]["repo"] == "repo:github.com/o/r"
    assert hits[0]["event_at"].startswith("2026-05-01T10:05")
    only = client.get("/session_events/search", params={"q": "스커트", "role": "user", "limit": 50})
    assert [h["session_id"] for h in only.json() if h["session_id"].startswith(tag)] == [first]
    literal = client.get("/session_events/search", params={"q": "0% lit"}).json()
    assert [h["session_id"] for h in literal if h["session_id"].startswith(tag)] == [second]
    wild = client.get("/session_events/search", params={"q": "_%"}).json()
    assert [h for h in wild if h["session_id"].startswith(tag)] == []


def test_the_search_window_is_cut_around_the_match(tag):
    session = f"{tag}-window"
    _write([_event(session, "a" * 1000 + "NEEDLE" + "b" * 1000, 0)])
    [hit] = [
        h
        for h in client.get("/session_events/search", params={"q": "needle"}).json()
        if h["session_id"] == session
    ]
    assert "NEEDLE" in hit["window"] and len(hit["window"]) == 300


def test_admin_delete_removes_the_callers_session_only(tag):
    gone, kept = f"{tag}-gone", f"{tag}-kept"
    _write([_event(gone, "a", 0), _event(gone, "b", 1), _event(kept, "c", 2)])
    assert client.delete(f"/admin/sessions/{gone}").json() == {"deleted": 2}
    assert _read(gone).status_code == 404
    assert _seqs(kept) == [(1, "c")]
    assert client.delete(f"/admin/sessions/{gone}").json() == {"deleted": 0}
