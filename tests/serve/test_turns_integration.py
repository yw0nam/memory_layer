"""Integration tests for the verbatim turn lane against the throwaway Postgres.

Each test names its sessions and namespaces with a random tag and deletes its rows afterwards.
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
from memory_base.serve.access import auth, namespaces
from memory_base.serve.notes import store as note_store
from memory_base.serve.turns import store

pytestmark = pytest.mark.integration

KEY = auth.KeyIdentity(key_id="test-hash", label="test")


@pytest.fixture
def tag(monkeypatch):
    tag = uuid.uuid4().hex[:10]

    async def fake_authenticate_request(plaintext_key):
        return KEY if plaintext_key == "test-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)
    yield tag

    async def cleanup():
        conn = await asyncpg.connect(db_url())
        try:
            await conn.execute(
                f'DELETE FROM "{PG_SCHEMA}".turns WHERE session_id LIKE $1', f"{tag}%"
            )
            await conn.execute(
                f"DELETE FROM \"{PG_SCHEMA}\".memory_chunks WHERE metadata->>'author' = $1", tag
            )
            await conn.execute(f'DELETE FROM "{PG_SCHEMA}".namespaces WHERE name = $1', f"ns-{tag}")
        finally:
            await conn.close()

    asyncio.run(cleanup())


def _api(key="test-key"):
    return TestClient(api.app, headers={"X-API-Key": key, "X-Author": "claude-code"})


def _post(client, session, content, **overrides):
    body = {
        "id": str(uuid.uuid4()),
        "origin": "claude_code",
        "session_id": session,
        "role": "user",
        "content": content,
    }
    return client.post("/turns", json={**body, **overrides})


def test_schema_has_the_columns_the_extension_and_no_grant_to_the_query_role(tag):
    async def scenario():
        conn = await asyncpg.connect(db_url())
        try:
            await ensure_schema(conn)
            columns = {
                row["column_name"]
                for row in await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = $1 AND table_name = 'turns'",
                    PG_SCHEMA,
                )
            }
            trgm = await conn.fetchval("SELECT count(*) FROM pg_extension WHERE extname='pg_trgm'")
            granted = await conn.fetchval(
                "SELECT has_table_privilege('memory_tables_query', $1, 'SELECT')",
                f"{PG_SCHEMA}.turns",
            )
            return columns, trgm, granted
        finally:
            await conn.close()

    columns, trgm, granted = asyncio.run(scenario())
    assert columns == {
        "id", "namespace", "origin", "session_id", "seq", "role", "content", "reasoning",
        "repo", "created_at",
    }  # fmt: skip
    assert trgm == 1
    assert granted is False


def test_appends_number_the_session_and_a_replay_stores_nothing(tag):
    client = _api()
    session = f"{tag}-a"
    first = _post(client, session, "first").json()
    turn_id = str(uuid.uuid4())
    second = _post(client, session, "second", id=turn_id, role="assistant")
    replay = _post(client, session, "second", id=turn_id, role="assistant")
    other = _post(client, session, "changed", id=turn_id, role="assistant")
    assert (first["seq"], first["stored"]) == (1, True)
    assert second.json() == {"id": turn_id, "seq": 2, "stored": True}
    assert replay.json() == {"id": turn_id, "seq": 2, "stored": False}
    assert other.status_code == 409
    assert _post(client, session, "third").json()["seq"] == 3


def test_a_session_keeps_the_namespace_of_its_first_turn(tag):
    client = _api()
    session = f"{tag}-ns"
    asyncio.run(namespaces.create_namespace(f"ns-{tag}"))
    assert _post(client, session, "one").status_code == 200
    moved = _post(client, session, "two", namespace=f"ns-{tag}")
    assert moved.status_code == 400
    assert "belongs to namespace" in moved.json()["error"]


def test_concurrent_appends_to_one_session_get_distinct_seqs(tag):
    async def scenario():
        rows = [
            {
                "id": str(uuid.uuid4()),
                "namespace": "default",
                "origin": "hermes",
                "session_id": f"{tag}-race",
                "role": "user",
                "content": f"turn {n}",
                "reasoning": None,
                "repo": None,
            }
            for n in range(8)
        ]
        return await asyncio.gather(*(store.append(row) for row in rows))

    results = asyncio.run(scenario())
    assert sorted(result["seq"] for result in results) == list(range(1, 9))


def test_a_korean_word_with_a_particle_and_a_literal_percent_are_found(tag):
    client = _api()
    session = f"{tag}-search"
    _post(client, session, "나는 회사를 그만두기로 했다")
    _post(client, session, f"discount is 100% off {tag}")
    _post(client, session, f"discount is 1000 off {tag}")
    korean = client.get("/turns/search", params={"q": "회사"}).json()
    assert [hit["session_id"] for hit in korean] == [session]
    percent = client.get("/turns/search", params={"q": f"100% off {tag}"}).json()
    assert [hit["seq"] for hit in percent] == [2]
    assert "reasoning" not in percent[0]
    assert "100%" in percent[0]["window"]


def test_read_session_with_contains_returns_matches_and_their_neighbours(tag):
    client = _api()
    session = f"{tag}-read"
    for n in range(1, 10):
        word = "needle" if n in (3, 4, 8) else "hay"
        _post(client, session, f"{word} number {n}", role="user" if n % 2 else "assistant")
    body = client.get(f"/sessions/claude_code/{session}", params={"contains": "NEEDLE"}).json()
    assert [(t["seq"], t["match"]) for t in body["turns"]] == [
        (2, False), (3, True), (4, True), (5, False), (7, False), (8, True), (9, False),
    ]  # fmt: skip
    assert body["next_offset"] is None
    paged = client.get(
        f"/sessions/claude_code/{session}", params={"contains": "needle", "offset": 4, "limit": 2}
    ).json()
    assert [t["seq"] for t in paged["turns"]] == [4, 5]
    assert paged["next_offset"] == 7
    assert (
        client.get(f"/sessions/claude_code/{session}", params={"contains": "n"}).status_code == 400
    )


def test_read_session_is_cut_at_the_character_cap(tag):
    client = _api()
    session = f"{tag}-big"
    for _ in range(4):
        _post(client, session, "x" * 15_000)
    body = client.get(f"/sessions/claude_code/{session}").json()
    assert [t["seq"] for t in body["turns"]] == [1, 2]
    assert body["next_offset"] == 3


def test_sessions_are_listed_newest_first_with_the_first_user_message_as_title(tag):
    client = _api()
    _post(client, f"{tag}-old", "  First   question\nabout  plans ", repo="repo:h/p")
    _post(client, f"{tag}-new", "y" * 300, origin="hermes")
    _post(client, f"{tag}-new", "an answer", origin="hermes", role="assistant")
    rows = [
        row
        for row in client.get("/sessions", params={"limit": 200}).json()
        if row["session_id"].startswith(tag)
    ]
    assert [row["session_id"] for row in rows] == [f"{tag}-new", f"{tag}-old"]
    assert rows[0]["title"] == "y" * store.TITLE_CHARS
    assert rows[0]["turns"] == 2
    assert rows[1]["title"] == "First question about plans"
    assert rows[1]["repo"] == "repo:h/p"
    only_hermes = client.get("/sessions", params={"origin": "hermes", "limit": 200}).json()
    assert f"{tag}-old" not in [row["session_id"] for row in only_hermes]
    by_repo = client.get("/sessions", params={"repo": "repo:h/p", "limit": 200}).json()
    assert [row["session_id"] for row in by_repo if row["session_id"].startswith(tag)] == [
        f"{tag}-old"
    ]


def test_a_read_without_a_namespace_covers_every_namespace_and_a_filter_narrows_it(tag):
    asyncio.run(namespaces.create_namespace(f"ns-{tag}"))
    client = _api()
    assert _post(client, f"{tag}-a", f"plan {tag}").status_code == 200
    assert _post(client, f"{tag}-b", f"plan {tag}", namespace=f"ns-{tag}").status_code == 200
    ids = lambda response: {row["session_id"] for row in response.json()}  # noqa: E731
    assert ids(client.get("/sessions", params={"limit": 200})) >= {f"{tag}-a", f"{tag}-b"}
    narrowed = client.get("/sessions", params={"namespace": f"ns-{tag}", "limit": 200})
    assert ids(narrowed) == {f"{tag}-b"}
    assert {row["session_id"] for row in client.get("/turns/search", params={"q": tag}).json()} == {
        f"{tag}-a",
        f"{tag}-b",
    }


def test_a_namespace_holding_turns_cannot_be_deleted_and_a_session_delete_empties_it(tag):
    asyncio.run(namespaces.create_namespace(f"ns-{tag}"))
    session = f"{tag}-del"
    _post(_api(), session, "kept", namespace=f"ns-{tag}")
    with pytest.raises(namespaces.NamespaceNotEmptyError):
        asyncio.run(namespaces.delete_namespace(f"ns-{tag}"))
    assert _api().delete(f"/admin/sessions/claude_code/{session}").json() == {"deleted": 1}
    asyncio.run(namespaces.delete_namespace(f"ns-{tag}"))


def test_a_note_names_its_source_session(tag):
    async def scenario():
        saved = await note_store.save_note(
            f"Source session round trip {tag}",
            kind="work",
            tags=["pytest"],
            author=tag,
            source_session={"origin": "hermes", "session_id": f"{tag}-src"},
        )
        listed = await note_store.list_notes(tags=["pytest"], authors=[tag])
        await db.close_pool()
        return saved, listed

    saved, listed = asyncio.run(scenario())
    assert [note["source_session"] for note in listed if note["id"] == saved["id"]] == [
        {"origin": "hermes", "session_id": f"{tag}-src"}
    ]


def test_sessions_are_listed_by_window_and_a_session_straddling_since_is_included(tag):
    client = _api()
    days_ago = {
        f"{tag}-old": [10],
        f"{tag}-straddle": [8, 4],
        f"{tag}-middle": [3],
        f"{tag}-new": [1],
    }
    for session, days in days_ago.items():
        for _ in days:
            _post(client, session, "turn")

    async def backdate():
        conn = await asyncpg.connect(db_url())
        try:
            for session, days in days_ago.items():
                ids = await conn.fetch(
                    f'SELECT id FROM "{PG_SCHEMA}".turns WHERE session_id = $1 ORDER BY seq',
                    session,
                )
                for row, day in zip(ids, days):
                    await conn.execute(
                        f'UPDATE "{PG_SCHEMA}".turns SET created_at = now() - make_interval(days => $2) '
                        "WHERE id = $1",
                        row["id"],
                        day,
                    )
        finally:
            await conn.close()

    asyncio.run(backdate())
    now = datetime.now(timezone.utc)
    window = {
        "since": (now - timedelta(days=6)).isoformat(),
        "until": (now - timedelta(days=2)).isoformat(),
        "limit": 200,
    }
    rows = client.get("/sessions", params=window).json()
    assert sorted(row["session_id"] for row in rows if row["session_id"].startswith(tag)) == [
        f"{tag}-middle",
        f"{tag}-straddle",
    ]
