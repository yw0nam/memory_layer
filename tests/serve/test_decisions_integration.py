"""Integration tests for the decision lane against a real Postgres.

Covers what a faked connection cannot: per-scope-and-author numbering, distinct
numbers under real concurrency, the supersede and retire transitions, that a
refused supersede consumes no number, and author isolation, through the real
REST app against the session's throwaway Postgres container.
"""

from __future__ import annotations

import asyncio
import uuid

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api
from memory_base.serve.decisions import store

pytestmark = pytest.mark.integration

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})
OTHER = {"X-Author": "natsume"}


@pytest.fixture
def scope():
    value = f"repo:github.com/zz{uuid.uuid4().hex[:8]}/r"
    yield value

    async def _cleanup():
        conn = await asyncpg.connect(db_url())
        try:
            await conn.execute(f'DELETE FROM "{PG_SCHEMA}".decisions WHERE scope = $1', value)
        finally:
            await conn.close()

    asyncio.run(_cleanup())


def _record(scope, title="t", headers=None, **fields):
    body = {"scope": scope, "title": title, "decision": "d", **fields}
    return client.post("/decisions", json=body, headers=headers)


def _numbers(scope, status="active", headers=None):
    rows = client.get("/decisions", params={"scope": scope, "status": status}, headers=headers)
    return [row["number"] for row in rows.json()]


def test_numbers_count_per_scope_and_author(scope):
    other_scope = scope + "x"
    try:
        assert _record(scope).json()["number"] == 1
        assert _record(scope).json()["number"] == 2
        assert _record(scope, headers=OTHER).json()["number"] == 1
        assert _record(other_scope).json()["number"] == 1
    finally:
        asyncio.run(_drop(other_scope))


async def _drop(scope):
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(f'DELETE FROM "{PG_SCHEMA}".decisions WHERE scope = $1', scope)
    finally:
        await conn.close()


def test_concurrent_records_get_distinct_numbers(scope):
    async def _run():
        await db.get_pool()
        records = [
            store.record_decision(author="claude-code", scope=scope, title=f"t{i}", decision="d")
            for i in range(6)
        ]
        try:
            return await asyncio.gather(*records)
        finally:
            await db.close_pool()

    numbers = sorted(result["number"] for result in asyncio.run(_run()))
    assert numbers == [1, 2, 3, 4, 5, 6]


def test_supersede_replaces_the_active_decision_and_links_both_ways(scope):
    _record(scope, "old")
    new = _record(scope, "new", supersedes=1)
    assert new.status_code == 201
    assert new.json() == {"number": 2, "superseded": 1}
    assert _numbers(scope) == [2]
    assert _numbers(scope, "all") == [2, 1]
    old_row = client.get("/decisions/1", params={"scope": scope}).json()
    new_row = client.get("/decisions/2", params={"scope": scope}).json()
    assert old_row["status"] == "superseded" and old_row["superseded_by"] == 2
    assert new_row["supersedes"] == 1 and new_row["superseded_by"] is None


def test_a_refused_supersede_consumes_no_number(scope):
    _record(scope)
    _record(scope, supersedes=1)
    assert _record(scope, supersedes=1).status_code == 400
    assert _record(scope, supersedes=9).status_code == 400
    assert _record(scope, supersedes=1, headers=OTHER).status_code == 400
    assert _record(scope).json()["number"] == 3


def test_retire_marks_an_active_decision_once(scope):
    _record(scope)
    retired = client.post("/decisions/retire", json={"scope": scope, "number": 1, "reason": "gone"})
    assert retired.status_code == 200
    assert _numbers(scope) == []
    row = client.get("/decisions/1", params={"scope": scope}).json()
    assert row["status"] == "retired" and row["retired_reason"] == "gone"
    again = client.post("/decisions/retire", json={"scope": scope, "number": 1, "reason": "gone"})
    assert again.status_code == 400
    assert _record(scope, supersedes=1).status_code == 400


def test_another_author_sees_nothing_and_cannot_retire(scope):
    _record(scope)
    assert _numbers(scope, "all", headers=OTHER) == []
    assert client.get("/decisions/1", params={"scope": scope}, headers=OTHER).status_code == 404
    other_retire = client.post(
        "/decisions/retire", json={"scope": scope, "number": 1, "reason": "x"}, headers=OTHER
    )
    assert other_retire.status_code == 400
    assert _numbers(scope) == [1]


def test_get_returns_the_fields_as_recorded(scope):
    _record(scope, "T", why="because", links=["https://x.y/1"], decided_on="2026-05-01")
    row = client.get("/decisions/1", params={"scope": scope}).json()
    assert row["title"] == "T" and row["decision"] == "d" and row["why"] == "because"
    assert row["links"] == ["https://x.y/1"]
    assert row["decided_on"] == "2026-05-01"
    assert row["status"] == "active" and row["supersedes"] is None
