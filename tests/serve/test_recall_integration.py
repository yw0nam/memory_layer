"""Integration test for /recall's decision branch against a real Postgres.

The reranker and the note search are faked (they are upstream services); the
decision lane reads real rows: only active decisions of the caller come back.
"""

from __future__ import annotations

import asyncio
import uuid

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api
from memory_base.serve.search import recall

pytestmark = pytest.mark.integration

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})
OTHER = {"X-Author": "natsume"}


@pytest.fixture
def scope(monkeypatch):
    async def fake_search(query, **kwargs):
        return []

    async def fake_rerank(query, texts):
        return [1.0] * len(texts)

    monkeypatch.setattr(recall, "search", fake_search)
    monkeypatch.setattr(recall, "rerank_scores", fake_rerank)
    value = f"repo:github.com/zz{uuid.uuid4().hex[:8]}/r"
    yield value

    async def _cleanup():
        conn = await asyncpg.connect(db_url())
        try:
            await conn.execute(f'DELETE FROM "{PG_SCHEMA}".decisions WHERE scope = $1', value)
        finally:
            await conn.close()

    asyncio.run(_cleanup())


def _record(scope, title, headers=None, **fields):
    body = {"scope": scope, "title": title, "decision": "d", **fields}
    assert client.post("/decisions", json=body, headers=headers).status_code == 201


def test_superseded_retired_and_other_authors_decisions_are_never_recalled(scope):
    _record(scope, "old")
    _record(scope, "new", supersedes=1)
    _record(scope, "gone")
    client.post("/decisions/retire", json={"scope": scope, "number": 3, "reason": "r"})
    _record(scope, "kept")
    _record(scope, "theirs", headers=OTHER)
    body = client.post("/recall", json={"query": "q", "scope": scope}).json()
    assert [d["title"] for d in body["decisions"]] == ["kept", "new"]
    theirs = client.post("/recall", json={"query": "q", "scope": scope}, headers=OTHER).json()
    assert [d["title"] for d in theirs["decisions"]] == ["theirs"]
