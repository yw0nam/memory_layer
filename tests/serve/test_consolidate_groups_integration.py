"""Integration tests for GET /admin/consolidate/groups against the throwaway Postgres.

Notes are seeded through ``save_note`` with the live embedder. The rule trio is worded so
every pair scores above the threshold passed here (unrelated notes score near 0.4) and
below the save-time similarity check, so no save is refused and no ``similar_ack`` is
written among them; the near copy scores far above that check.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api, auth, consolidate, namespaces
from memory_base.serve.notes import NOTE_SIMILAR_THRESHOLD, save_note

pytestmark = pytest.mark.integration

THRESHOLD = 0.60
RULES = (
    "The owner wants coding tasks delegated to subagents in git worktrees, "
    "and the main session only reviews their output.",
    "Delegate implementation work to a subagent working in its own worktree; "
    "the main agent reviews rather than writes code.",
    "When the user assigns a coding job, hand it to a worktree subagent "
    "and keep the primary session for review only.",
)
RESTATED = (
    "Coding work should go to subagents in separate worktrees while the main session reviews."
)
NEAR_COPY = (
    "The owner wants coding tasks delegated to subagents in git worktrees, "
    "and the main session only reviews the output."
)
UNRELATED = (
    "The staging Postgres runs on port 5433 with pgvector 0.8 installed.",
    "The user's cat is named Mochi and likes salmon treats.",
)

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


@pytest.fixture(autouse=True)
def consolidator(monkeypatch):
    identity = auth.KeyIdentity(
        key_id="consolidator-key-hash",
        label="consolidator",
        home="default",
        is_admin=True,
        allowed=frozenset(),
        authors=frozenset({"consolidator"}),
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "test-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)


async def _save(content, namespace, allow_similar=False):
    saved = await save_note(
        content,
        kind="work",
        tags=["rules"],
        namespace=namespace,
        author="claude-code",
        allow_similar=allow_similar,
    )
    return saved["id"]


async def _archive(note_id):
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(
            f'UPDATE "{PG_SCHEMA}".memory_chunks SET archived_at = $2 WHERE id = $1',
            note_id,
            time.time(),
        )
    finally:
        await conn.close()


async def _drop(*names):
    conn = await asyncpg.connect(db_url())
    try:
        await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE namespace = ANY($1::text[])',
            list(names),
        )
        await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".namespaces WHERE name = ANY($1::text[])', list(names)
        )
    finally:
        await conn.close()


async def _rows(namespace):
    conn = await asyncpg.connect(db_url())
    try:
        rows = await conn.fetch(
            f'SELECT * FROM "{PG_SCHEMA}".memory_chunks WHERE namespace = $1 ORDER BY id',
            namespace,
        )
        return [tuple(row.items()) for row in rows]
    finally:
        await conn.close()


async def _seed(main, other):
    for name in (main, other):
        await namespaces.create_namespace(name)
    archived = await _save(RESTATED, main)
    await _archive(archived)
    rules = [await _save(text, main) for text in RULES]
    unrelated = [await _save(text, main) for text in UNRELATED]
    elsewhere = await _save(RESTATED, other)
    return {"archived": archived, "rules": rules, "unrelated": unrelated, "elsewhere": elsewhere}


@pytest.fixture(scope="module")
def seeded():
    suffix = uuid.uuid4().hex[:8]
    main, other = f"consolidate-{suffix}", f"consolidate-other-{suffix}"
    try:
        ids = asyncio.run(_seed(main, other))
        yield {"main": main, "other": other, **ids}
    finally:
        asyncio.run(_drop(main, other))


def _fetch(*names, **params):
    response = client.get(
        "/admin/consolidate/groups",
        params=[("namespace", name) for name in names]
        + [(key, str(value)) for key, value in params.items()],
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_rule_restatements_form_one_group_and_nothing_else_leaks(seeded):
    body = _fetch(seeded["main"], seeded["other"], threshold=THRESHOLD)
    main = body["namespaces"][seeded["main"]]
    assert main["active_notes"] == 5
    [group] = main["groups"]
    assert {m["id"] for m in group["members"]} == set(seeded["rules"])
    assert THRESHOLD <= group["min_score"] <= group["max_score"] < NOTE_SIMILAR_THRESHOLD
    assert main["acknowledged"] == 0
    grouped = {m["id"] for g in main["groups"] for m in g["members"]}
    deferred = {i for d in main["deferred"] for i in d["ids"]}
    assert grouped.isdisjoint(seeded["unrelated"])
    assert deferred.isdisjoint(seeded["unrelated"])
    text = json.dumps(body)
    assert seeded["archived"] not in text
    assert seeded["elsewhere"] not in json.dumps(main)
    other = body["namespaces"][seeded["other"]]
    assert other["active_notes"] == 1
    assert other["groups"] == []
    assert not set(seeded["rules"]) & {m["id"] for g in other["groups"] for m in g["members"]}


def test_the_call_changes_no_row(seeded):
    before = asyncio.run(_rows(seeded["main"]))
    _fetch(seeded["main"], threshold=THRESHOLD)
    assert asyncio.run(_rows(seeded["main"])) == before


def test_pair_search_never_uses_the_embedding_index(seeded):
    async def plan():
        conn = await asyncpg.connect(db_url())
        try:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                for statement in consolidate.EXACT_SEARCH_SETTINGS:
                    await conn.execute(statement)
                rows = await conn.fetch(
                    "EXPLAIN " + consolidate.PAIRS_SQL, seeded["main"], 5, THRESHOLD
                )
            return "\n".join(row[0] for row in rows)
        finally:
            await conn.close()

    text = asyncio.run(plan())
    assert "memory_chunks__vec" not in text


def test_a_fully_acknowledged_group_is_counted_not_returned():
    name = f"consolidate-ack-{uuid.uuid4().hex[:8]}"

    async def seed():
        await namespaces.create_namespace(name)
        await _save(RULES[0], name)
        await _save(NEAR_COPY, name, allow_similar=True)

    try:
        asyncio.run(seed())
        body = _fetch(name, threshold=THRESHOLD)["namespaces"][name]
        assert body["acknowledged"] == 1
        assert body["groups"] == []
        assert body["deferred"] == []
        assert body["pairs"] == 1
    finally:
        asyncio.run(_drop(name))
