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
from memory_base.serve import api
from memory_base.serve.access import namespaces
from memory_base.serve.consolidation import groups
from memory_base.serve.notes.store import NOTE_SIMILAR_THRESHOLD, save_note

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

client = TestClient(api.app, headers={"X-API-Key": "test-key", "X-Author": "claude-code"})


async def _save(content, namespace, allow_similar=False, author="claude-code"):
    saved = await save_note(
        content,
        kind="work",
        tags=["rules"],
        namespace=namespace,
        author=author,
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
    deferred = {d["id"] for d in main["deferred"]}
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
                for statement in groups.EXACT_SEARCH_SETTINGS:
                    await conn.execute(statement)
                rows = await conn.fetch(
                    "EXPLAIN " + groups.PAIRS_SQL, seeded["main"], "claude-code", 5, THRESHOLD
                )
            return "\n".join(row[0] for row in rows)
        finally:
            await conn.close()

    text = asyncio.run(plan())
    assert "memory_chunks__vec" not in text


def test_an_acknowledged_pair_is_counted_not_grouped():
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


def test_groups_never_mix_two_authors_and_all_builds_one_set_per_author():
    name = f"consolidate-authors-{uuid.uuid4().hex[:8]}"

    async def seed():
        await namespaces.create_namespace(name)
        mine = [await _save(text, name, author="claude-code") for text in RULES]
        theirs = [await _save(text, name, author="natsume") for text in RULES]
        return mine, theirs

    try:
        mine, theirs = asyncio.run(seed())
        own = _fetch(name, threshold=THRESHOLD)["namespaces"][name]
        assert {m["id"] for g in own["groups"] for m in g["members"]} == set(mine)
        assert not {m["id"] for g in own["groups"] for m in g["members"]} & set(theirs)

        every = _fetch(name, threshold=THRESHOLD, author="all")["namespaces"][name]
        grouped = [{m["id"] for m in g["members"]} for g in every["groups"]]
        assert set(mine) in grouped
        assert set(theirs) in grouped
        for group in every["groups"]:
            assert len({m["author"] for m in group["members"]}) == 1

        body = _fetch(name, threshold=THRESHOLD, author="all")
        assert body["params"]["author"] == "all"
        named = _fetch(name, threshold=THRESHOLD, author="natsume")["namespaces"][name]
        assert {m["id"] for g in named["groups"] for m in g["members"]} == set(theirs)
    finally:
        asyncio.run(_drop(name))


def test_a_merge_keeps_the_members_author_and_the_action_the_header_author():
    name = f"consolidate-merge-{uuid.uuid4().hex[:8]}"

    async def seed():
        await namespaces.create_namespace(name)
        return [await _save(text, name, author="natsume") for text in RULES]

    try:
        rules = asyncio.run(seed())
        section = _fetch(name, threshold=THRESHOLD, author="all")["namespaces"][name]
        [group] = section["groups"]
        merged = "Coding tasks go to subagents in their own git worktrees, and the main session only reviews their output."
        body = {
            "run_id": "run-merge",
            "model": "test-model",
            "threshold": THRESHOLD,
            "verdicts": [
                {
                    "namespace": name,
                    "group_key": group["key"],
                    "idempotency_key": uuid.uuid4().hex,
                    "member_ids": sorted(m["id"] for m in group["members"]),
                    "action": "merge",
                    "merged_text": merged,
                    "reason": "the notes state one delegation rule",
                }
            ],
        }
        response = client.post(
            "/admin/consolidate/verdicts", json=body, headers={"X-Author": "claude-code"}
        )
        assert response.status_code == 200, response.text
        [result] = response.json()["results"]
        assert result["status"] == "applied", result
        conn = asyncio.run(_query_rows(name))
        replacement = next(r for r in conn if r["id"] == result["replacement_id"])
        assert replacement["metadata"]["author"] == "natsume"
        assert "shared" not in replacement["metadata"]
        actions = asyncio.run(_query_actions(name))
        [action] = actions
        assert action["author"] == "claude-code"
        assert action["member_ids"] == sorted(rules)
    finally:
        asyncio.run(_drop(name))


async def _query_rows(namespace):
    conn = await asyncpg.connect(db_url())
    try:
        rows = await conn.fetch(
            f'SELECT id, metadata FROM "{PG_SCHEMA}".memory_chunks WHERE namespace = $1',
            namespace,
        )
        return [{"id": row["id"], "metadata": json.loads(row["metadata"])} for row in rows]
    finally:
        await conn.close()


async def _query_actions(namespace):
    conn = await asyncpg.connect(db_url())
    try:
        return await conn.fetch(
            f'SELECT * FROM "{PG_SCHEMA}".consolidation_actions WHERE namespace = $1',
            namespace,
        )
    finally:
        await conn.close()
