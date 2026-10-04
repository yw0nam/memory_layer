"""Integration tests for consolidation verdicts, apply, lineage, and undo.

They run against the throwaway Postgres with the live embedder. Each test seeds its own
namespace through ``save_note``. Every pair inside the rule trio, the changelog trio, and
the tea pair scores above the threshold passed here and below the save-time similarity
check; pairs across those sets and the unrelated notes score below the threshold.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import asyncpg
import httpx
import pytest
from starlette.testclient import TestClient

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api, auth, namespaces
from memory_base.serve.notes.store import note_id, save_note

pytestmark = pytest.mark.integration

PARAMS = {"threshold": 0.60, "neighbors": 5, "max_group": 6, "max_group_chars": 12000}
RULES = (
    "The owner wants coding tasks delegated to subagents in git worktrees, "
    "and the main session only reviews their output.",
    "Delegate implementation work to a subagent working in its own worktree; "
    "the main agent reviews rather than writes code.",
    "When the user assigns a coding job, hand it to a worktree subagent "
    "and keep the primary session for review only.",
)
CHANGELOG = (
    "Release notes for this project are generated from git history and never edited by hand.",
    "Do not hand-edit the changelog; it is produced automatically from the commit log at "
    "release time.",
    "The changelog file is built from commits when a version is tagged, so manual edits get "
    "overwritten.",
)
TEA = (
    "The user drinks green tea every morning instead of coffee.",
    "Morning routine: no coffee; the user prefers a pot of sencha.",
)
UNRELATED = (
    "The staging Postgres runs on port 5433 with pgvector 0.8 installed.",
    "The user's cat is named Mochi and likes salmon treats.",
)
MERGED_RULE = (
    "Coding tasks go to subagents in their own git worktrees, and the main session only "
    "reviews their output."
)
SUCCESSOR = (
    "Coding tasks go to worktree subagents; the main session reviews their output and merges "
    "it once the checks pass."
)
RESTATED_MERGE = (
    "Implementation is handed to worktree subagents; the lead session limits itself to reviewing."
)
SECOND_SUCCESSOR = (
    "Coding tasks go to worktree subagents; the main session reviews, merges after green "
    "checks, and deletes the worktree."
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


async def _query(sql, *args):
    conn = await asyncpg.connect(db_url())
    try:
        return await conn.fetch(sql, *args)
    finally:
        await conn.close()


def query(sql, *args):
    return asyncio.run(_query(sql, *args))


async def _save(content, namespace, supersedes=None):
    saved = await save_note(
        content,
        kind="work",
        tags=["rules"],
        namespace=namespace,
        author="claude-code",
        supersedes=supersedes,
        allow_similar=supersedes is not None,
    )
    return saved["id"]


def save(content, namespace, supersedes=None):
    return asyncio.run(_save(content, namespace, supersedes))


@pytest.fixture()
def space():
    """A fresh namespace seeded with the given sets; removed with its actions afterwards."""
    created: list[str] = []

    def seed(*sets):
        name = f"verdicts-{uuid.uuid4().hex[:8]}"
        created.append(name)

        async def run():
            await namespaces.create_namespace(name)
            return [[await _save(text, name) for text in texts] for texts in sets]

        return name, asyncio.run(run())

    yield seed
    for name in created:
        query(f'DELETE FROM "{PG_SCHEMA}".consolidation_actions WHERE namespace = $1', name)
        query(f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE namespace = $1', name)
        query(f'DELETE FROM "{PG_SCHEMA}".namespaces WHERE name = $1', name)


def rows(namespace):
    found = query(
        f'SELECT * FROM "{PG_SCHEMA}".memory_chunks WHERE namespace = $1 ORDER BY id', namespace
    )
    return [tuple(row.items()) for row in found]


def note_row(note_id_):
    [row] = query(
        f"""SELECT archived_at, metadata, chunk_kind, content_raw
            FROM "{PG_SCHEMA}".memory_chunks WHERE id = $1""",
        note_id_,
    )
    return {**dict(row), "metadata": json.loads(row["metadata"])}


def actions(namespace):
    return query(
        f'SELECT * FROM "{PG_SCHEMA}".consolidation_actions WHERE namespace = $1 ORDER BY id',
        namespace,
    )


def groups(namespace):
    response = client.get(
        "/admin/consolidate/groups",
        params={"namespace": namespace, **{k: str(v) for k, v in PARAMS.items()}},
    )
    assert response.status_code == 200, response.text
    return response.json()["namespaces"][namespace]


def group_of(section, ids):
    [found] = [g for g in section["groups"] if {m["id"] for m in g["members"]} == set(ids)]
    return found


def verdict(namespace, group, action, key=None, **fields):
    return {
        "namespace": namespace,
        "group_key": group["key"],
        "idempotency_key": key or uuid.uuid4().hex,
        "member_ids": sorted(m["id"] for m in group["members"]),
        "action": action,
        "reason": "the notes state one delegation rule",
        **fields,
    }


def request_body(*items, run_id="run-1", **top):
    return {
        "run_id": run_id,
        "author": "consolidator",
        "model": "test-model",
        **PARAMS,
        **top,
        "verdicts": list(items),
    }


def submit(*items, **top):
    response = client.post("/admin/consolidate/verdicts", json=request_body(*items, **top))
    assert response.status_code == 200, response.text
    return response.json()["results"]


def undo(action_id):
    return client.post(
        "/admin/consolidate/undo", json={"action_id": action_id, "author": "consolidator"}
    )


def concurrently(*bodies):
    async def run():
        # The pool is created before the requests race: its lock binds to one event loop.
        await db.get_pool()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": "test-key"},
        ) as http:
            responses = await asyncio.gather(
                *(http.post("/admin/consolidate/verdicts", json=body) for body in bodies)
            )
        return [r.json()["results"][0] for r in responses]

    return asyncio.run(run())


def test_a_dry_run_plans_without_writing(space):
    ns, (rules, _) = space(RULES, UNRELATED)
    group = group_of(groups(ns), rules)
    before = rows(ns)
    [planned, invalid] = submit(
        verdict(ns, group, "merge", merged_text=MERGED_RULE),
        verdict(ns, group, "retire", retire_ids=sorted(rules)),
        dry_run=True,
    )
    rid = note_id(ns, MERGED_RULE)
    assert planned["status"] == "planned"
    assert planned["action_id"] is None
    assert planned["archived_ids"] == sorted(rules)
    assert planned["survivor_ids"] == [rid]
    assert planned["replacement_id"] == rid
    assert invalid["status"] == "rejected"
    assert rows(ns) == before
    assert actions(ns) == []


def test_a_merge_applies_with_lineage_and_undo_restores_the_members_exactly(space):
    ns, (rules, unrelated) = space(RULES, UNRELATED)
    original = {i: note_row(i) for i in rules}
    group = group_of(groups(ns), rules)
    [applied] = submit(verdict(ns, group, "merge", merged_text=f" {MERGED_RULE} "))
    rid = note_id(ns, MERGED_RULE)
    assert applied["status"] == "applied", applied
    assert applied["replacement_id"] == rid

    [action] = actions(ns)
    assert action["id"] == applied["action_id"]
    assert action["action"] == "merge"
    assert action["replacement_created"] is True
    assert list(action["member_ids"]) == sorted(rules)
    assert json.loads(action["result"]) == applied

    replacement = note_row(rid)
    assert replacement["archived_at"] is None
    assert replacement["content_raw"] == MERGED_RULE
    assert replacement["metadata"]["merged_from"] == sorted(rules)
    assert set(replacement["metadata"]["merged_dates"]) == set(rules)
    assert replacement["metadata"]["consolidation_action"] == action["id"]
    assert replacement["metadata"]["author"] == "consolidator"
    assert replacement["metadata"]["tags"] == ["rules"]
    for member in rules:
        archived = note_row(member)
        assert archived["archived_at"] == action["applied_at"]
        assert archived["metadata"]["consolidated_into"] == [rid]
        assert archived["metadata"]["archived_by"] == "consolidator"

    section = groups(ns)
    assert not {m["id"] for g in section["groups"] for m in g["members"]} & set(rules)

    listing = client.get("/admin/consolidate/actions", params={"namespace": ns})
    assert listing.status_code == 200
    body = listing.json()
    assert [a["id"] for a in body["actions"]] == [action["id"]]
    assert set(body["notes"]) == {*rules, rid}
    assert body["notes"][rid]["merged_from"] == sorted(rules)
    assert body["notes"][rules[0]]["consolidated_into"] == [rid]

    response = undo(action["id"])
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["restored_ids"] == sorted(rules)
    assert result["archived_ids"] == [rid]
    for member in rules:
        assert note_row(member) == original[member]
    replacement = note_row(rid)
    assert replacement["archived_at"] is not None
    assert replacement["metadata"]["undone_action"] == action["id"]
    [action] = actions(ns)
    assert action["undone_at"] is not None
    assert action["undone_by"] == "consolidator"

    section = groups(ns)
    assert section["groups"] == []
    assert section["cached"] == 1
    again = undo(action["id"])
    assert again.status_code == 200
    assert again.json() == result


def test_a_retire_archives_only_the_named_member(space):
    ns, (rules, _) = space(RULES, UNRELATED)
    group = group_of(groups(ns), rules)
    retired, *kept = sorted(rules)
    kept_before = {i: note_row(i) for i in kept}
    [applied] = submit(verdict(ns, group, "retire", retire_ids=[retired]))
    assert applied["status"] == "applied", applied
    assert applied["archived_ids"] == [retired]
    assert applied["survivor_ids"] == kept
    row = note_row(retired)
    assert row["archived_at"] is not None
    assert row["metadata"]["consolidated_into"] == kept
    assert {i: note_row(i) for i in kept} == kept_before


def test_a_resubmitted_verdict_is_a_duplicate(space):
    ns, (rules, _) = space(RULES, UNRELATED)
    item = verdict(ns, group_of(groups(ns), rules), "keep", key="keep-1")
    [first] = submit(item)
    [second] = submit(item)
    assert first["status"] == "applied"
    assert second == {**first, "status": "duplicate"}
    assert len(actions(ns)) == 1
    [reused] = submit(item, run_id="run-2")
    assert reused["status"] == "rejected"
    assert reused["reason"] == "idempotency key reused with a different payload"


def test_concurrent_requests_respect_the_action_cap(space):
    ns, (rules, changelog, _) = space(RULES, CHANGELOG, TEA)
    section = groups(ns)
    first = verdict(ns, group_of(section, rules), "retire", retire_ids=[min(rules)])
    second = verdict(ns, group_of(section, changelog), "retire", retire_ids=[min(changelog)])
    results = concurrently(request_body(first, max_actions=1), request_body(second, max_actions=1))
    assert sorted(r["status"] for r in results) == ["applied", "rejected"]
    assert [r["reason"] for r in results if r["status"] == "rejected"] == ["action cap reached"]
    assert len(actions(ns)) == 1


def test_identical_concurrent_requests_apply_once(space):
    ns, (rules, _) = space(RULES, UNRELATED)
    item = verdict(ns, group_of(groups(ns), rules), "retire", retire_ids=[min(rules)])
    results = concurrently(request_body(item), request_body(item))
    assert sorted(r["status"] for r in results) == ["applied", "duplicate"]
    assert len(actions(ns)) == 1


def test_keep_is_applied_at_the_action_cap(space):
    ns, (rules, changelog, tea) = space(RULES, CHANGELOG, TEA)
    section = groups(ns)
    [retired] = submit(
        verdict(ns, group_of(section, rules), "retire", retire_ids=[min(rules)]), max_actions=1
    )
    [kept] = submit(verdict(ns, group_of(section, changelog), "keep"), max_actions=1)
    [capped] = submit(
        verdict(ns, group_of(section, tea), "retire", retire_ids=[min(tea)]), max_actions=1
    )
    assert (retired["status"], kept["status"], capped["status"]) == (
        "applied",
        "applied",
        "rejected",
    )
    assert capped["reason"] == "action cap reached"


def test_a_supersede_between_fetch_and_submit_is_stale(space):
    ns, (rules, _) = space(RULES, UNRELATED)
    group = group_of(groups(ns), rules)
    save(SUCCESSOR, ns, supersedes=rules[0])
    [result] = submit(verdict(ns, group, "retire", retire_ids=[rules[1]]))
    assert result["status"] == "stale"
    assert isinstance(result["current_groups"], list)
    assert actions(ns) == []


def test_undo_is_refused_after_a_member_was_archived_restored_and_archived(space):
    ns, (rules, _) = space(RULES, UNRELATED)
    [applied] = submit(verdict(ns, group_of(groups(ns), rules), "merge", merged_text=MERGED_RULE))
    member = rules[0]
    restored = client.post("/admin/restore", json={"ids": [member], "confirm": True})
    assert restored.json() == {"restored": 1}
    archived = client.post(
        "/admin/archive", json={"ids": [member], "author": "consolidator", "confirm": True}
    )
    assert archived.json()["archived"] == 1
    before = rows(ns)
    response = undo(applied["action_id"])
    assert response.status_code == 409
    assert member in response.json()["error"]
    assert rows(ns) == before


def test_undo_is_refused_when_an_active_note_supersedes_the_replacement(space):
    ns, (rules, _) = space(RULES, UNRELATED)
    [applied] = submit(verdict(ns, group_of(groups(ns), rules), "merge", merged_text=MERGED_RULE))
    rid = applied["replacement_id"]
    successor = save(SUCCESSOR, ns, supersedes=rid)
    assert note_row(rid)["metadata"]["replaced_by"] == successor
    assert client.post("/admin/restore", json={"ids": [rid], "confirm": True}).status_code == 200
    restored = note_row(rid)
    assert "replaced_by" not in restored["metadata"]
    assert "archived_by" not in restored["metadata"]
    before = rows(ns)
    response = undo(applied["action_id"])
    assert response.status_code == 409
    assert successor in response.json()["error"]
    assert rows(ns) == before


def test_undo_is_refused_through_an_archived_intermediate_successor(space):
    ns, (rules, _) = space(RULES, UNRELATED)
    [applied] = submit(verdict(ns, group_of(groups(ns), rules), "merge", merged_text=MERGED_RULE))
    rid = applied["replacement_id"]
    middle = save(SUCCESSOR, ns, supersedes=rid)
    last = save(SECOND_SUCCESSOR, ns, supersedes=middle)
    assert client.post("/admin/restore", json={"ids": [rid], "confirm": True}).status_code == 200
    before = rows(ns)
    [action_before] = actions(ns)
    response = undo(applied["action_id"])
    assert response.status_code == 409
    assert last in response.json()["error"]
    assert rows(ns) == before
    assert actions(ns) == [action_before]


def test_a_later_keep_on_the_replacement_does_not_block_the_undo(space):
    ns, (rules, _) = space(RULES, UNRELATED)
    [merged] = submit(verdict(ns, group_of(groups(ns), rules), "merge", merged_text=MERGED_RULE))
    rid = merged["replacement_id"]
    restated = save(RESTATED_MERGE, ns)
    [kept] = submit(verdict(ns, group_of(groups(ns), [rid, restated]), "keep"))
    assert kept["status"] == "applied", kept
    response = undo(merged["action_id"])
    assert response.status_code == 200, response.text
    assert note_row(rid)["archived_at"] is not None
    assert all(note_row(member)["archived_at"] is None for member in rules)
