"""Deleting an entity on the throwaway Postgres: unlink notes, delete facts, free the names."""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import asyncpg
import httpx
import pytest

from memory_base.core import db
from memory_base.core.config import EMB_DIM, PG_SCHEMA, db_url
from memory_base.retrieval.search import metadata_dict
from memory_base.serve import api
from memory_base.serve.common.entities import lock_slugs, require_registered
from memory_base.serve.consolidation import verdicts
from memory_base.serve.entities import store

pytestmark = pytest.mark.integration

HEADERS = {"X-API-Key": "test-key"}
ZERO = "[" + ",".join(["0"] * EMB_DIM) + "]"
NOTES = f'"{PG_SCHEMA}".memory_chunks'
FACTS = f'"{PG_SCHEMA}".facts'
ENTITIES = f'"{PG_SCHEMA}".entities'


async def _entity(conn, slug, name=None):
    now = time.time()
    await conn.execute(
        f"""INSERT INTO {ENTITIES}
            (slug, kind, name, created_by, updated_by, consent_method,
             consented_at, created_at, updated_at)
            VALUES ($1, 'thing', $2, 't', 't', 'import', $3, $3, $3)""",
        slug,
        name or slug,
        now,
    )


async def _note(conn, note_id, tag, metadata, archived_at=None):
    await conn.execute(
        f"""INSERT INTO {NOTES}
            (id, source_type, source_ref, chunk_kind, session_id, content_raw,
             embedding, ts_last_active, archived_at, metadata)
            VALUES ($1, 'agent_note', 'save_memory', 'work', $1, $2, $3::halfvec, $4, $5,
                    $6::jsonb)""",
        note_id,
        f"integration note {tag} {note_id}",
        ZERO,
        time.time(),
        archived_at,
        json.dumps(metadata),
    )


async def _fact(conn, entity, attribute, value_entity=None, archived_at=None, replaced_by=None):
    return await conn.fetchval(
        f"""INSERT INTO {FACTS}
            (id, namespace, entity, attribute, value_text, value_entity, author, created_at,
             archived_at, replaced_by)
            VALUES (gen_random_uuid(), 'default', $1, $2, $3, $4, 't', $5, $6, $7)
            RETURNING id""",
        entity,
        attribute,
        None if value_entity else "v",
        value_entity,
        time.time(),
        archived_at,
        replaced_by,
    )


async def _note_metadata(conn, note_id):
    return metadata_dict(
        await conn.fetchval(f"SELECT metadata FROM {NOTES} WHERE id = $1", note_id)
    )


async def _cleanup(conn, tag):
    await db.close_pool()
    await conn.execute(f"DELETE FROM {FACTS} WHERE entity LIKE $1", f"it-{tag}-%")
    await conn.execute(f"DELETE FROM {NOTES} WHERE content_raw LIKE $1", f"integration note {tag}%")
    await conn.execute(f'DELETE FROM "{PG_SCHEMA}".consolidation_actions WHERE run_id = $1', tag)
    await conn.execute(f"DELETE FROM {ENTITIES} WHERE slug LIKE $1", f"it-{tag}-%")
    await conn.close()


def test_delete_unlinks_notes_removes_facts_and_frees_the_names():
    tag = uuid.uuid4().hex[:10]
    slug, other = f"it-{tag}-x", f"it-{tag}-o"
    active, archived, untouched = (f"note:default:it{tag}{n}" for n in "abc")
    original = {
        "author": "hermes",
        "tags": ["t"],
        "entities": [slug, other],
        "entities_set_by": "z",
    }

    async def scenario():
        conn = await asyncpg.connect(db_url())
        try:
            await _entity(conn, slug, f"Kuro {tag}")
            await _entity(conn, other)
            await _note(conn, active, tag, original)
            await _note(conn, archived, tag, {"entities": [slug]}, archived_at=time.time())
            await _note(conn, untouched, tag, {"entities": [other]})
            await _fact(conn, slug, "city")
            await _fact(conn, slug, "self", value_entity=slug)
            incoming = await _fact(conn, other, "works-on", value_entity=slug)
            await _fact(conn, other, "role", archived_at=time.time(), replaced_by=incoming)
            before = {n: await _note_metadata(conn, n) for n in (active, archived, untouched)}
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
            ) as client:
                url = f"/entities/{slug}/delete"
                body = {"author": "claude-code"}
                dry = await client.post(url, json={**body, "dry_run": True}, headers=HEADERS)
                untouched_by_dry = {
                    n: await _note_metadata(conn, n) for n in (active, archived, untouched)
                }
                kept_by_dry = await conn.fetchval(f"SELECT count(*) FROM {FACTS}")
                stale = await client.post(
                    url, json={**body, "expect": {"notes": 1, "facts": 2, "incoming": 1}},
                    headers=HEADERS,
                )  # fmt: skip
                expect = {"notes": 2, "facts": 2, "incoming": 1}
                done = await client.post(url, json={**body, "expect": expect}, headers=HEADERS)
                again = await client.post(url, json=body, headers=HEADERS)
                reuse = await client.post(
                    "/entities",
                    json={
                        "entities": [{"slug": slug, "kind": "thing", "name": f"Kuro {tag}"}],
                        **body,
                        "consent_method": "conversation",
                    },
                    headers=HEADERS,
                )
            after = {n: await _note_metadata(conn, n) for n in (active, archived, untouched)}
            archived_at = await conn.fetchval(
                f"SELECT archived_at FROM {NOTES} WHERE id=$1", archived
            )
            facts = await conn.fetch(f"SELECT entity, attribute, replaced_by FROM {FACTS} WHERE entity LIKE $1", f"it-{tag}-%")  # fmt: skip
            linked = await conn.fetchval(f"SELECT count(*) FROM {NOTES} WHERE metadata->'entities' ? $1", slug)  # fmt: skip
            return dict(**locals())
        finally:
            await _cleanup(conn, tag)

    r = asyncio.run(scenario())
    plan = {"note_ids": sorted([active, archived]), "notes": 2, "facts": 2, "incoming": 1}
    assert r["dry"].status_code == 200 and r["dry"].json() == plan
    assert r["untouched_by_dry"] == r["before"] and r["kept_by_dry"] == 4
    assert r["stale"].status_code == 409
    assert r["done"].status_code == 200 and r["done"].json() == plan
    assert r["again"].status_code == 400
    stamped = {"entities_set_by": "claude-code"}
    now_active = {**original, "entities": [other], **stamped}
    assert {k: v for k, v in r["after"][active].items() if k != "entities_set_at"} == now_active
    assert isinstance(r["after"][active]["entities_set_at"], float)
    assert set(r["after"][archived]) == {"entities_set_by", "entities_set_at"}
    assert r["archived_at"] is not None
    assert r["after"][untouched] == r["before"][untouched]
    assert [(f["entity"], f["attribute"], f["replaced_by"]) for f in r["facts"]] == [
        (other, "role", None)
    ]
    assert r["reuse"].status_code == 200 and r["linked"] == 0


def test_undo_that_would_restore_a_deleted_slug_is_refused():
    tag = uuid.uuid4().hex[:10]
    slug = f"it-{tag}-x"
    gone, survivor = f"note:default:it{tag}a", f"note:default:it{tag}b"
    applied_at = time.time() - 10

    async def scenario():
        conn = await asyncpg.connect(db_url())
        try:
            await _entity(conn, slug)
            await _note(
                conn, gone, tag, {"entities": [slug], "consolidated_into": [survivor]}, applied_at
            )
            await _note(conn, survivor, tag, {})
            action_id = await conn.fetchval(
                f"""INSERT INTO "{PG_SCHEMA}".consolidation_actions
                    (idempotency_key, payload_hash, run_id, namespace, action, group_key,
                     member_ids, archived_ids, survivor_ids, prior, applied_at, author,
                     reason, result)
                    VALUES ($1, 'h', $2, 'default', 'retire', $1, $3, $4, $5, $6::jsonb, $7,
                            'it', 'r', '{{}}'::jsonb)
                    RETURNING id""",
                f"it-{tag}", tag, [gone, survivor], [gone], [survivor],
                json.dumps({gone: {"entities": [slug]}}), applied_at,
            )  # fmt: skip
            await store.delete(slug, "claude-code")
            with pytest.raises(verdicts.UndoRefused) as refused:
                await verdicts.undo(action_id, "claude-code")
            await _entity(conn, slug)
            with pytest.raises(verdicts.UndoRefused) as reregistered:
                await verdicts.undo(action_id, "claude-code")
            archived_at = await conn.fetchval(f"SELECT archived_at FROM {NOTES} WHERE id=$1", gone)
            return refused.value, reregistered.value, archived_at
        finally:
            await _cleanup(conn, tag)

    refused, reregistered, archived_at = asyncio.run(scenario())
    assert "entities changed after the action" in str(refused)
    assert "entities changed after the action" in str(reregistered)
    assert archived_at is not None


def test_a_delete_waits_for_a_write_that_holds_the_slug_and_then_unlinks_it():
    tag = uuid.uuid4().hex[:10]
    slug, note = f"it-{tag}-x", f"note:default:it{tag}a"

    async def scenario():
        conn = await asyncpg.connect(db_url())
        writer = await asyncpg.connect(db_url())
        try:
            await _entity(conn, slug)
            async with writer.transaction():
                await lock_slugs(writer)
                await require_registered(writer, [slug])
                deleting = asyncio.ensure_future(store.delete(slug, "claude-code"))
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(deleting), 1)
                await _note(writer, note, tag, {"entities": [slug]})
            await asyncio.wait_for(deleting, 30)
            return (
                await _note_metadata(conn, note),
                await conn.fetchval(f"SELECT count(*) FROM {ENTITIES} WHERE slug = $1", slug),
            )
        finally:
            await writer.close()
            await _cleanup(conn, tag)

    metadata, remaining = asyncio.run(scenario())
    assert "entities" not in metadata
    assert remaining == 0
