"""Notes naming an entity through POST /save_memory, listed by GET /notes?entity=, on the throwaway Postgres."""

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
from memory_base.serve import api

pytestmark = pytest.mark.integration

HEADERS = {"X-API-Key": "test-key", "X-Author": "claude-code"}


def test_a_note_names_an_entity_and_the_entity_filter_returns_only_that_note():
    tag = uuid.uuid4().hex[:10]
    slug = f"it-{tag}"

    def save(text, entities):
        return {
            "content": f"integration note {tag} {text}",
            "kind": "work",
            "tags": ["it"],
            "entities": entities,
        }

    async def scenario():
        conn = await asyncpg.connect(db_url())
        now = time.time()
        await conn.execute(
            f"""INSERT INTO "{PG_SCHEMA}".entities
                (slug, kind, name, created_by, updated_by, consent_method,
                 consented_at, created_at, updated_at)
                VALUES ($1, 'thing', $2, 'test', 'test', 'import', $3, $3, $3)""",
            slug,
            f"Entity {tag}",
            now,
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
            ) as client:
                named = await client.post(
                    "/save_memory",
                    json=save("naming the entity", [slug]),
                    headers=HEADERS,
                )
                await client.post(
                    "/save_memory",
                    json=save("zebra quartz lantern", []),
                    headers=HEADERS,
                )
                unknown = await client.post(
                    "/save_memory",
                    json=save("naming a stranger", ["no-such-entity"]),
                    headers=HEADERS,
                )
                listed = await client.get(f"/notes?entity={slug}", headers=HEADERS)
            return named, unknown, listed
        finally:
            await db.close_pool()
            await conn.execute(
                f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE content_raw LIKE $1',
                f"integration note {tag}%",
            )
            await conn.execute(f'DELETE FROM "{PG_SCHEMA}".entities WHERE slug = $1', slug)
            await conn.close()

    named, unknown, listed = asyncio.run(scenario())
    assert named.status_code == 200
    assert unknown.status_code == 400
    assert "no-such-entity" in unknown.json()["error"]
    rows = listed.json()
    assert [r["id"] for r in rows] == [named.json()["id"]]
    assert rows[0]["entities"] == [slug]


def test_link_entities_previews_then_writes_only_changed_notes():
    tag = uuid.uuid4().hex[:10]
    slug, other = f"it-{tag}", f"ot-{tag}"

    async def scenario():
        conn = await asyncpg.connect(db_url())
        now = time.time()
        for entity in (slug, other):
            await conn.execute(
                f"""INSERT INTO "{PG_SCHEMA}".entities
                    (slug, kind, name, created_by, updated_by, consent_method,
                     consented_at, created_at, updated_at)
                    VALUES ($1, 'thing', $1, 'test', 'test', 'import', $2, $2, $2)""",
                entity,
                now,
            )
        ids = [f"note:default:it{tag}{n}" for n in range(2)]
        for note_id, entities in zip(ids, ([slug], [])):
            await conn.execute(
                f"""INSERT INTO "{PG_SCHEMA}".memory_chunks
                    (id, source_type, source_ref, chunk_kind, session_id, content_raw,
                     embedding, ts_last_active, metadata)
                    VALUES ($1, 'agent_note', $1, 'work', $1, $2, $3::halfvec, $4,
                            $5::jsonb)""",
                note_id,
                f"integration note {tag} {note_id}",
                "[" + ",".join(["0"] * EMB_DIM) + "]",
                now,
                json.dumps({"author": "claude-code", "entities": entities}),
            )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
            ) as client:
                body = {"ids": ids, "add": [slug], "remove": [other]}

                async def listed():
                    rows = (await client.get(f"/notes?entity={slug}", headers=HEADERS)).json()
                    return {row["id"]: row for row in rows if row["id"] in ids}

                preview = await client.post("/notes/entities", json=body, headers=HEADERS)
                after_preview = await listed()
                confirmed = await client.post(
                    "/notes/entities", json={**body, "confirm": True}, headers=HEADERS
                )
                after_confirm = await listed()
                refused = await client.post(
                    "/notes/entities",
                    json={**body, "ids": [*ids, "note:default:missing"], "confirm": True},
                    headers=HEADERS,
                )
            return ids, preview, after_preview, confirmed, after_confirm, refused
        finally:
            await db.close_pool()
            await conn.execute(
                f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE content_raw LIKE $1',
                f"integration note {tag}%",
            )
            await conn.execute(
                f'DELETE FROM "{PG_SCHEMA}".entities WHERE slug = ANY($1)', [slug, other]
            )
            await conn.close()

    ids, preview, after_preview, confirmed, after_confirm, refused = asyncio.run(scenario())
    kept, linked = ids
    assert [n["changed"] for n in preview.json()["notes"]] == [False, True]
    assert list(after_preview) == [kept]
    assert confirmed.json() == preview.json()
    assert set(after_confirm) == {kept, linked}
    assert "entities_set_at" not in after_confirm[kept]
    assert after_confirm[linked]["entities_set_by"] == "claude-code"
    assert isinstance(after_confirm[linked]["entities_set_at"], float)
    assert refused.status_code == 400 and refused.json()["ids"] == ["note:default:missing"]
