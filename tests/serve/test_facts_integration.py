"""Facts and the entity card on the throwaway Postgres: write, read both sides, replace, delete."""

from __future__ import annotations

import asyncio
import time
import uuid

import asyncpg
import httpx
import pytest

from memory_base.core import db
from memory_base.core.config import EMB_DIM, PG_SCHEMA, db_url
from memory_base.serve import api

pytestmark = pytest.mark.integration

HEADERS = {"X-API-Key": "test-key"}


def test_facts_are_written_read_from_both_sides_replaced_and_archived_with_their_note(
    monkeypatch,
):
    tag = uuid.uuid4().hex[:10]
    person, project, note = f"it-{tag}-p", f"it-{tag}-j", f"note:default:it-{tag}"

    async def scenario():
        conn = await asyncpg.connect(db_url())
        now = time.time()
        for slug in (person, project):
            await conn.execute(
                f"""INSERT INTO "{PG_SCHEMA}".entities
                    (slug, kind, name, created_by, updated_by, consent_method,
                     consented_at, created_at, updated_at)
                    VALUES ($1, 'thing', $1, 't', 't', 'import', $2, $2, $2)""",
                slug,
                now,
            )
        await conn.execute(
            f"""INSERT INTO "{PG_SCHEMA}".memory_chunks
                (id, source_type, source_ref, chunk_kind, session_id, content_raw,
                 embedding, ts_last_active)
                VALUES ($1, 'agent_note', 'save_memory', 'work', $1, 'it note',
                        $2::halfvec, $3)""",
            note,
            "[" + ",".join(["0"] * EMB_DIM) + "]",
            now,
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
            ) as client:

                async def write(*ops):
                    response = await client.post(
                        "/admin/facts",
                        json={"author": "consolidator", "ops": list(ops)},
                        headers=HEADERS,
                    )
                    assert response.status_code == 200, response.text
                    return response.json()["results"]

                relation, text = await write(
                    {"op": "add", "entity": person, "attribute": "works-on",
                     "value_entity": project, "note_id": note},
                    {"op": "add", "entity": person, "attribute": "city",
                     "value_text": "Busan", "note_id": note},
                )  # fmt: skip
                outgoing = (await client.get(f"/entities/{person}", headers=HEADERS)).json()
                incoming = (await client.get(f"/entities/{project}", headers=HEADERS)).json()
                [replaced] = await write(
                    {"op": "replace", "id": text["id"], "value_text": "Seoul", "note_id": note}
                )
                past = (
                    await client.get(f"/entities/{person}?include_past=true", headers=HEADERS)
                ).json()
                await client.post(
                    "/admin/notes/delete", json={"ids": [note], "confirm": True}, headers=HEADERS
                )
                after = (
                    await client.get(f"/entities/{person}?include_past=true", headers=HEADERS)
                ).json()
            return relation, text, replaced, outgoing, incoming, past, after
        finally:
            await db.close_pool()
            await conn.execute(f'DELETE FROM "{PG_SCHEMA}".facts WHERE entity = $1', person)
            await conn.execute(f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE id = $1', note)
            await conn.execute(
                f'DELETE FROM "{PG_SCHEMA}".entities WHERE slug = ANY($1)', [person, project]
            )
            await conn.close()

    relation, text, replaced, outgoing, incoming, past, after = asyncio.run(scenario())
    assert {f["id"] for f in outgoing["facts"]} == {relation["id"], text["id"]}
    assert [f["id"] for f in incoming["incoming"]] == [relation["id"]]
    current = {f["id"] for f in past["facts"] if f["archived_at"] is None}
    assert current == {relation["id"], replaced["id"]}
    assert {f["id"]: f["replaced_by"] for f in past["facts"]}[text["id"]] == replaced["id"]
    assert all(f["archived_at"] is not None and f["note_id"] is None for f in after["facts"])
