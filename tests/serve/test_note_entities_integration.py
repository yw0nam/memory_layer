"""GET /notes?entity= against the throwaway Postgres: only the note naming the entity returns."""

from __future__ import annotations

import asyncio
import time
import uuid

import asyncpg
import httpx
import pytest

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve import api
from memory_base.serve.notes import store

pytestmark = pytest.mark.integration


def test_list_notes_entity_filter_returns_only_the_naming_note():
    tag = uuid.uuid4().hex[:10]
    slug = f"it-{tag}"

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
            named = await store.save_note(
                f"integration note {tag} about the entity",
                kind="work",
                tags=["it"],
                entities=[slug],
            )
            other = await store.save_note(
                f"integration note {tag} about nothing in particular zebra",
                kind="work",
                tags=["it"],
                allow_similar=True,
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
            ) as client:
                response = await client.get(
                    f"/notes?entity={slug}", headers={"X-API-Key": "test-key"}
                )
            return named["id"], other["id"], response
        finally:
            await db.close_pool()
            await conn.execute(
                f"""DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE content_raw LIKE $1""",
                f"integration note {tag}%",
            )
            await conn.execute(f'DELETE FROM "{PG_SCHEMA}".entities WHERE slug = $1', slug)
            await conn.close()

    named_id, _other_id, response = asyncio.run(scenario())
    assert response.status_code == 200
    rows = response.json()
    assert [r["id"] for r in rows] == [named_id]
    assert rows[0]["entities"] == [slug]
