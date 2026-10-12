"""Integration tests for the entity registry against the throwaway Postgres.

Each test names its entities with a random tag and deletes its rows afterwards.
"""

from __future__ import annotations

import asyncio
import uuid

import asyncpg
import pytest

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.serve.entities import store

pytestmark = pytest.mark.integration


def _tag() -> str:
    return uuid.uuid4().hex[:10]


def _item(tag: str, suffix: str, name: str) -> dict:
    return {"slug": f"it-{tag}-{suffix}", "kind": "thing", "name": name}


async def _run(coroutine, tag: str):
    try:
        return await coroutine
    finally:
        await db.close_pool()
        conn = await asyncpg.connect(db_url())
        try:
            await conn.execute(
                f'DELETE FROM "{PG_SCHEMA}".entities WHERE slug LIKE $1', f"it-{tag}-%"
            )
        finally:
            await conn.close()


def test_a_created_entity_is_listed_and_its_name_is_then_held():
    tag = _tag()

    async def scenario():
        [row] = await store.upsert([_item(tag, "a", f"Kuro {tag}")], "claude-code", "conversation")
        listed = [e["slug"] for e in await store.list_entities(False)]
        with pytest.raises(store.Collision) as held:
            await store.upsert([_item(tag, "b", f"KURO {tag}")], "claude-code", "conversation")
        return row, listed, held.value

    row, listed, held = asyncio.run(_run(scenario(), tag))
    assert row["slug"] in listed
    assert held.holder == f"it-{tag}-a"


def test_two_concurrent_creates_with_one_name_let_exactly_one_win():
    tag = _tag()

    async def scenario():
        return await asyncio.gather(
            *(
                store.upsert([_item(tag, suffix, f"Shiro {tag}")], "claude-code", "conversation")
                for suffix in ("a", "b")
            ),
            return_exceptions=True,
        )

    results = asyncio.run(_run(scenario(), tag))
    assert sum(isinstance(r, store.Collision) for r in results) == 1
    assert sum(isinstance(r, list) for r in results) == 1
