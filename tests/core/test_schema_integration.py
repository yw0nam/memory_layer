"""The schema against a real Postgres: a database from before the card migration."""

from __future__ import annotations

import asyncio

import asyncpg
import pytest

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.core.schema import ensure_schema

pytestmark = pytest.mark.integration


async def _agent_profiles_after_ensure_schema() -> str | None:
    conn = await asyncpg.connect(db_url())
    try:
        await ensure_schema(conn)
        await conn.execute(f'CREATE TABLE "{PG_SCHEMA}".agent_profiles (id bigserial PRIMARY KEY)')
        await ensure_schema(conn)
        return await conn.fetchval("SELECT to_regclass($1)::text", f'"{PG_SCHEMA}".agent_profiles')
    finally:
        await conn.close()


def test_a_database_that_has_agent_profiles_loses_it():
    assert asyncio.run(_agent_profiles_after_ensure_schema()) is None
