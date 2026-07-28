"""Tests for the shared asyncpg connection pool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg
import pytest

from memory_base.core.config import db_url


class StubPool:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.closed = False
        self.terminated = False

    async def close(self):
        if self.loop is not asyncio.get_running_loop() or self.loop.is_closed():
            raise RuntimeError("Event loop is closed")
        self.closed = True

    def terminate(self):
        self.terminated = True


def test_request_paths_do_not_open_direct_asyncpg_connections():
    source_root = Path(__file__).parents[2] / "src" / "memory_base"
    offenders = [
        path
        for package in ("serve", "retrieval")
        for path in (source_root / package).rglob("*.py")
        if "asyncpg.connect(" in path.read_text()
    ]
    assert offenders == []


def test_close_pool_terminates_pool_bound_to_closed_loop(monkeypatch):
    from memory_base.core import db

    pools = []

    async def create_pool(*args, **kwargs):
        del args, kwargs
        pool = StubPool()
        pools.append(pool)
        return pool

    monkeypatch.setenv("DB_URL", "postgresql://unused")
    monkeypatch.setattr(db.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "_pool_loop", None)

    asyncio.run(db.get_pool())
    asyncio.run(db.close_pool())

    assert pools[0].terminated is True
    assert pools[0].closed is False


@pytest.mark.integration
def test_pool_reuses_closes_and_rebinds_across_event_loops():
    from memory_base.core.db import close_pool, get_pool

    async def db_reachable():
        connection = await asyncpg.connect(db_url(), timeout=5)
        await connection.close()

    try:
        asyncio.run(db_reachable())
    except Exception:
        pytest.skip("DB is not configured or not reachable")

    async def reuse_then_close():
        first = await get_pool()
        assert await get_pool() is first
        await close_pool()
        replacement = await get_pool()
        assert replacement is not first
        return replacement

    first_loop_pool = asyncio.run(reuse_then_close())

    async def rebind_then_close():
        pool = await get_pool()
        await close_pool()
        return pool

    second_loop_pool = asyncio.run(rebind_then_close())
    assert second_loop_pool is not first_loop_pool
