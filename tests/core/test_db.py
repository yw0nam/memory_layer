"""Tests for the shared asyncpg connection pool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def test_request_paths_do_not_open_direct_asyncpg_connections():
    source_root = Path(__file__).parents[2] / "src" / "memory_base"
    offenders = [
        path
        for package in ("serve", "retrieval")
        for path in (source_root / package).rglob("*.py")
        if "asyncpg.connect(" in path.read_text()
    ]
    assert offenders == []


@pytest.mark.integration
def test_pool_reuses_closes_and_rebinds_across_event_loops():
    from memory_base.core.db import close_pool, get_pool

    async def reuse_then_close():
        first = await get_pool()
        assert await get_pool() is first
        await close_pool()
        replacement = await get_pool()
        assert replacement is not first
        return replacement

    first_loop_pool = asyncio.run(reuse_then_close())
    second_loop_pool = asyncio.run(get_pool())
    assert second_loop_pool is not first_loop_pool
    asyncio.run(close_pool())
