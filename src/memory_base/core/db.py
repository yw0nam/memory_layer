"""Process-wide asyncpg connection pool."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

from memory_base.core.config import db_url

DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
DB_POOL_ACQUIRE_TIMEOUT = float(os.getenv("DB_POOL_ACQUIRE_TIMEOUT", "30"))

_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    """Lazy-create the process-wide pool, bound to the running event loop."""
    global _pool, _pool_loop

    loop = asyncio.get_running_loop()
    if _pool is not None and _pool_loop is loop:
        return _pool

    async with _pool_lock:
        if _pool is not None and _pool_loop is loop:
            return _pool
        if _pool is not None:
            stale_pool = _pool
            _pool = None
            _pool_loop = None
            try:
                stale_pool.terminate()
            except Exception:
                pass

        new_pool = await asyncpg.create_pool(
            db_url(),
            min_size=DB_POOL_MIN,
            max_size=DB_POOL_MAX,
        )
        _pool = new_pool
        _pool_loop = loop
        return new_pool


async def close_pool() -> None:
    """Close and drop the pool; safe to call when no pool exists."""
    global _pool, _pool_loop

    running_loop = asyncio.get_running_loop()
    async with _pool_lock:
        pool = _pool
        pool_loop = _pool_loop
        _pool = None
        _pool_loop = None
    if pool is None:
        return
    if pool_loop is not running_loop or pool_loop.is_closed():
        try:
            pool.terminate()
        except Exception:
            pass
        return
    await pool.close()


@asynccontextmanager
async def acquire() -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection from the shared pool."""
    pool = await get_pool()
    async with pool.acquire(timeout=DB_POOL_ACQUIRE_TIMEOUT) as connection:
        yield connection
