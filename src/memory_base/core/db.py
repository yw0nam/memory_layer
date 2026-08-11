"""Process-wide asyncpg connection pool."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

from memory_base.core.config import db_url, require_env
from memory_base.core.schema import TABLES_QUERY_ROLE

DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
DB_POOL_ACQUIRE_TIMEOUT = float(os.getenv("DB_POOL_ACQUIRE_TIMEOUT", "30"))

_pool: asyncpg.Pool | None = None
_pool_loop: asyncio.AbstractEventLoop | None = None
# asyncio.Lock binds only on contention; later cross-loop contention can raise RuntimeError.
# Uncontended cross-loop use remains safe.
_pool_lock = asyncio.Lock()

_table_query_pool: asyncpg.Pool | None = None
_table_query_pool_loop: asyncio.AbstractEventLoop | None = None
_table_query_pool_lock = asyncio.Lock()


async def _init_table_query_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec(
        "jsonb",
        schema="pg_catalog",
        encoder=json.dumps,
        decoder=json.loads,
        format="text",
    )


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
    try:
        await asyncio.wait_for(pool.close(), timeout=10)
    except (TimeoutError, asyncio.TimeoutError):
        pool.terminate()


async def get_table_query_pool() -> asyncpg.Pool:
    """Lazy-create the dedicated pool authenticated as the restricted query role."""
    global _table_query_pool, _table_query_pool_loop

    loop = asyncio.get_running_loop()
    if _table_query_pool is not None and _table_query_pool_loop is loop:
        return _table_query_pool

    async with _table_query_pool_lock:
        if _table_query_pool is not None and _table_query_pool_loop is loop:
            return _table_query_pool
        if _table_query_pool is not None:
            stale_pool = _table_query_pool
            _table_query_pool = None
            _table_query_pool_loop = None
            try:
                stale_pool.terminate()
            except Exception:
                pass

        new_pool = await asyncpg.create_pool(
            db_url(),
            user=TABLES_QUERY_ROLE,
            password=require_env("TABLES_QUERY_PASSWORD"),
            min_size=1,
            max_size=2,
            init=_init_table_query_connection,
        )
        _table_query_pool = new_pool
        _table_query_pool_loop = loop
        return new_pool


async def close_table_query_pool() -> None:
    """Close and drop the dedicated table-query pool."""
    global _table_query_pool, _table_query_pool_loop

    running_loop = asyncio.get_running_loop()
    async with _table_query_pool_lock:
        pool = _table_query_pool
        pool_loop = _table_query_pool_loop
        _table_query_pool = None
        _table_query_pool_loop = None
    if pool is None:
        return
    if pool_loop is not running_loop or pool_loop.is_closed():
        try:
            pool.terminate()
        except Exception:
            pass
        return
    try:
        await asyncio.wait_for(pool.close(), timeout=10)
    except (TimeoutError, asyncio.TimeoutError):
        pool.terminate()


@asynccontextmanager
async def acquire(timeout: float | None = None) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection from the shared pool."""
    pool = await get_pool()
    acquire_timeout = DB_POOL_ACQUIRE_TIMEOUT if timeout is None else timeout
    async with pool.acquire(timeout=acquire_timeout) as connection:
        yield connection


@asynccontextmanager
async def acquire_table_query() -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection authenticated as the restricted table-query role."""
    pool = await get_table_query_pool()
    async with pool.acquire(timeout=DB_POOL_ACQUIRE_TIMEOUT) as connection:
        yield connection
