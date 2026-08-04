"""Explicit namespace registry backing POST/GET/DELETE /namespaces."""

from __future__ import annotations

import re
import time
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once

DEFAULT_NAMESPACE = "default"
SLUG_RE = re.compile(r"^[a-z0-9_-]{1,64}$")


class NamespaceError(ValueError):
    """Base error for namespace registry violations; REST maps it to 400."""


class NamespaceExistsError(NamespaceError):
    """A namespace with this name is already registered."""


class NamespaceReservedError(NamespaceError):
    """The reserved 'default' namespace cannot be deleted."""


class NamespaceNotFoundError(NamespaceError):
    """The namespace is not registered."""


class NamespaceNotEmptyError(NamespaceError):
    """The namespace still has memory_chunks rows."""


def validate_namespace_name(name: Any) -> str:
    """Validate a namespace slug: lowercase letters, digits, '_'/'-', 1-64 chars."""
    if not isinstance(name, str) or not SLUG_RE.fullmatch(name):
        raise NamespaceError("namespace must match ^[a-z0-9_-]{1,64}$")
    return name


async def _exists_in_conn(conn: Any, name: str) -> bool:
    return bool(
        await conn.fetchval(
            f'SELECT EXISTS(SELECT 1 FROM "{PG_SCHEMA}".namespaces WHERE name = $1)', name
        )
    )


async def require_registered(conn: Any, name: str) -> None:
    """Raise NamespaceError unless name is registered; reuses the caller's connection."""
    if not await _exists_in_conn(conn, name):
        raise NamespaceError(f"unregistered namespace: {name}")


async def namespace_exists(name: str) -> bool:
    """Return whether name is registered."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        return await _exists_in_conn(conn, name)


async def create_namespace(name: Any) -> dict[str, Any]:
    """Register a new namespace; raises NamespaceExistsError on duplicate."""
    validate_namespace_name(name)
    now = time.time()
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        inserted = await conn.fetchval(
            f"""
            INSERT INTO "{PG_SCHEMA}".namespaces (name, created_at)
            VALUES ($1, $2)
            ON CONFLICT (name) DO NOTHING
            RETURNING name
            """,
            name,
            now,
        )
    if inserted is None:
        raise NamespaceExistsError(f"namespace already exists: {name}")
    return {"name": name, "created_at": now}


async def list_namespaces() -> list[dict[str, Any]]:
    """List every registered namespace, oldest first."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f'SELECT name, created_at FROM "{PG_SCHEMA}".namespaces ORDER BY created_at'
        )
    return [dict(row) for row in rows]


async def delete_namespace(name: str) -> None:
    """Unregister a namespace; refuses the reserved default, unknown, or non-empty ones."""
    if name == DEFAULT_NAMESPACE:
        raise NamespaceReservedError("the 'default' namespace is reserved and cannot be deleted")
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            if not await _exists_in_conn(conn, name):
                raise NamespaceNotFoundError(f"unknown namespace: {name}")
            has_rows = await conn.fetchval(
                f'SELECT EXISTS(SELECT 1 FROM "{PG_SCHEMA}".memory_chunks WHERE namespace = $1)',
                name,
            )
            if has_rows:
                raise NamespaceNotEmptyError(f"namespace still has rows: {name}")
            await conn.execute(f'DELETE FROM "{PG_SCHEMA}".namespaces WHERE name = $1', name)
