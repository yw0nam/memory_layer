"""Agent-owned profiles: each owner's `self` part and its `user` part, delivered at session start.

An owner is an agent's author slug. The owner replaces its `self` part whenever it chooses;
its `user` part changes only with the user's consent, recorded on the version, and only when
written against the current user version. Every version is kept. The server calls no model.
"""

from __future__ import annotations

import time
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.serve.common.http import iso

_VERSIONS = f'"{PG_SCHEMA}".agent_profiles'

LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended('profile:' || $1, 0))"
LATEST_SQL = f"""
SELECT version, content, created_at FROM {_VERSIONS}
WHERE owner = $1 AND part = $2
ORDER BY version DESC
LIMIT 1
"""
INSERT_VERSION_SQL = f"""
INSERT INTO {_VERSIONS}
  (owner, part, version, content, author, reason, consent_method, consented_at, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
RETURNING version
"""
VERSIONS_SQL = f"""
SELECT version, content, author, reason, consent_method, consented_at, created_at FROM {_VERSIONS}
WHERE owner = $1 AND part = $2
ORDER BY version DESC
LIMIT $3
"""


class Stale(Exception):
    """The submitted base no longer matches the owner's current user version."""

    def __init__(self, version: int) -> None:
        super().__init__("stale")
        self.version = version


def _now() -> float:
    return time.time()


def _version(row: Any) -> int:
    return 0 if row is None else row["version"]


def _served(row: Any) -> dict[str, Any] | None:
    if row is None or not row["content"]:
        return None
    return {"content": row["content"], "created_at": iso(row["created_at"])}


async def read(owner: str) -> dict[str, Any]:
    """The owner's latest parts, from one snapshot."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            own = await conn.fetchrow(LATEST_SQL, owner, "self")
            user = await conn.fetchrow(LATEST_SQL, owner, "user")
    return {
        "owner": owner,
        "self_version": _version(own),
        "self": _served(own),
        "user_version": _version(user),
        "user": _served(user),
    }


async def write_self(owner: str, content: str) -> dict[str, Any]:
    """Store the owner's next `self` version under its lock, unless it equals the latest."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="read_committed"):
            await conn.execute(LOCK_SQL, owner)
            latest = await conn.fetchrow(LATEST_SQL, owner, "self")
            if latest is not None and latest["content"] == content:
                return {"status": "unchanged", "version": latest["version"]}
            version = await conn.fetchval(
                INSERT_VERSION_SQL,
                owner,
                "self",
                _version(latest) + 1,
                content,
                owner,
                None,
                None,
                None,
                _now(),
            )
    return {"status": "written", "version": version}


async def write_user(
    owner: str, content: str, reason: str, base_version: int, consent_method: str, dry_run: bool
) -> dict[str, Any]:
    """Store the owner's next `user` version under its lock, written against its current one.

    A dry run checks and writes nothing; it returns the current content. Raises Stale.
    """
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="read_committed"):
            await conn.execute(LOCK_SQL, owner)
            latest = await conn.fetchrow(LATEST_SQL, owner, "user")
            current = _version(latest)
            if base_version != current:
                raise Stale(current)
            if latest is not None and latest["content"] == content:
                return {"status": "unchanged", "version": current}
            if dry_run:
                return {
                    "status": "valid",
                    "version": current,
                    "current": "" if latest is None else latest["content"],
                }
            now = _now()
            version = await conn.fetchval(
                INSERT_VERSION_SQL,
                owner,
                "user",
                current + 1,
                content,
                owner,
                reason,
                consent_method,
                now,
                now,
            )
    return {"status": "written", "version": version}


async def versions(owner: str, part: str, limit: int) -> list[dict[str, Any]]:
    """One part's versions newest first, with their authorship and consent record."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(VERSIONS_SQL, owner, part, limit)
    return [
        {
            "version": row["version"],
            "content": row["content"],
            "author": row["author"],
            "reason": row["reason"],
            "consent_method": row["consent_method"],
            "consented_at": iso(row["consented_at"]),
            "created_at": iso(row["created_at"]),
        }
        for row in rows
    ]
