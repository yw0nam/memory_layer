"""Agent-owned profiles: each owner's `self` part and its `user` part, delivered at session start.

An owner is an agent's author slug. The owner replaces its `self` part whenever it chooses;
its `user` part changes only when a proposal written against the current user version is
approved. Every version and every proposal is kept. The server calls no model.
"""

from __future__ import annotations

import time
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.serve.common.http import iso

USER_AUTHOR = "user"


_VERSIONS = f'"{PG_SCHEMA}".agent_profiles'
_PROPOSALS = f'"{PG_SCHEMA}".profile_proposals'
_PROPOSAL_COLUMNS = (
    "id, owner, status, base_version, reason, content, created_at, decided_at, decision_note"
)

LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended('profile:' || $1, 0))"
LATEST_SQL = f"""
SELECT version, content, created_at FROM {_VERSIONS}
WHERE owner = $1 AND part = $2
ORDER BY version DESC
LIMIT 1
"""
INSERT_VERSION_SQL = f"""
INSERT INTO {_VERSIONS} (owner, part, version, content, author, proposal_id, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7)
RETURNING version
"""
VERSIONS_SQL = f"""
SELECT version, content, author, proposal_id, created_at FROM {_VERSIONS}
WHERE owner = $1 AND part = $2
ORDER BY version DESC
LIMIT $3
"""
PROPOSAL_OWNER_SQL = f"SELECT owner FROM {_PROPOSALS} WHERE id = $1"
PROPOSAL_SQL = f"SELECT {_PROPOSAL_COLUMNS} FROM {_PROPOSALS} WHERE id = $1"
PENDING_SQL = f"""
SELECT id, created_at, reason, base_version FROM {_PROPOSALS}
WHERE owner = $1 AND status = 'pending'
"""
SUPERSEDE_SQL = f"""
UPDATE {_PROPOSALS} SET status = 'superseded', decided_at = $2
WHERE owner = $1 AND status = 'pending'
RETURNING id
"""
INSERT_PROPOSAL_SQL = f"""
INSERT INTO {_PROPOSALS} (owner, content, reason, base_version, status, created_at)
VALUES ($1, $2, $3, $4, 'pending', $5)
RETURNING id
"""
DECIDE_SQL = f"""
UPDATE {_PROPOSALS} SET status = $2, decided_at = $3, decision_note = $4
WHERE id = $1
"""
PROPOSALS_SQL = f"""
SELECT {_PROPOSAL_COLUMNS} FROM {_PROPOSALS}
WHERE ($1::text IS NULL OR owner = $1) AND ($2::text IS NULL OR status = $2)
ORDER BY created_at DESC, id DESC
LIMIT $3
"""


class Stale(Exception):
    """The submitted base no longer matches the owner's current user version."""

    def __init__(self, version: int) -> None:
        super().__init__("stale")
        self.version = version


class NotPending(Exception):
    """The proposal was already approved, rejected, or superseded."""

    def __init__(self, status: str) -> None:
        super().__init__("not_pending")
        self.status = status


class NotFound(LookupError):
    """No proposal has this id."""


def _now() -> float:
    return time.time()


def _version(row: Any) -> int:
    return 0 if row is None else row["version"]


def _served(row: Any) -> dict[str, Any] | None:
    if row is None or not row["content"]:
        return None
    return {"content": row["content"], "created_at": iso(row["created_at"])}


def _proposal(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "owner": row["owner"],
        "status": row["status"],
        "base_version": row["base_version"],
        "reason": row["reason"],
        "content": row["content"],
        "created_at": iso(row["created_at"]),
        "decided_at": iso(row["decided_at"]),
        "decision_note": row["decision_note"],
    }


async def read(owner: str) -> dict[str, Any]:
    """The owner's latest parts and pending proposal, from one snapshot."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            own = await conn.fetchrow(LATEST_SQL, owner, "self")
            user = await conn.fetchrow(LATEST_SQL, owner, "user")
            pending = await conn.fetchrow(PENDING_SQL, owner)
    if pending is not None:
        pending = {
            "id": pending["id"],
            "created_at": iso(pending["created_at"]),
            "reason": pending["reason"],
            "base_version": pending["base_version"],
        }
    return {
        "owner": owner,
        "self_version": _version(own),
        "self": _served(own),
        "user_version": _version(user),
        "user": _served(user),
        "pending_proposal": pending,
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
                _now(),
            )
    return {"status": "written", "version": version}


async def propose(owner: str, content: str, reason: str, base_version: int) -> dict[str, Any]:
    """Store a pending proposal written against the current user version.

    The earlier pending proposal is superseded in the same transaction. Raises Stale.
    """
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="read_committed"):
            await conn.execute(LOCK_SQL, owner)
            current = _version(await conn.fetchrow(LATEST_SQL, owner, "user"))
            if base_version != current:
                raise Stale(current)
            now = _now()
            superseded = await conn.fetchval(SUPERSEDE_SQL, owner, now)
            proposal_id = await conn.fetchval(
                INSERT_PROPOSAL_SQL, owner, content, reason, base_version, now
            )
    return {"id": proposal_id, "status": "pending", "superseded": superseded}


async def _decide(proposal_id: int, status: str, note: str | None) -> dict[str, Any]:
    """Record a decision; the proposal's status and base are read only under the owner lock."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="read_committed"):
            owner = await conn.fetchval(PROPOSAL_OWNER_SQL, proposal_id)
            if owner is None:
                raise NotFound(f"unknown proposal: {proposal_id}")
            await conn.execute(LOCK_SQL, owner)
            row = await conn.fetchrow(PROPOSAL_SQL, proposal_id)
            if row["status"] != "pending":
                raise NotPending(row["status"])
            now = _now()
            result: dict[str, Any] = {"status": status}
            if status == "approved":
                current = _version(await conn.fetchrow(LATEST_SQL, owner, "user"))
                if row["base_version"] != current:
                    raise Stale(current)
                result["version"] = await conn.fetchval(
                    INSERT_VERSION_SQL,
                    owner,
                    "user",
                    current + 1,
                    row["content"],
                    USER_AUTHOR,
                    proposal_id,
                    now,
                )
            await conn.execute(DECIDE_SQL, proposal_id, status, now, note)
    return result


async def approve(proposal_id: int, note: str | None) -> dict[str, Any]:
    """Store the proposal as the next user version. Raises NotFound, NotPending, or Stale."""
    return await _decide(proposal_id, "approved", note)


async def reject(proposal_id: int, note: str | None) -> dict[str, Any]:
    """Mark the proposal rejected. Raises NotFound or NotPending."""
    return await _decide(proposal_id, "rejected", note)


async def proposal(proposal_id: int) -> dict[str, Any]:
    """One proposal with the owner's current user version and content, from one snapshot."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            row = await conn.fetchrow(PROPOSAL_SQL, proposal_id)
            if row is None:
                raise NotFound(f"unknown proposal: {proposal_id}")
            latest = await conn.fetchrow(LATEST_SQL, row["owner"], "user")
    current = _served(latest)
    return {
        **_proposal(row),
        "current_user_version": _version(latest),
        "current_user_content": None if current is None else current["content"],
    }


async def list_proposals(owner: str | None, status: str | None, limit: int) -> list[dict[str, Any]]:
    """Proposals newest first, of one owner or of every owner."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(PROPOSALS_SQL, owner, status, limit)
    return [_proposal(row) for row in rows]


async def versions(owner: str, part: str, limit: int) -> list[dict[str, Any]]:
    """One part's versions newest first, with their authorship."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(VERSIONS_SQL, owner, part, limit)
    return [
        {
            "version": row["version"],
            "content": row["content"],
            "author": row["author"],
            "proposal_id": row["proposal_id"],
            "created_at": iso(row["created_at"]),
        }
        for row in rows
    ]
