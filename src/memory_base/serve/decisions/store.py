"""The decision lane: numbered, author-scoped decisions of one repository.

A decision is read by (scope, author, number) and by the lists below, never by
note search. Its number is the next one for that scope and author, taken under a
transaction advisory lock; a supersede replaces an active decision in the same
transaction, and a refused one stores nothing.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.serve.common.scope import normalize_scope

TITLE_MAX_CHARS = 200
BODY_MAX_CHARS = 1500
REASON_MAX_CHARS = 1500
MAX_LINKS = 10
LINK_MAX_CHARS = 500
LIST_MAX_LIMIT = 200
STATUSES = ("active", "all")

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TABLE = f'"{PG_SCHEMA}".decisions'


class DecisionNotFound(Exception):
    """No decision with that number exists for the caller in that scope."""


def repo_scope(scope: Any) -> str:
    """The normalized scope, which must be a repository scope."""
    scope = normalize_scope(scope)
    if not scope.startswith("repo:"):
        raise ValueError("scope must be repo:<origin>")
    return scope


def _number(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > limit:
        raise ValueError(f"{name} must be at most {limit} chars")
    return value


def _validated(title: Any, decision: Any, why: Any, links: Any, decided_on: Any) -> tuple:
    title = _text(title, "title", TITLE_MAX_CHARS)
    decision = _text(decision, "decision", BODY_MAX_CHARS)
    if why is not None:
        why = _text(why, "why", BODY_MAX_CHARS)
    if len(decision) + len(why or "") > BODY_MAX_CHARS:
        raise ValueError(f"decision and why together must be at most {BODY_MAX_CHARS} chars")
    links = [] if links is None else links
    if (
        not isinstance(links, list)
        or len(links) > MAX_LINKS
        or any(not isinstance(link, str) or len(link) > LINK_MAX_CHARS for link in links)
    ):
        raise ValueError(
            f"links must be at most {MAX_LINKS} strings of at most {LINK_MAX_CHARS} chars"
        )
    if decided_on is None:
        day = datetime.now(timezone.utc).date()
    else:
        try:
            if not _DATE_RE.fullmatch(decided_on):
                raise ValueError
            day = date.fromisoformat(decided_on)
        except (TypeError, ValueError):
            raise ValueError("decided_on must be an ISO date (YYYY-MM-DD)") from None
    return title, decision, why, links, day


def _refused(number: int) -> ValueError:
    return ValueError(f"D-{number} is not an active decision of this scope and author")


async def record_decision(
    *,
    author: str,
    scope: str,
    title: Any,
    decision: Any,
    why: Any = None,
    links: Any = None,
    decided_on: Any = None,
    supersedes: Any = None,
) -> dict[str, Any]:
    """Store one decision under the next number; `superseded` is the replaced number or None."""
    scope = repo_scope(scope)
    title, decision, why, links, day = _validated(title, decision, why, links, decided_on)
    if supersedes is not None:
        supersedes = _number(supersedes, "supersedes")
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            # ponytail: one hashed lock per (scope, author); a collision only over-serializes.
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", f"{scope}\x1f{author}"
            )
            number = await conn.fetchval(
                f"SELECT coalesce(max(number), 0) + 1 FROM {_TABLE} WHERE scope = $1 AND author = $2",
                scope,
                author,
            )
            if supersedes is not None:
                replaced = await conn.fetchval(
                    f"""
                    UPDATE {_TABLE} SET status = 'superseded', superseded_by = $4
                    WHERE scope = $1 AND author = $2 AND number = $3 AND status = 'active'
                    RETURNING number
                    """,
                    scope,
                    author,
                    supersedes,
                    number,
                )
                if replaced is None:
                    raise _refused(supersedes)
            await conn.execute(
                f"""
                INSERT INTO {_TABLE}
                  (scope, author, number, title, decision, why, links, decided_on, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, 'active', clock_timestamp())
                """,
                scope,
                author,
                number,
                title,
                decision,
                why,
                json.dumps(links),
                day,
            )
    return {"number": number, "superseded": supersedes}


async def retire_decision(*, author: str, scope: str, number: Any, reason: Any) -> dict[str, Any]:
    """Retire an active decision with a reason; any other state is refused."""
    scope = repo_scope(scope)
    number = _number(number, "number")
    reason = _text(reason, "reason", REASON_MAX_CHARS)
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        retired = await conn.fetchval(
            f"""
            UPDATE {_TABLE} SET status = 'retired', retired_reason = $4
            WHERE scope = $1 AND author = $2 AND number = $3 AND status = 'active'
            RETURNING number
            """,
            scope,
            author,
            number,
            reason,
        )
    if retired is None:
        raise _refused(number)
    return {"number": number, "status": "retired"}


async def list_decisions(
    *, author: str, scope: str, status: str = "active", limit: int = LIST_MAX_LIMIT
) -> list[dict[str, Any]]:
    """Header rows, newest number first; `status` is "active" or "all"."""
    scope = repo_scope(scope)
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= LIST_MAX_LIMIT:
        raise ValueError(f"limit must be an integer between 1 and {LIST_MAX_LIMIT}")
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f"""
            SELECT number, title, decided_on, status FROM {_TABLE}
            WHERE scope = $1 AND author = $2 AND ($3 = 'all' OR status = 'active')
            ORDER BY number DESC LIMIT $4
            """,
            scope,
            author,
            status,
            limit,
        )
    return [{**row, "decided_on": row["decided_on"].isoformat()} for row in rows]


async def active_decisions(
    *, author: str, scope: str, limit: int = LIST_MAX_LIMIT
) -> list[dict[str, Any]]:
    """The newest active decisions in full, for recall; newest number first."""
    scope = repo_scope(scope)
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f"""
            SELECT number, title, decision, why, links, decided_on FROM {_TABLE}
            WHERE scope = $1 AND author = $2 AND status = 'active'
            ORDER BY number DESC LIMIT $3
            """,
            scope,
            author,
            limit,
        )
    return [
        {**row, "links": json.loads(row["links"]), "decided_on": row["decided_on"].isoformat()}
        for row in rows
    ]


async def get_decision(*, author: str, scope: str, number: Any) -> dict[str, Any]:
    """One decision in full with its lineage: the number it replaced and the one that replaced it."""
    scope = repo_scope(scope)
    number = _number(number, "number")
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        row = await conn.fetchrow(
            f"""
            SELECT d.number, d.title, d.decision, d.why, d.links, d.decided_on, d.status,
                   old.number AS supersedes, d.superseded_by, d.retired_reason, d.created_at
            FROM {_TABLE} d
            LEFT JOIN {_TABLE} old
              ON old.scope = d.scope AND old.author = d.author AND old.superseded_by = d.number
            WHERE d.scope = $1 AND d.author = $2 AND d.number = $3
            """,
            scope,
            author,
            number,
        )
    if row is None:
        raise DecisionNotFound(f"no decision D-{number} in this scope")
    return {
        **row,
        "links": json.loads(row["links"]),
        "decided_on": row["decided_on"].isoformat(),
        "created_at": row["created_at"].isoformat(),
    }
