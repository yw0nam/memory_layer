"""The verbatim turn lane: SQL for appending, listing, reading, and searching turns.

A turn is stored as the client delivered it, never embedded and never ranked. Rows of one
session are numbered under a per-session advisory lock, so a replayed append is detected
and concurrent appends get distinct seqs.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.core.secrets import find_secret
from memory_base.serve.access import namespaces as namespace_registry

ROLES = ("user", "assistant")
MAX_TEXT_CHARS = 64_000
MIN_QUERY_CHARS = 2
TITLE_CHARS = 120
WINDOW_CHARS = 300
MAX_RESPONSE_CHARS = 40_000
DEFAULT_LIMIT = 50
MAX_LIMIT = 200

_TURNS = f'"{PG_SCHEMA}".turns'
LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))"
_PAYLOAD_FIELDS = ("namespace", "origin", "session_id", "role", "content", "reasoning", "repo")


class Conflict(ValueError):
    """An id already holds a different turn."""


class NotFound(ValueError):
    """No session the key can read has that origin and id."""


def _scope(namespaces: list[str] | None, args: list[Any], column: str = "namespace") -> str:
    """A namespace predicate; None means unfiltered."""
    if namespaces is None:
        return "TRUE"
    args.append(namespaces)
    return f"{column} = ANY(${len(args)}::text[])"


def _like(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _utc(ts: float | None) -> datetime | None:
    return None if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc)


def _require_query(text: str) -> None:
    if len(text) < MIN_QUERY_CHARS:
        raise ValueError(f"the query must hold at least {MIN_QUERY_CHARS} characters")


async def append(row: dict[str, Any]) -> dict[str, Any]:
    """Store a turn under the next seq of its session; a replay returns the stored one."""
    for field in ("content", "reasoning"):
        secret = find_secret(row[field]) if row.get(field) else None
        if secret is not None:
            raise ValueError(
                f"refused: {field} looks like a credential ({secret}); keep it out of the turn"
            )
    turn_id = uuid.UUID(row["id"])
    lock_key = f"turns:{row['origin']}:{row['session_id']}"
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await conn.execute(LOCK_SQL, lock_key)
            stored = await conn.fetchrow(
                f"SELECT id, namespace, origin, session_id, role, content, reasoning, repo, seq "
                f"FROM {_TURNS} WHERE id = $1",
                turn_id,
            )
            if stored is not None:
                if all(stored[field] == row.get(field) for field in _PAYLOAD_FIELDS):
                    return {"id": str(stored["id"]), "seq": stored["seq"], "stored": False}
                raise Conflict("id already holds a different turn")
            owner = await conn.fetchval(
                f"SELECT namespace FROM {_TURNS} WHERE origin = $1 AND session_id = $2 LIMIT 1",
                row["origin"],
                row["session_id"],
            )
            if owner is not None and owner != row["namespace"]:
                raise ValueError(f"session {row['session_id']!r} belongs to namespace {owner!r}")
            await namespace_registry.require_registered(conn, row["namespace"])
            try:
                inserted = await conn.fetchrow(
                    f"""
                    INSERT INTO {_TURNS}
                      (id, namespace, origin, session_id, seq, role, content, reasoning, repo,
                       created_at)
                    VALUES ($1, $2, $3, $4,
                      (SELECT COALESCE(MAX(seq), 0) + 1 FROM {_TURNS}
                       WHERE origin = $3 AND session_id = $4),
                      $5, $6, $7, $8, now())
                    RETURNING id, seq, created_at
                    """,
                    turn_id,
                    row["namespace"],
                    row["origin"],
                    row["session_id"],
                    row["role"],
                    row["content"],
                    row["reasoning"],
                    row["repo"],
                )
            except asyncpg.UniqueViolationError:
                raise Conflict("id already holds a different turn") from None
    return {"id": str(inserted["id"]), "seq": inserted["seq"], "stored": True}


async def list_sessions(
    namespaces: list[str] | None,
    since: float | None,
    until: float | None,
    origin: str | None,
    repo: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Sessions newest first by their last turn, each titled by its first user message."""
    args: list[Any] = []
    where = [_scope(namespaces, args, "t.namespace")]
    having = ["TRUE"]
    if origin is not None:
        args.append(origin)
        where.append(f"t.origin = ${len(args)}")
    if repo is not None:
        args.append(repo)
        having.append(f"MIN(t.repo) = ${len(args)}")
    if since is not None:
        args.append(_utc(since))
        having.append(f"MAX(t.created_at) >= ${len(args)}")
    if until is not None:
        args.append(_utc(until))
        having.append(f"MIN(t.created_at) < ${len(args)}")
    args.append(limit)
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f"""
            SELECT t.origin, t.session_id, t.namespace, MIN(t.repo) AS repo,
                   MIN(t.created_at) AS started_at, MAX(t.created_at) AS last_at,
                   COUNT(*) AS turns,
                   (SELECT left(btrim(regexp_replace(u.content, '\\s+', ' ', 'g')), {TITLE_CHARS})
                    FROM {_TURNS} u
                    WHERE u.origin = t.origin AND u.session_id = t.session_id
                      AND u.role = 'user'
                    ORDER BY u.seq LIMIT 1) AS title
            FROM {_TURNS} t
            WHERE {" AND ".join(where)}
            GROUP BY t.origin, t.session_id, t.namespace
            HAVING {" AND ".join(having)}
            ORDER BY last_at DESC, t.session_id
            LIMIT ${len(args)}
            """,
            *args,
        )
    return [
        {
            **dict(row),
            "started_at": row["started_at"].isoformat(),
            "last_at": row["last_at"].isoformat(),
            "title": row["title"] or "",
        }
        for row in rows
    ]


def _cut(rows: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], int | None]:
    """The rows within the row and character caps, and the seq of the first row left out."""
    kept: list[dict[str, Any]] = []
    total = 0
    for row in rows[:limit]:
        size = len(row["content"]) + len(row["reasoning"] or "")
        if kept and total + size > MAX_RESPONSE_CHARS:
            break
        kept.append(row)
        total += size
    return kept, rows[len(kept)]["seq"] if len(kept) < len(rows) else None


async def read_session(
    namespaces: list[str] | None,
    origin: str,
    session_id: str,
    contains: str | None,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    """A session's turns in seq order from `offset`, or only the matches of `contains` with
    one neighbour on each side."""
    if contains is not None:
        _require_query(contains)
    scope_args: list[Any] = [origin, session_id]
    scope = _scope(namespaces, scope_args)
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        session = await conn.fetchrow(
            f"""
            SELECT namespace, MIN(repo) AS repo FROM {_TURNS}
            WHERE origin = $1 AND session_id = $2 AND {scope} GROUP BY namespace
            """,
            *scope_args,
        )
        if session is None:
            raise NotFound(f"no session {session_id!r} from {origin!r} is readable by this key")
        args = [origin, session_id, offset, limit + 1]
        if contains is None:
            rows = await conn.fetch(
                f"""
                SELECT seq, role, content, reasoning, created_at, FALSE AS match
                FROM {_TURNS}
                WHERE origin = $1 AND session_id = $2 AND seq >= $3
                ORDER BY seq LIMIT $4
                """,
                *args,
            )
        else:
            rows = await conn.fetch(
                f"""
                SELECT seq, role, content, reasoning, created_at,
                       content ILIKE $5 ESCAPE '\\' AS match
                FROM {_TURNS} t
                WHERE origin = $1 AND session_id = $2 AND seq >= $3
                  AND EXISTS (
                    SELECT 1 FROM {_TURNS} m
                    WHERE m.origin = $1 AND m.session_id = $2
                      AND m.seq BETWEEN t.seq - 1 AND t.seq + 1
                      AND m.content ILIKE $5 ESCAPE '\\')
                ORDER BY seq LIMIT $4
                """,
                *args,
                _like(contains),
            )
    kept, next_offset = _cut([dict(row) for row in rows], limit)
    turns = [
        {
            "seq": row["seq"],
            "role": row["role"],
            "content": row["content"],
            "reasoning": row["reasoning"],
            "created_at": row["created_at"].isoformat(),
            **({"match": row["match"]} if contains is not None else {}),
        }
        for row in kept
    ]
    return {
        "origin": origin,
        "session_id": session_id,
        "namespace": session["namespace"],
        "repo": session["repo"],
        "turns": turns,
        "next_offset": next_offset,
    }


def _window(content: str, query: str) -> str:
    """WINDOW_CHARS of content centred on the first case-insensitive match."""
    folded = content.casefold()
    at = folded.find(query.casefold()) if len(folded) == len(content) else -1
    if at < 0:
        return content[:WINDOW_CHARS]
    start = max(0, min(at + len(query) // 2 - WINDOW_CHARS // 2, len(content) - WINDOW_CHARS))
    return content[start : start + WINDOW_CHARS]


async def search(
    namespaces: list[str] | None,
    q: str,
    since: float | None,
    until: float | None,
    origin: str | None,
    repo: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Turns whose content holds `q` as a case-insensitive substring, newest first."""
    _require_query(q)
    args: list[Any] = [_like(q)]
    where = ["content ILIKE $1 ESCAPE '\\'", _scope(namespaces, args)]
    for column, value, op in (
        ("origin", origin, "="),
        ("repo", repo, "="),
        ("created_at", _utc(since), ">="),
        ("created_at", _utc(until), "<"),
    ):
        if value is not None:
            args.append(value)
            where.append(f"{column} {op} ${len(args)}")
    args.append(min(limit, MAX_LIMIT))
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f"""
            SELECT origin, session_id, namespace, seq, role, created_at, content
            FROM {_TURNS}
            WHERE {" AND ".join(where)}
            ORDER BY created_at DESC, id
            LIMIT ${len(args)}
            """,
            *args,
        )
    return [
        {
            "origin": row["origin"],
            "session_id": row["session_id"],
            "namespace": row["namespace"],
            "seq": row["seq"],
            "role": row["role"],
            "created_at": row["created_at"].isoformat(),
            "window": _window(row["content"], q),
        }
        for row in rows
    ]


async def delete_session(origin: str, session_id: str) -> int:
    """Hard-delete every row of a session; returns the count."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        status = await conn.execute(
            f"DELETE FROM {_TURNS} WHERE origin = $1 AND session_id = $2", origin, session_id
        )
    return int(status.split()[-1])
