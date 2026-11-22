"""The session record: SQL for storing, listing, reading, and searching Claude Code events.

An event is stored as the client delivered it, never embedded and never ranked. The rows
of one (author, session) are numbered under an advisory lock, so a replayed batch is
recognised and concurrent batches get distinct seqs. Every read is scoped to the author.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once

ROLES = ("user", "assistant", "tool_call", "tool_result")
MAX_EVENTS = 500
MAX_BODY_BYTES = 2_000_000
MAX_CONTENT_CHARS = 64_000
MIN_QUERY_CHARS = 2
TITLE_CHARS = 120
WINDOW_CHARS = 300
LIST_MAX_LIMIT = 100
READ_MAX_LIMIT = 200
READ_MAX_CHARS = 200_000
SEARCH_MAX_LIMIT = 50

_TABLE = f'"{PG_SCHEMA}".session_events'
LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))"


def _lock_key(author: str, session_id: str) -> str:
    return f"{author}\x1f{session_id}"


_PAYLOAD_FIELDS = (
    "session_id", "role", "content", "tool_name", "tool_use_id", "event_at", "repo", "host", "cwd"
)  # fmt: skip


class NotFound(ValueError):
    """The caller has no session with that id."""


def _like(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _utc(ts: float | None) -> datetime | None:
    return None if ts is None else datetime.fromtimestamp(ts, tz=timezone.utc)


async def store_events(author: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Store a batch under the next seq of each session; a stored id is `known` or a conflict."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            sessions = sorted({event["session_id"] for event in events})
            # Locks are taken in sorted order, so two batches over shared sessions cannot deadlock.
            for session_id in sessions:
                await conn.execute(LOCK_SQL, _lock_key(author, session_id))
            stored = await conn.fetch(
                f"SELECT id, seq, {', '.join(_PAYLOAD_FIELDS)} FROM {_TABLE} "
                "WHERE author = $1 AND id = ANY($2::uuid[])",
                author,
                [str(event["id"]) for event in events],
            )
            seen = {str(row["id"]): dict(row) for row in stored}
            last = await conn.fetch(
                f"SELECT session_id, max(seq) AS last FROM {_TABLE} "
                "WHERE author = $1 AND session_id = ANY($2::text[]) GROUP BY session_id",
                author,
                sessions,
            )
            seq = {row["session_id"]: row["last"] for row in last}
            rows, known, conflicts = [], 0, []
            for event in events:
                key = str(event["id"])
                if (old := seen.get(key)) is not None:
                    if all(old[field] == event[field] for field in _PAYLOAD_FIELDS):
                        known += 1
                    else:
                        conflicts.append(key)
                    continue
                seq[event["session_id"]] = seq.get(event["session_id"], 0) + 1
                seen[key] = event
                rows.append(
                    (key, author, event["session_id"], seq[event["session_id"]])
                    + tuple(event[field] for field in _PAYLOAD_FIELDS[1:])
                )
            # A concurrent first write of the same id is skipped instead of failing the batch.
            await conn.executemany(
                f"""
                INSERT INTO {_TABLE}
                  (id, author, session_id, seq, role, content, tool_name,
                   tool_use_id, event_at, repo, host, cwd)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT DO NOTHING
                """,
                rows,
            )
    return {"stored": len(rows), "known": known, "conflicts": conflicts}


async def list_sessions(
    *, author: str, repo: str | None, since: float | None, until: float | None, limit: int
) -> list[dict[str, Any]]:
    """The author's sessions newest first by their last event, each titled by its first user
    event that does not open with a tag, or its first user event when every one does."""
    args: list[Any] = [author]
    having = ["TRUE"]
    for expr, op, value in (
        ("min(t.repo)", "=", repo),
        ("max(t.event_at)", ">=", _utc(since)),
        ("min(t.event_at)", "<", _utc(until)),
    ):
        if value is not None:
            args.append(value)
            having.append(f"{expr} {op} ${len(args)}")
    args.append(limit)
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f"""
            SELECT t.session_id, min(t.repo) AS repo, min(t.host) AS host, min(t.cwd) AS cwd,
                   min(t.event_at) AS started_at, max(t.event_at) AS last_at,
                   count(*) AS events,
                   (SELECT left(btrim(regexp_replace(u.content, '\\s+', ' ', 'g')), {TITLE_CHARS})
                    FROM {_TABLE} u
                    WHERE u.author = t.author AND u.session_id = t.session_id AND u.role = 'user'
                    ORDER BY u.content ~ '^\\s*<[A-Za-z][A-Za-z0-9_-]*>', u.seq
                    LIMIT 1) AS title
            FROM {_TABLE} t
            WHERE t.author = $1
            GROUP BY t.author, t.session_id
            HAVING {" AND ".join(having)}
            ORDER BY last_at DESC, t.session_id
            LIMIT ${len(args)}
            """,
            *args,
        )
    return [
        {
            **row,
            "started_at": row["started_at"].isoformat(),
            "last_at": row["last_at"].isoformat(),
            "title": row["title"] or "",
        }
        for row in map(dict, rows)
    ]


async def read_session(
    *,
    author: str,
    session_id: str,
    from_seq: int,
    limit: int,
    role: str | None,
    contains: str | None,
) -> dict[str, Any]:
    """The session's events in seq order from `from_seq`, or only the matches of `contains`
    with one neighbour on each side.

    A page holds at most `limit` events and READ_MAX_CHARS content characters (always at
    least one event); `next_from_seq` is the seq of the first event left out, or None.
    """
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        if not await conn.fetchval(
            f"SELECT 1 FROM {_TABLE} WHERE author = $1 AND session_id = $2 LIMIT 1",
            author,
            session_id,
        ):
            raise NotFound(f"no session {session_id!r} of yours")
        columns = "seq, role, tool_name, event_at, content"
        if contains is None:
            rows = await conn.fetch(
                f"""
                SELECT {columns} FROM {_TABLE}
                WHERE author = $1 AND session_id = $2 AND seq >= $3
                  AND ($5::text IS NULL OR role = $5)
                ORDER BY seq LIMIT $4
                """,
                author,
                session_id,
                from_seq,
                limit + 1,
                role,
            )
        else:
            # A match is a content hit of the asked role; its neighbours come back in any role.
            def hit(alias: str) -> str:
                return (
                    f"({alias}.content ILIKE $6 ESCAPE '\\' "
                    f"AND ($5::text IS NULL OR {alias}.role = $5))"
                )

            rows = await conn.fetch(
                f"""
                SELECT {columns}, {hit("t")} AS match
                FROM {_TABLE} t
                WHERE t.author = $1 AND t.session_id = $2 AND t.seq >= $3
                  AND EXISTS (
                    SELECT 1 FROM {_TABLE} m
                    WHERE m.author = $1 AND m.session_id = $2
                      AND m.seq BETWEEN t.seq - 1 AND t.seq + 1 AND {hit("m")})
                ORDER BY t.seq LIMIT $4
                """,
                author,
                session_id,
                from_seq,
                limit + 1,
                role,
                _like(contains),
            )
    events, chars = [], 0
    for row in rows[:limit]:
        chars += len(row["content"])
        if events and chars > READ_MAX_CHARS:
            break
        events.append({**row, "event_at": row["event_at"].isoformat()})
    more = rows[len(events)] if len(rows) > len(events) else None
    return {
        "session_id": session_id,
        "events": events,
        "next_from_seq": more["seq"] if more is not None else None,
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
    *, author: str, q: str, repo: str | None, role: str | None, limit: int
) -> list[dict[str, Any]]:
    """The author's events whose content holds `q` as a case-insensitive substring, newest first."""
    args: list[Any] = [author, _like(q)]
    where = ["author = $1", "content ILIKE $2 ESCAPE '\\'"]
    for column, value in (("repo", repo), ("role", role)):
        if value is not None:
            args.append(value)
            where.append(f"{column} = ${len(args)}")
    args.append(limit)
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f"""
            SELECT session_id, seq, role, event_at, repo, content FROM {_TABLE}
            WHERE {" AND ".join(where)}
            ORDER BY event_at DESC, session_id, seq
            LIMIT ${len(args)}
            """,
            *args,
        )
    return [
        {
            "session_id": row["session_id"],
            "seq": row["seq"],
            "role": row["role"],
            "event_at": row["event_at"].isoformat(),
            "repo": row["repo"],
            "window": _window(row["content"], q),
        }
        for row in rows
    ]


async def delete_session(*, author: str, session_id: str) -> int:
    """Hard-delete the author's rows of a session under its lock; returns the count."""
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await conn.execute(LOCK_SQL, _lock_key(author, session_id))
            status = await conn.execute(
                f"DELETE FROM {_TABLE} WHERE author = $1 AND session_id = $2", author, session_id
            )
    return int(status.split()[-1])
