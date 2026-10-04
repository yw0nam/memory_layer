"""Agent-owned profiles: each owner's `self` part and its `user` part, delivered at session start.

An owner is an agent's author slug. The owner replaces its `self` part whenever it chooses;
its `user` part changes only when a key carrying the `user` author approves a proposal
written against the current user version. Authority comes only from the key's authors.
Every version and every proposal is kept. The server calls no model.
"""

from __future__ import annotations

import re
import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.core.secrets import find_secret
from memory_base.serve.common.http import error, iso, json_body
from memory_base.serve.keys import AUTHOR_RE

PARTS = ("self", "user")
STATUSES = ("pending", "approved", "rejected", "superseded")
USER_AUTHOR = "user"
RESERVED_OWNERS = frozenset({USER_AUTHOR, "consolidator"})
DEFAULT_SELF_MAX_CHARS = 4000
DEFAULT_USER_MAX_CHARS = 3000
MIN_MAX_CHARS = 200
MAX_MAX_CHARS = 20000
MAX_REASON_CHARS = 1000
MAX_NOTE_CHARS = 1000
MAX_BASE_VERSION = 2_147_483_647
MAX_PROPOSAL_ID = 9_223_372_036_854_775_807
DEFAULT_LIMIT = 20
MAX_LIMIT = 200

_VERSIONS = f'"{PG_SCHEMA}".agent_profiles'
_PROPOSALS = f'"{PG_SCHEMA}".profile_proposals'
_PROPOSAL_COLUMNS = (
    "id, owner, status, base_version, reason, content, created_at, decided_at, decision_note"
)
_POSITIVE = re.compile(r"[1-9][0-9]*")
_DIGITS = re.compile(r"[0-9]+")

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


class RequestError(ValueError):
    """A request breaks the schema; REST maps it to 400."""


class Refused(ValueError):
    """A text fails a content check; nothing is stored. REST maps it to 400."""

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details


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


# ---- request validation -----------------------------------------------------


def _owner(value: Any) -> str:
    if not isinstance(value, str) or not AUTHOR_RE.fullmatch(value) or value in RESERVED_OWNERS:
        raise RequestError(
            "owner must be an author slug matching ^[a-z0-9][a-z0-9-]{0,39}$ "
            "other than 'user' or 'consolidator'"
        )
    return value


def _fields(body: Any, required: set[str], optional: set[str]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise RequestError("JSON body must be an object")
    unknown = sorted(set(body) - required - optional)
    if unknown:
        raise RequestError(f"unknown field: {', '.join(unknown)}")
    missing = sorted(required - set(body))
    if missing:
        raise RequestError(f"missing field: {', '.join(missing)}")
    return body


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise RequestError(f"{name} must be an integer between {low} and {high}")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RequestError(f"{name} must be a string")
    return value.strip()


def _no_secret(text: str, name: str) -> None:
    secret = find_secret(text)
    if secret is not None:
        raise Refused(f"{name} contains a credential ({secret})")


def _content(value: Any, max_chars: int) -> str:
    content = _string(value, "content")
    if len(content) > max_chars:
        raise Refused(
            f"content is {len(content)} characters, over max_chars {max_chars}",
            chars=len(content),
            max_chars=max_chars,
        )
    _no_secret(content, "content")
    return content


def _bounded_text(value: Any, name: str, low: int, high: int) -> str:
    text = _string(value, name)
    if not low <= len(text) <= high:
        raise RequestError(f"{name} must hold {low} to {high} characters after stripping")
    _no_secret(text, name)
    return text


def _max_chars(body: dict[str, Any], default: int) -> int:
    return _integer(body.get("max_chars", default), "max_chars", MIN_MAX_CHARS, MAX_MAX_CHARS)


def parse_self(body: Any) -> tuple[str, str]:
    """`{owner, content, max_chars?}` → (owner, stripped content)."""
    fields = _fields(body, {"owner", "content"}, {"max_chars"})
    owner = _owner(fields["owner"])
    return owner, _content(fields["content"], _max_chars(fields, DEFAULT_SELF_MAX_CHARS))


def parse_proposal(body: Any) -> tuple[str, str, str, int]:
    """`{owner, content, reason, base_version, max_chars?}` → (owner, content, reason, base)."""
    fields = _fields(body, {"owner", "content", "reason", "base_version"}, {"max_chars"})
    owner = _owner(fields["owner"])
    base_version = _integer(fields["base_version"], "base_version", 0, MAX_BASE_VERSION)
    content = _content(fields["content"], _max_chars(fields, DEFAULT_USER_MAX_CHARS))
    reason = _bounded_text(fields["reason"], "reason", 1, MAX_REASON_CHARS)
    return owner, content, reason, base_version


def parse_decision(body: Any) -> str | None:
    """`{note?}` → the stripped note, or None when omitted."""
    fields = _fields(body, set(), {"note"})
    if "note" not in fields:
        return None
    return _bounded_text(fields["note"], "note", 0, MAX_NOTE_CHARS)


def parse_proposal_id(text: str) -> int:
    if not _POSITIVE.fullmatch(text) or int(text) > MAX_PROPOSAL_ID:
        raise RequestError("proposal id must be a positive 64-bit integer")
    return int(text)


def _query(request: Request, allowed: tuple[str, ...]) -> dict[str, str]:
    """The query parameters, each known and given at most once."""
    names = [name for name, _ in request.query_params.multi_items()]
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise RequestError(f"unknown query parameter: {', '.join(unknown)}")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise RequestError(f"query parameter given more than once: {', '.join(repeated)}")
    return dict(request.query_params)


def _limit(value: str | None) -> int:
    if value is None:
        return DEFAULT_LIMIT
    if not _DIGITS.fullmatch(value) or not 1 <= int(value) <= MAX_LIMIT:
        raise RequestError(f"limit must be an integer between 1 and {MAX_LIMIT}")
    return int(value)


async def _body(request: Request) -> dict[str, Any]:
    try:
        return await json_body(request)
    except Exception as exc:
        raise RequestError(f"invalid JSON body: {exc}") from None


# ---- storage ----------------------------------------------------------------


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


# ---- routes -----------------------------------------------------------------


def _may_read(key: Any, owner: str) -> bool:
    return owner in key.authors or USER_AUTHOR in key.authors


def _owner_denied(owner: str) -> JSONResponse:
    return error(f"owner {owner!r} is not one of this key's authors", 403)


def _user_denied() -> JSONResponse:
    return error(f"a key with the {USER_AUTHOR!r} author is required", 403)


def _refused(exc: Refused) -> JSONResponse:
    return JSONResponse({"error": str(exc), **exc.details}, status_code=400)


def _stale(exc: Stale) -> JSONResponse:
    return JSONResponse({"error": "stale", "version": exc.version}, status_code=409)


async def profile_route(request: Request) -> JSONResponse:
    """One owner's served parts, their versions, and its pending proposal."""
    key = request.state.key
    try:
        owner = _owner(_query(request, ("owner",)).get("owner"))
    except RequestError as exc:
        return error(str(exc))
    if not _may_read(key, owner):
        return _owner_denied(owner)
    return JSONResponse(await read(owner))


async def self_route(request: Request) -> JSONResponse:
    """Replace the owner's `self` part; `unchanged` when it equals the latest version."""
    key = request.state.key
    try:
        _query(request, ())
        owner, content = parse_self(await _body(request))
    except RequestError as exc:
        return error(str(exc))
    except Refused as exc:
        return _refused(exc)
    if owner not in key.authors:
        return _owner_denied(owner)
    return JSONResponse(await write_self(owner, content))


async def propose_route(request: Request) -> JSONResponse:
    """Propose a full replacement of the owner's `user` part; 409 when its base is stale."""
    key = request.state.key
    try:
        _query(request, ())
        owner, content, reason, base_version = parse_proposal(await _body(request))
    except RequestError as exc:
        return error(str(exc))
    except Refused as exc:
        return _refused(exc)
    if owner not in key.authors:
        return _owner_denied(owner)
    try:
        result = await propose(owner, content, reason, base_version)
    except Stale as exc:
        return _stale(exc)
    return JSONResponse(result, status_code=201)


async def proposals_route(request: Request) -> JSONResponse:
    """Proposals newest first; without `owner`, every owner's, for user-author keys only."""
    key = request.state.key
    try:
        query = _query(request, ("owner", "status", "limit"))
        owner = _owner(query["owner"]) if "owner" in query else None
        status = query.get("status")
        if status is not None and status not in STATUSES:
            raise RequestError(f"status must be one of {STATUSES}")
        limit = _limit(query.get("limit"))
    except RequestError as exc:
        return error(str(exc))
    if owner is None and USER_AUTHOR not in key.authors:
        return _user_denied()
    if owner is not None and not _may_read(key, owner):
        return _owner_denied(owner)
    return JSONResponse(await list_proposals(owner, status, limit))


async def proposal_route(request: Request) -> JSONResponse:
    """One proposal with the owner's current user content."""
    key = request.state.key
    try:
        proposal_id = parse_proposal_id(request.path_params["proposal_id"])
        _query(request, ())
    except RequestError as exc:
        return error(str(exc))
    try:
        result = await proposal(proposal_id)
    except NotFound as exc:
        return error(str(exc), 404)
    if not _may_read(key, result["owner"]):
        return _owner_denied(result["owner"])
    return JSONResponse(result)


async def _decision_route(request: Request, decide) -> JSONResponse:
    if USER_AUTHOR not in request.state.key.authors:
        return _user_denied()
    try:
        proposal_id = parse_proposal_id(request.path_params["proposal_id"])
        _query(request, ())
        note = parse_decision(await _body(request))
    except RequestError as exc:
        return error(str(exc))
    except Refused as exc:
        return _refused(exc)
    try:
        return JSONResponse(await decide(proposal_id, note))
    except NotFound as exc:
        return error(str(exc), 404)
    except NotPending as exc:
        return JSONResponse({"error": "not_pending", "status": exc.status}, status_code=409)
    except Stale as exc:
        return _stale(exc)


async def approve_route(request: Request) -> JSONResponse:
    """Store a pending proposal as the owner's next user version; user-author keys only."""
    return await _decision_route(request, approve)


async def reject_route(request: Request) -> JSONResponse:
    """Reject a pending proposal; user-author keys only."""
    return await _decision_route(request, reject)


async def versions_route(request: Request) -> JSONResponse:
    """One part's versions newest first."""
    key = request.state.key
    try:
        query = _query(request, ("owner", "part", "limit"))
        owner = _owner(query.get("owner"))
        part = query.get("part")
        if part not in PARTS:
            raise RequestError(f"part must be one of {PARTS}")
        limit = _limit(query.get("limit"))
    except RequestError as exc:
        return error(str(exc))
    if not _may_read(key, owner):
        return _owner_denied(owner)
    return JSONResponse(await versions(owner, part, limit))
