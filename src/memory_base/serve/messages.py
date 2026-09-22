"""The message lane: addressed, once-claimed signals that are never embedded.

A message is read by address, not by similarity, so it is stored as canonical
Markdown without an embedding and can never surface in search. A send without
a scope is a general message for a namespace; a send with a portable scope
(``repo:<origin>`` or ``project:<organization>/<project>``) is a handoff — the
latest snapshot of a work state, superseding the pending one in the same
transaction. Claims are at-most-once: a single conditional UPDATE decides the
winner by commit order.
"""

from __future__ import annotations

import ipaddress
import os
import re
import unicodedata
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

import asyncpg

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.schema import ensure_schema_once
from memory_base.serve import namespaces

MESSAGE_MAX_TTL_DAYS = 30
MESSAGE_TTL_DAYS = int(os.getenv("MESSAGE_TTL_DAYS", "7"))
if not 1 <= MESSAGE_TTL_DAYS <= MESSAGE_MAX_TTL_DAYS:
    raise RuntimeError(
        f"MESSAGE_TTL_DAYS must be 1..{MESSAGE_MAX_TTL_DAYS}, got {MESSAGE_TTL_DAYS}"
    )
MESSAGE_MAX_CONTENT_BYTES = 4096
LIST_MESSAGES_DEFAULT_LIMIT = 50
LIST_MESSAGES_MAX_LIMIT = 100
MAX_REFS = 10
IDEMPOTENCY_KEY_MAX_CHARS = 128
SCOPE_MAX_CHARS = 512
GENERAL_STATUSES = ("info",)
HANDOFF_STATUSES = ("in_progress", "blocked", "completed")
VERIFICATION_STATUSES = ("passed", "failed", "not_run")

_PROJECT_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SCP_ORIGIN_RE = re.compile(r"(?:(?P<user>[^:@]+)@)?(?P<host>[^:/]+):(?P<path>.+)")
_ORIGIN_HOST_RE = re.compile(r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")

_PUBLIC_COLUMNS = (
    "id, namespace, purpose, scope, subject, status, author, created_at, expires_at, content"
)


class MessageConflict(Exception):
    """The message exists but is not claimable, or an idempotency key collided."""


class MessageNotFound(Exception):
    """No message with that id is visible to the caller."""


def normalize_subject(subject: Any) -> str:
    """NFKC-fold, trim, and collapse whitespace so a heading cannot escape."""
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("subject must be a non-empty string")
    collapsed = " ".join(unicodedata.normalize("NFKC", subject).split())
    if not collapsed:
        raise ValueError("subject must be a non-empty string")
    return collapsed


def subject_key_for(subject: str) -> str:
    """The normalized subject folded for equality across senders and harnesses."""
    return normalize_subject(subject).casefold()


def normalize_scope(scope: Any) -> str:
    """Validate a portable scope: repo:<normalized-origin> or project:<org>/<proj>.

    Canonical repo scopes (`repo:github.com/org/repo`) round-trip unchanged;
    https and scp-style SSH origins additionally normalize onto that form. The
    hostname is lowercased; repository path case is preserved. A local checkout
    is not portable, so the host must be a dotted public hostname.
    """
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("scope must be repo:<origin> or project:<organization>/<project>")
    scope = scope.strip()
    if scope.startswith("repo:"):
        origin = scope[len("repo:") :]
        if any(char.isspace() for char in origin):
            raise ValueError("repo scope origin must not contain whitespace")
        if "://" in origin:
            parts = urlsplit(origin)
            if parts.scheme != "https" or not parts.hostname:
                raise ValueError("repo scope origin must be an https URL, scp origin, or host/path")
            if parts.username is not None or parts.password is not None:
                raise ValueError("repo scope origin must not embed credentials")
            if parts.port is not None:
                raise ValueError("repo scope origin must not include a port")
            host, path = parts.hostname, parts.path
        elif (scp := _SCP_ORIGIN_RE.fullmatch(origin)) is not None:
            if scp["user"] not in (None, "git"):
                raise ValueError("repo scope origin must not embed credentials")
            host, path = scp["host"], scp["path"]
        else:
            host, _, path = origin.partition("/")
            path = path.split("?", 1)[0].split("#", 1)[0]
        if not _ORIGIN_HOST_RE.fullmatch(host):
            raise ValueError(f"repo scope origin must name a remote host, not {host!r}")
        path = "/" + path.strip("/")
        if path.endswith(".git"):
            path = path[: -len(".git")]
        if not path.strip("/"):
            raise ValueError("repo scope origin must include a repository path")
        normalized = f"repo:{host.lower()}{path}"
        if len(normalized) > SCOPE_MAX_CHARS:
            raise ValueError(f"scope must be at most {SCOPE_MAX_CHARS} chars")
        return normalized
    if scope.startswith("project:"):
        rest = scope[len("project:") :]
        segments = rest.split("/")
        if len(segments) != 2 or not all(_PROJECT_SEGMENT_RE.fullmatch(s) for s in segments):
            raise ValueError("project scope must be project:<organization>/<project>")
        return f"project:{segments[0].lower()}/{segments[1].lower()}"
    raise ValueError("scope must be repo:<origin> or project:<organization>/<project>")


def _validate_ref(ref: Any) -> str:
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("each ref must be an absolute https URL")
    ref = ref.strip()
    if any(char.isspace() for char in ref):
        raise ValueError(f"ref must not contain whitespace: {ref!r}")
    parts = urlsplit(ref)
    if parts.scheme != "https" or not parts.netloc:
        raise ValueError(f"ref must be an absolute https URL: {ref!r}")
    if parts.username is not None or parts.password is not None:
        raise ValueError(f"ref must not embed credentials: {ref!r}")
    host = parts.hostname
    if host is None:
        raise ValueError(f"ref must include a host: {ref!r}")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"ref must not point at localhost: {ref!r}")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if address.is_loopback or address.is_private or address.is_link_local:
            raise ValueError(f"ref must not point at a local address: {ref!r}")
    return ref


def validate_refs(refs: Any) -> list[str] | None:
    """Validate the refs list without ever fetching: absolute https, no local targets."""
    if refs is None:
        return None
    if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
        raise ValueError("refs must be a list of https URLs")
    if len(refs) > MAX_REFS:
        raise ValueError(f"refs must hold at most {MAX_REFS} URLs")
    return [_validate_ref(ref) for ref in refs]


def validate_verification(verification: Any) -> dict[str, str] | None:
    """Null, or exactly {command, status, result} with a known status."""
    if verification is None:
        return None
    expected = {"command", "status", "result"}
    if not isinstance(verification, dict) or set(verification) != expected:
        raise ValueError(f"verification must be exactly {expected}")
    command = verification["command"]
    result = verification["result"]
    if not isinstance(command, str) or not command.strip():
        raise ValueError("verification command must be a non-empty string")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("verification result must be a non-empty string")
    if verification["status"] not in VERIFICATION_STATUSES:
        raise ValueError(f"verification status must be one of {VERIFICATION_STATUSES}")
    return {"command": command, "status": verification["status"], "result": result}


def _quote(text: str) -> str:
    """Blockquote every line so injected headings or fences cannot escape."""
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines())


def _quote_labeled(label: str, value: str) -> str:
    """A labeled Verification value with every line blockquoted."""
    first, _, rest = value.partition("\n")
    lines = [f"> {label}: {first}" if first.strip() else f"> {label}:"]
    lines.extend(f"> {line}" if line.strip() else ">" for line in rest.splitlines())
    return "\n".join(lines)


def render_content(
    subject: str,
    status: str,
    result: str,
    next_text: str | None,
    verification: dict[str, str] | None,
    refs: list[str] | None,
) -> str:
    """Render the canonical Markdown; reject instead of truncating past 4 KiB."""
    sections = [f"# {subject}", "## Status", _quote(status), "## Result", _quote(result)]
    if next_text is not None:
        sections += ["## Next", _quote(next_text)]
    if verification is not None:
        command = "\n".join(verification["command"].splitlines())
        result = "\n".join(verification["result"].splitlines())
        sections += [
            "## Verification",
            "\n".join(
                [
                    f"> Status: {verification['status']}",
                    _quote_labeled("Command", command),
                    _quote_labeled("Result", result),
                ]
            ),
        ]
    if refs:
        sections += ["## References", *[f"- {ref}" for ref in refs]]
    content = "\n\n".join(sections)
    if len(content.encode("utf-8")) > MESSAGE_MAX_CONTENT_BYTES:
        raise ValueError(
            f"rendered content exceeds {MESSAGE_MAX_CONTENT_BYTES} bytes; shorten the message"
        )
    return content


def resolve_expires_at(expires_at: Any) -> datetime:
    """The caller's expiry (future, at most MESSAGE_MAX_TTL_DAYS out) or now + TTL."""
    now = datetime.now(timezone.utc)
    if expires_at is None:
        return now + timedelta(days=MESSAGE_TTL_DAYS)
    if not isinstance(expires_at, str) or not expires_at.strip():
        raise ValueError("expires_at must be an ISO 8601 datetime string")
    try:
        parsed = datetime.fromisoformat(expires_at.strip())
    except ValueError:
        raise ValueError(f"invalid expires_at: {expires_at!r}") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed <= now:
        raise ValueError("expires_at must be in the future")
    if parsed > now + timedelta(days=MESSAGE_MAX_TTL_DAYS):
        raise ValueError(f"expires_at must be at most {MESSAGE_MAX_TTL_DAYS} days out")
    return parsed


def validate_idempotency_key(idempotency_key: Any) -> str | None:
    if idempotency_key is None:
        return None
    if not isinstance(idempotency_key, str):
        raise ValueError("idempotency_key must be a string")
    key = idempotency_key.strip()
    if not key or len(key) > IDEMPOTENCY_KEY_MAX_CHARS:
        raise ValueError(f"idempotency_key must be 1..{IDEMPOTENCY_KEY_MAX_CHARS} chars")
    return key


def derive_purpose_and_check_status(scope: str | None, status: Any) -> str:
    """Scope presence decides the purpose; each purpose owns its status set."""
    purpose = "handoff" if scope is not None else "message"
    allowed = HANDOFF_STATUSES if purpose == "handoff" else GENERAL_STATUSES
    if status not in allowed:
        raise ValueError(f"{purpose} status must be one of {allowed}")
    return purpose


def check_next(purpose: str, status: str, next_text: Any) -> str | None:
    """in_progress/blocked carry a nonblank next; completed carries none."""
    if next_text is None:
        if purpose == "handoff" and status in ("in_progress", "blocked"):
            raise ValueError(f"a {status} handoff requires a nonblank next")
        return None
    if not isinstance(next_text, str) or not next_text.strip():
        raise ValueError("next must be a non-empty string or null")
    if purpose == "handoff" and status == "completed":
        raise ValueError("a completed handoff must set next to null")
    return next_text


def public_row(row: asyncpg.Record | dict[str, Any]) -> dict[str, Any]:
    """The contracted response shape; lifecycle timestamps and key ids stay inside.

    `status` is the report state (info, in_progress, blocked, completed) and
    never changes; a 200 claim or cancel response itself proves the delivery
    transition.
    """
    return {
        "id": str(row["id"]),
        "namespace": row["namespace"],
        "purpose": row["purpose"],
        "scope": row["scope"],
        "subject": row["subject"],
        "status": row["status"],
        "author": row["author"],
        "created_at": row["created_at"].isoformat(),
        "expires_at": row["expires_at"].isoformat(),
        "content": row["content"],
    }


async def send_message(
    key,
    *,
    namespace: str,
    author: str,
    subject: str,
    status: str,
    result: str,
    next_text: str | None = None,
    verification: dict[str, str] | None = None,
    refs: list[str] | None = None,
    scope: str | None = None,
    idempotency_key: str | None = None,
    expires_at: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Validate, render, and store one message; the second value marks a replay.

    A handoff insert terminalizes older pending snapshots of the same
    namespace+scope+subject_key in the same transaction. With an idempotency
    key, an identical effective request replays the stored row and a different
    one is refused.
    """
    if not isinstance(result, str) or not result.strip():
        raise ValueError("result must be a non-empty string")
    subject = normalize_subject(subject)
    subject_key = subject_key_for(subject)
    scope = normalize_scope(scope) if scope is not None else None
    purpose = derive_purpose_and_check_status(scope, status)
    next_text = check_next(purpose, status, next_text)
    verification = validate_verification(verification)
    refs = validate_refs(refs)
    content = render_content(subject, status, result, next_text, verification, refs)
    expires = resolve_expires_at(expires_at)
    expires_specified = expires_at is not None
    idem = validate_idempotency_key(idempotency_key)

    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        async with conn.transaction():
            await namespaces.require_registered(conn, namespace)
            if purpose == "handoff":
                # ponytail: one hashed lock per chain; a collision only over-serializes.
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                    f"{namespace}\x1f{scope}\x1f{subject_key}",
                )
            if idem is not None:
                existing = await conn.fetchrow(
                    f"""
                    SELECT {_PUBLIC_COLUMNS}, subject_key, idempotency_key
                    FROM "{PG_SCHEMA}".messages
                    WHERE sender_key = $1 AND idempotency_key = $2
                    """,
                    key.key_id,
                    idem,
                )
                if existing is not None:
                    same = _same_intent(
                        existing,
                        namespace,
                        purpose,
                        scope,
                        subject_key,
                        content,
                        author,
                        expires,
                        expires_specified,
                    )
                    if not same:
                        raise MessageConflict(
                            f"idempotency_key {idem!r} was already used for a different message"
                        )
                    return public_row(existing), True
            try:
                async with conn.transaction():
                    row = await conn.fetchrow(
                        f"""
                        INSERT INTO "{PG_SCHEMA}".messages
                          (id, namespace, purpose, scope, subject, subject_key, status,
                           content, author, sender_key, idempotency_key, expires_at,
                           created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                                clock_timestamp())
                        RETURNING {_PUBLIC_COLUMNS}
                        """,
                        uuid.uuid4(),
                        namespace,
                        purpose,
                        scope,
                        subject,
                        subject_key,
                        status,
                        content,
                        author,
                        key.key_id,
                        idem,
                        expires,
                    )
            except asyncpg.UniqueViolationError:
                # A concurrent twin won the unique (sender_key, idempotency_key) index.
                twin = await conn.fetchrow(
                    f"""
                    SELECT {_PUBLIC_COLUMNS}, subject_key, idempotency_key
                    FROM "{PG_SCHEMA}".messages
                    WHERE sender_key = $1 AND idempotency_key = $2
                    """,
                    key.key_id,
                    idem,
                )
                if twin is None:
                    raise
                same = _same_intent(
                    twin,
                    namespace,
                    purpose,
                    scope,
                    subject_key,
                    content,
                    author,
                    expires,
                    expires_specified,
                )
                if not same:
                    raise MessageConflict(
                        f"idempotency_key {idem!r} was already used for a different message"
                    ) from None
                return public_row(twin), True
            if purpose == "handoff":
                await conn.execute(
                    f"""
                    UPDATE "{PG_SCHEMA}".messages
                    SET superseded_at = now()
                    WHERE namespace = $1 AND purpose = 'handoff' AND scope = $2
                      AND subject_key = $3 AND id <> $4
                      AND claimed_at IS NULL AND cancelled_at IS NULL
                      AND superseded_at IS NULL
                    """,
                    namespace,
                    scope,
                    subject_key,
                    row["id"],
                )
    return public_row(row), False


def _same_intent(
    existing: asyncpg.Record,
    namespace: str,
    purpose: str,
    scope: str | None,
    subject_key: str,
    content: str,
    author: str,
    expires: datetime,
    expires_specified: bool,
) -> bool:
    """An idempotent replay must carry the same effective request.

    A caller-supplied expiry must match exactly; a server-defaulted one is not
    part of the intent, or a replayed identical request would always conflict.
    """
    return (
        existing["namespace"] == namespace
        and existing["purpose"] == purpose
        and existing["scope"] == scope
        and existing["subject_key"] == subject_key
        and existing["content"] == content
        and existing["author"] == author
        and (not expires_specified or existing["expires_at"] == expires)
    )


async def list_messages(
    *,
    namespaces: list[str] | None = None,
    purpose: str | None = None,
    scope: str | None = None,
    subject: str | None = None,
    limit: int = LIST_MESSAGES_DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Pending, unexpired messages newest-first; no query and no embedding call."""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= LIST_MESSAGES_MAX_LIMIT
    ):
        raise ValueError(f"limit must be an integer between 1 and {LIST_MESSAGES_MAX_LIMIT}")
    if purpose is not None and purpose not in ("message", "handoff"):
        raise ValueError("purpose must be message or handoff")
    if scope is not None:
        scope = normalize_scope(scope)
    subject_key = subject_key_for(normalize_subject(subject)) if subject is not None else None
    predicates = [
        "claimed_at IS NULL",
        "cancelled_at IS NULL",
        "superseded_at IS NULL",
        "expires_at > now()",
    ]
    args: list[Any] = []
    if namespaces is not None:
        args.append(namespaces)
        predicates.append(f"namespace = ANY(${len(args)}::text[])")
    if purpose is not None:
        args.append(purpose)
        predicates.append(f"purpose = ${len(args)}")
    if scope is not None:
        args.append(scope)
        predicates.append(f"scope = ${len(args)}")
    if subject_key is not None:
        args.append(subject_key)
        predicates.append(f"subject_key = ${len(args)}")
    args.append(limit)
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f"""
            SELECT {_PUBLIC_COLUMNS}
            FROM "{PG_SCHEMA}".messages
            WHERE {" AND ".join(predicates)}
            ORDER BY created_at DESC, id DESC
            LIMIT ${len(args)}
            """,
            *args,
        )
    return [public_row(row) for row in rows]


async def claim_message(message_id: uuid.UUID, key, *, connection=None) -> dict[str, Any]:
    """Claim a pending message at most once; the conditional UPDATE decides."""

    async def _claim(conn):
        row = await conn.fetchrow(
            f'SELECT namespace FROM "{PG_SCHEMA}".messages WHERE id = $1', message_id
        )
        if row is None or not key.permits(row["namespace"]):
            raise MessageNotFound(f"no claimable message {message_id}")
        updated = await conn.fetchrow(
            f"""
            UPDATE "{PG_SCHEMA}".messages
            SET claimed_at = now()
            WHERE id = $1
              AND claimed_at IS NULL AND cancelled_at IS NULL AND superseded_at IS NULL
              AND expires_at > now()
            RETURNING {_PUBLIC_COLUMNS}
            """,
            message_id,
        )
        if updated is None:
            raise MessageConflict(f"message {message_id} is not pending")
        return public_row(updated)

    if connection is not None:
        return await _claim(connection)
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        return await _claim(conn)


async def cancel_message(message_id: uuid.UUID, key, *, connection=None) -> dict[str, Any]:
    """A sender withdraws its own pending message; an admin cancels any pending."""

    async def _cancel(conn):
        row = await conn.fetchrow(
            f'SELECT namespace, sender_key FROM "{PG_SCHEMA}".messages WHERE id = $1', message_id
        )
        if row is None or not key.permits(row["namespace"]):
            raise MessageNotFound(f"no cancellable message {message_id}")
        if not (key.is_admin or row["sender_key"] == key.key_id):
            raise MessageNotFound(f"no cancellable message {message_id}")
        updated = await conn.fetchrow(
            f"""
            UPDATE "{PG_SCHEMA}".messages
            SET cancelled_at = now()
            WHERE id = $1
              AND claimed_at IS NULL AND cancelled_at IS NULL AND superseded_at IS NULL
              AND expires_at > now()
            RETURNING {_PUBLIC_COLUMNS}
            """,
            message_id,
        )
        if updated is None:
            raise MessageConflict(f"message {message_id} is not pending")
        return public_row(updated)

    if connection is not None:
        return await _cancel(connection)
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        return await _cancel(conn)


_TERMINAL_PREDICATE = (
    "(claimed_at IS NOT NULL OR cancelled_at IS NOT NULL "
    "OR superseded_at IS NOT NULL OR expires_at <= now())"
)


async def terminal_messages(namespaces: list[str] | None = None) -> list[dict[str, Any]]:
    """Claimed, cancelled, superseded, or expired messages, for the purge preview."""
    args: list[Any] = []
    scope_sql = ""
    if namespaces is not None:
        args.append(namespaces)
        scope_sql = f" AND namespace = ANY(${len(args)}::text[])"
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        rows = await conn.fetch(
            f"""
            SELECT {_PUBLIC_COLUMNS}
            FROM "{PG_SCHEMA}".messages
            WHERE {_TERMINAL_PREDICATE}{scope_sql}
            ORDER BY created_at DESC
            """,
            *args,
        )
    return [public_row(row) for row in rows]


async def delete_terminal_messages(namespaces: list[str] | None = None) -> int:
    """Delete claimed/cancelled/superseded/expired messages; releasing idempotency keys."""
    args: list[Any] = []
    scope_sql = ""
    if namespaces is not None:
        args.append(namespaces)
        scope_sql = f" AND namespace = ANY(${len(args)}::text[])"
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        status = await conn.execute(
            f'DELETE FROM "{PG_SCHEMA}".messages WHERE {_TERMINAL_PREDICATE}{scope_sql}',
            *args,
        )
        return int(status.rsplit(" ", 1)[-1])
