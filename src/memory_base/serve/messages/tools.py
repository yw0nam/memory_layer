"""MCP tools for the message lane: send, list, claim, and cancel addressed signals."""

from __future__ import annotations

import uuid
from typing import Any

from mcp.server.fastmcp import Context

from memory_base.serve.common import rest_client


def _message_uuid(message_id: Any) -> str:
    """Ids come from other senders' rows, so never interpolate one unparsed."""
    try:
        return str(uuid.UUID(str(message_id)))
    except (ValueError, AttributeError, TypeError):
        raise ValueError(f"message_id must be a UUID: {message_id!r}") from None


async def send_message(
    subject: str,
    result: str,
    author: str,
    status: str = "info",
    next_step: str | None = None,
    verification: dict[str, str] | None = None,
    refs: list[str] | None = None,
    scope: str | None = None,
    namespace: str | None = None,
    idempotency_key: str | None = None,
    expires_at: str | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Send an addressed, one-time signal that is never embedded or searchable.

    Without `scope` this is a general message for a namespace: `status` must be
    "info", and `next_step`/`verification` stay null unless there is something
    to act on. With a portable `scope` (`repo:<origin>` or
    `project:<organization>/<project>`) this is a handoff: the latest snapshot
    of a work state, `status` "in_progress", "blocked", or "completed";
    "in_progress" and "blocked" require a nonblank `next_step`, "completed"
    requires none. A new snapshot supersedes the pending one for the same
    subject, and any sender who can access the namespace may publish it.

    `verification`, when given, is exactly {command, status, result} with
    status "passed", "failed", or "not_run". `refs` holds at most 10 absolute
    https URLs; localhost, private/loopback IP literals, and file URLs are
    rejected. `author` must be in the calling key's author allowlist.
    `idempotency_key` (max 128 chars) replays an identical send instead of
    duplicating it. `expires_at` is an ISO 8601 datetime at most 30 days out;
    without it a general message expires after the server's configured default
    TTL and a handoff never expires (it leaves the lane by claim, supersede, or
    cancel).
    """
    body: dict[str, Any] = {
        "subject": subject,
        "result": result,
        "author": author,
        "status": status,
    }
    if next_step is not None:
        body["next"] = next_step
    if verification is not None:
        body["verification"] = verification
    if refs is not None:
        body["refs"] = refs
    if scope is not None:
        body["scope"] = scope
    if namespace is not None:
        body["namespace"] = namespace
    if idempotency_key is not None:
        body["idempotency_key"] = idempotency_key
    if expires_at is not None:
        body["expires_at"] = expires_at
    return await rest_client.call(
        "POST",
        "/messages",
        json=body,
        headers=rest_client.auth_headers(ctx),
    )


async def list_messages(
    namespace: str | None = None,
    purpose: str | None = None,
    scope: str | None = None,
    subject: str | None = None,
    limit: int | None = None,
    ctx: Context | None = None,
) -> list[dict[str, Any]]:
    """List pending, unexpired messages addressed to the caller's namespaces.

    No query and no embedding call: messages are read by address, never by
    similarity, and never appear in search results. Returns newest-first rows
    with exactly id, namespace, purpose, scope, subject, status (the report
    state: info, in_progress, blocked, or completed), author, created_at,
    expires_at (null for a handoff without an expiry), and content — the
    canonical Markdown.

    `purpose` is "message" or "handoff"; `scope` is the portable scope the
    message was sent with; `subject` matches after the same normalization the
    server applies on send (any spelling variant of the subject finds it).
    `namespace` narrows to one namespace the caller's API key can access;
    omitted, it covers every accessible namespace. `limit` defaults to 50,
    max 100.
    """
    params: list[tuple[str, str]] = []
    if namespace is not None:
        params.append(("namespace", namespace))
    if purpose is not None:
        params.append(("purpose", purpose))
    if scope is not None:
        params.append(("scope", scope))
    if subject is not None:
        params.append(("subject", subject))
    if limit is not None:
        params.append(("limit", str(limit)))
    return await rest_client.call(
        "GET",
        "/messages",
        params=params,
        headers=rest_client.auth_headers(ctx),
    )


async def claim_message(message_id: str, ctx: Context | None = None) -> dict[str, Any]:
    """Claim a pending message so no other reader receives it.

    Claiming is exclusive and immediate: exactly one claim of a message
    succeeds; a second claim of the same id is refused. Call this at the start
    of a session for each message from list_messages you are about to act on.
    A 200 is the claim; the returned row keeps its original report `status`.
    An unknown id, or one outside the caller's namespaces, gets a 404; an
    already-claimed, cancelled, superseded, or expired id gets a 409.
    """
    return await rest_client.call(
        "POST",
        f"/messages/{_message_uuid(message_id)}/claim",
        headers=rest_client.auth_headers(ctx),
    )


async def cancel_message(message_id: str, ctx: Context | None = None) -> dict[str, Any]:
    """Withdraw a pending message the sender no longer wants delivered.

    A sender can cancel its own pending messages; an admin key can cancel any
    accessible pending message. A 200 is the cancellation; the returned row
    keeps its report `status`. Unknown, unauthorized, or no-longer-pending ids
    get 404/409.
    """
    return await rest_client.call(
        "DELETE",
        f"/messages/{_message_uuid(message_id)}",
        headers=rest_client.auth_headers(ctx),
    )
