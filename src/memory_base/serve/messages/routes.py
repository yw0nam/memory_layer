"""REST routes for the message lane: request validation, authority, and status codes."""

from __future__ import annotations

import uuid

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.retrieval.search import normalize_namespaces
from memory_base.serve.common.http import error, json_body
from memory_base.serve.messages import store

MESSAGE_BODY_FIELDS = frozenset(
    {
        "namespace",
        "author",
        "subject",
        "status",
        "result",
        "next",
        "verification",
        "refs",
        "scope",
        "idempotency_key",
        "expires_at",
    }
)


async def send_route(request: Request) -> JSONResponse:
    """Publish a message, or a handoff snapshot when a scope is present."""
    key = request.state.key
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    unknown = sorted(set(body) - MESSAGE_BODY_FIELDS)
    if unknown:
        return error(f"unknown field(s): {', '.join(unknown)}")
    namespace = body.get("namespace", key.home)
    if not isinstance(namespace, str) or not namespace.strip():
        return error("namespace must be a non-empty string")
    if not key.permits(namespace):
        return error(f"namespace {namespace!r} is outside the caller's allowed set", 403)
    author = body.get("author")
    if not isinstance(author, str) or not author.strip():
        return error("author is required")
    if author not in key.authors:
        return error(f"author {author!r} is not permitted for this key", 403)
    try:
        row, replayed = await store.send_message(
            key,
            namespace=namespace,
            author=author,
            subject=body.get("subject"),
            status=body.get("status"),
            result=body.get("result"),
            next_text=body.get("next"),
            verification=body.get("verification"),
            refs=body.get("refs"),
            scope=body.get("scope"),
            idempotency_key=body.get("idempotency_key"),
            expires_at=body.get("expires_at"),
        )
    except store.MessageConflict as exc:
        return error(str(exc), 409)
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(row, status_code=200 if replayed else 201)


async def list_route(request: Request) -> JSONResponse:
    """List pending messages by filters alone — no query, no embedding call."""
    key = request.state.key
    params = request.query_params
    try:
        requested_namespaces = normalize_namespaces(params.getlist("namespace") or None)
    except ValueError as exc:
        return error(str(exc))
    if requested_namespaces is not None and not key.permits_all(set(requested_namespaces)):
        return error("requested namespaces are outside the caller's allowed set", 403)
    scope = requested_namespaces
    if scope is None and not key.is_admin:
        scope = sorted(key.allowed)
    raw_limit = params.get("limit")
    try:
        limit = store.LIST_MESSAGES_DEFAULT_LIMIT if raw_limit is None else int(raw_limit)
    except ValueError:
        return error("limit must be an integer")
    if not 1 <= limit <= store.LIST_MESSAGES_MAX_LIMIT:
        return error(f"limit must be between 1 and {store.LIST_MESSAGES_MAX_LIMIT}")
    purpose = params.get("purpose") or None
    try:
        rows = await store.list_messages(
            namespaces=scope,
            purpose=purpose,
            scope=params.get("scope") or None,
            subject=params.get("subject") or None,
            limit=limit,
        )
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(rows)


def _message_id(request: Request) -> uuid.UUID | None:
    raw = request.path_params["message_id"]
    try:
        return uuid.UUID(raw)
    except ValueError:
        return None


async def claim_route(request: Request) -> JSONResponse:
    """Claim a pending message at most once; stale or terminal ids get 409."""
    message_id = _message_id(request)
    if message_id is None:
        return error("message id must be a UUID")
    try:
        row = await store.claim_message(message_id, request.state.key)
    except store.MessageConflict as exc:
        return error(str(exc), 409)
    except store.MessageNotFound as exc:
        return error(str(exc), 404)
    return JSONResponse(row)


async def cancel_route(request: Request) -> JSONResponse:
    """Cancel a pending message; the sender, or an admin for any accessible one."""
    message_id = _message_id(request)
    if message_id is None:
        return error("message id must be a UUID")
    try:
        row = await store.cancel_message(message_id, request.state.key)
    except store.MessageConflict as exc:
        return error(str(exc), 409)
    except store.MessageNotFound as exc:
        return error(str(exc), 404)
    return JSONResponse(row)
