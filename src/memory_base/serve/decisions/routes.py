"""REST routes for the decision lane: request validation and status codes."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.serve.common.http import error, json_body, reject_author
from memory_base.serve.decisions import store

RECORD_FIELDS = frozenset(
    {"scope", "title", "decision", "why", "links", "decided_on", "supersedes"}
)
RETIRE_FIELDS = frozenset({"scope", "number", "reason"})


async def _write_body(request: Request, allowed: frozenset[str]):
    """The parsed body, or the 400 response a bad one gets."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    if rejected := reject_author(body):
        return rejected
    if unknown := sorted(set(body) - allowed):
        return error(f"unknown field(s): {', '.join(unknown)}")
    return body


async def record_route(request: Request) -> JSONResponse:
    """Record a decision, optionally superseding an active one."""
    body = await _write_body(request, RECORD_FIELDS)
    if isinstance(body, JSONResponse):
        return body
    try:
        result = await store.record_decision(
            author=request.state.author,
            scope=body.get("scope"),
            title=body.get("title"),
            decision=body.get("decision"),
            why=body.get("why"),
            links=body.get("links"),
            decided_on=body.get("decided_on"),
            supersedes=body.get("supersedes"),
        )
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(result, status_code=201)


async def retire_route(request: Request) -> JSONResponse:
    """Retire an active decision with a reason."""
    body = await _write_body(request, RETIRE_FIELDS)
    if isinstance(body, JSONResponse):
        return body
    try:
        result = await store.retire_decision(
            author=request.state.author,
            scope=body.get("scope"),
            number=body.get("number"),
            reason=body.get("reason"),
        )
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(result)


async def list_route(request: Request) -> JSONResponse:
    """List the caller's decisions in a scope, newest number first."""
    params = request.query_params
    try:
        limit = int(params.get("limit", store.LIST_MAX_LIMIT))
    except ValueError:
        return error("limit must be an integer")
    try:
        rows = await store.list_decisions(
            author=request.state.author,
            scope=params.get("scope"),
            status=params.get("status", "active"),
            limit=limit,
        )
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(rows)


async def get_route(request: Request) -> JSONResponse:
    """One of the caller's decisions in full."""
    try:
        number = int(request.path_params["number"])
    except ValueError:
        return error("number must be an integer")
    try:
        row = await store.get_decision(
            author=request.state.author, scope=request.query_params.get("scope"), number=number
        )
    except store.DecisionNotFound as exc:
        return error(str(exc), 404)
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(row)
