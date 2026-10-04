"""REST routes for consolidation: issue groups, apply verdicts, list and undo actions."""

from __future__ import annotations

import math

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.serve import namespaces
from memory_base.serve.common.http import error, json_body
from memory_base.serve.consolidation import groups, verdicts

CONSOLIDATE_AUTHOR = "consolidator"


def _scalar(request: Request, name: str, parse, default, valid, rule: str):
    """One optional query value: parsed, range-checked, and given at most once."""
    values = request.query_params.getlist(name)
    if len(values) > 1:
        raise ValueError(f"{name} must be given at most once")
    if not values:
        return default
    try:
        value = parse(values[0])
    except ValueError:
        raise ValueError(f"{name} must be {rule}") from None
    if not valid(value):
        raise ValueError(f"{name} must be {rule}")
    return value


def _consolidator_denied(key) -> JSONResponse | None:
    if key.is_admin and CONSOLIDATE_AUTHOR in key.authors:
        return None
    return error(f"admin key with {CONSOLIDATE_AUTHOR!r} in its authors required", 403)


async def groups_route(request: Request) -> JSONResponse:
    """List groups of active notes that may state the same thing; changes no note."""
    denied = _consolidator_denied(request.state.key)
    if denied is not None:
        return denied
    try:
        threshold = _scalar(
            request,
            "threshold",
            float,
            groups.DEFAULT_THRESHOLD,
            lambda x: math.isfinite(x) and groups.MIN_THRESHOLD < x <= groups.MAX_THRESHOLD,
            f"a number in ({groups.MIN_THRESHOLD:g}, {groups.MAX_THRESHOLD:g}]",
        )
        neighbors = _scalar(
            request,
            "neighbors",
            int,
            groups.DEFAULT_NEIGHBORS,
            lambda x: groups.MIN_NEIGHBORS <= x <= groups.MAX_NEIGHBORS,
            f"an integer between {groups.MIN_NEIGHBORS} and {groups.MAX_NEIGHBORS}",
        )
        max_group = _scalar(
            request,
            "max_group",
            int,
            groups.DEFAULT_MAX_GROUP,
            lambda x: groups.MIN_MAX_GROUP <= x <= groups.MAX_MAX_GROUP,
            f"an integer between {groups.MIN_MAX_GROUP} and {groups.MAX_MAX_GROUP}",
        )
        max_group_chars = _scalar(
            request,
            "max_group_chars",
            int,
            groups.DEFAULT_MAX_GROUP_CHARS,
            lambda x: x >= groups.MIN_MAX_GROUP_CHARS,
            f"an integer of at least {groups.MIN_MAX_GROUP_CHARS}",
        )
        limit = _scalar(
            request,
            "limit",
            int,
            groups.DEFAULT_LIMIT,
            lambda x: groups.MIN_LIMIT <= x <= groups.MAX_LIMIT,
            f"an integer between {groups.MIN_LIMIT} and {groups.MAX_LIMIT}",
        )
    except ValueError as exc:
        return error(str(exc))
    requested = request.query_params.getlist("namespace")
    if any(not name.strip() for name in requested):
        return error("namespace must not be blank")
    registered = {row["name"] for row in await namespaces.list_namespaces()}
    unknown = sorted(set(requested) - registered)
    if unknown:
        return error(f"unregistered namespace: {', '.join(unknown)}")
    names = sorted(set(requested or registered))
    snapshots = await groups.read_snapshots(names, threshold, neighbors)
    return JSONResponse(
        {
            "params": {
                "namespace": names,
                "threshold": threshold,
                "neighbors": neighbors,
                "max_group": max_group,
                "max_group_chars": max_group_chars,
                "limit": limit,
            },
            "procedure_version": groups.PROCEDURE_VERSION,
            "namespaces": {
                name: groups.namespace_report(
                    name,
                    snapshot,
                    threshold=threshold,
                    max_group=max_group,
                    max_group_chars=max_group_chars,
                    limit=limit,
                )
                for name, snapshot in snapshots.items()
            },
        }
    )


async def verdicts_route(request: Request) -> JSONResponse:
    """Validate and apply (or plan, on a dry run) verdicts on issued groups, each alone."""
    key = request.state.key
    denied = _consolidator_denied(key)
    if denied is not None:
        return denied
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    registered = {row["name"] for row in await namespaces.list_namespaces()}
    try:
        batch = verdicts.parse_batch(body, registered)
    except verdicts.RequestError as exc:
        return error(str(exc))
    if batch.author not in key.authors:
        return error(f"author {batch.author!r} is not permitted for this key", 403)
    return JSONResponse({"results": await verdicts.process_batch(batch)})


async def undo_route(request: Request) -> JSONResponse:
    """Reverse one consolidation action: 404 unknown, 409 refused with nothing changed."""
    key = request.state.key
    denied = _consolidator_denied(key)
    if denied is not None:
        return denied
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    try:
        action_id, author = verdicts.parse_undo(body)
    except verdicts.RequestError as exc:
        return error(str(exc))
    if author not in key.authors:
        return error(f"author {author!r} is not permitted for this key", 403)
    try:
        result = await verdicts.undo(action_id, author)
    except verdicts.UndoNotFound as exc:
        return error(str(exc), 404)
    except verdicts.UndoRefused as exc:
        return error(str(exc), 409)
    return JSONResponse(result)


async def actions_route(request: Request) -> JSONResponse:
    """List consolidation actions newest first with the notes they reference."""
    denied = _consolidator_denied(request.state.key)
    if denied is not None:
        return denied
    try:
        filters = {
            name: _scalar(request, name, str, None, lambda x: bool(x.strip()), "non-blank")
            for name in ("namespace", "run_id", "note_id")
        }
        limit = _scalar(
            request,
            "limit",
            int,
            verdicts.DEFAULT_ACTIONS_LIMIT,
            lambda x: 1 <= x <= verdicts.MAX_ACTIONS_LIMIT,
            f"an integer between 1 and {verdicts.MAX_ACTIONS_LIMIT}",
        )
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse(await verdicts.list_actions(**filters, limit=limit))
