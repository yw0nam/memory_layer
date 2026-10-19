"""REST routes for facts: the consolidation write batch and the entity card read.

A write is attributed to the X-Author header; the card is read across every namespace
unless filtered.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.core.secrets import find_secret
from memory_base.retrieval.search import normalize_namespaces
from memory_base.serve.common.entities import SLUG_RE, entity_row
from memory_base.serve.common.http import (
    RequestError,
    bounded_text,
    error,
    object_fields,
    request_body,
)
from memory_base.serve.facts import store

MAX_OPS = 100
MAX_VALUE_CHARS = 300
ATTRIBUTE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
LIST_VALUE_RE = re.compile(r"^[^,]+(,\s*[^,]+)+(\s+and\s+[^,]+)?$")
NAMES_JOINED_RE = re.compile(r"^[A-Z][\w.-]*(?: [A-Z][\w.-]*)? and [A-Z][\w.-]*(?: [A-Z][\w.-]*)?$")
VALUES = {"value_text", "value_entity"}
SHAPES = {
    "add": ({"entity", "attribute", "note_id"}, VALUES | {"allow_multiple"}),
    "replace": ({"id", "note_id"}, VALUES | {"attribute"}),
    "remove": ({"id", "note_id"}, set()),
}


def _slug(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SLUG_RE.fullmatch(value):
        raise RequestError(f"{name} must match {SLUG_RE.pattern}")
    return value


def _attribute(value: Any) -> str:
    if not isinstance(value, str) or not ATTRIBUTE_RE.fullmatch(value):
        raise RequestError(f"attribute must match {ATTRIBUTE_RE.pattern}")
    return value


def _value_text(value: Any) -> str:
    text = bounded_text(value, "value_text", 1, MAX_VALUE_CHARS)
    if find_secret(text) is not None:
        raise RequestError(f"value_text contains a credential ({find_secret(text)})")
    if LIST_VALUE_RE.fullmatch(text) or NAMES_JOINED_RE.fullmatch(text):
        raise RequestError("value lists several names; one row per element")
    return text


def _op(raw: Any) -> dict[str, Any]:
    kind = raw.get("op") if isinstance(raw, dict) else None
    if kind not in SHAPES:
        raise RequestError(f"op must be one of {sorted(SHAPES)}")
    required, optional = SHAPES[kind]
    fields = object_fields({k: v for k, v in raw.items() if k != "op"}, required, optional)
    op: dict[str, Any] = {
        "op": kind,
        "note_id": bounded_text(fields["note_id"], "note_id", 1, 200),
    }
    if "entity" in required:
        op["entity"] = _slug(fields["entity"], "entity")
    if "id" in required:
        try:
            op["id"] = uuid.UUID(str(fields["id"]))
        except ValueError:
            raise RequestError("id must be a fact id") from None
    if "attribute" in fields:
        op["attribute"] = _attribute(fields["attribute"])
    if kind != "remove":
        given = {name for name in VALUES if fields.get(name) is not None}
        if len(given) != 1:
            raise RequestError("exactly one of value_text and value_entity is required")
        if "value_text" in given:
            op["value_text"] = _value_text(fields["value_text"])
        else:
            op["value_entity"] = _slug(fields["value_entity"], "value_entity")
    if kind == "add":
        allow = fields.get("allow_multiple", False)
        if not isinstance(allow, bool):
            raise RequestError("allow_multiple must be a boolean")
        op["allow_multiple"] = allow
    return op


def parse_batch(body: Any) -> tuple[list[dict[str, Any]], bool]:
    """`{ops, dry_run?}` → (ops, dry_run)."""
    fields = object_fields(body, {"ops"}, {"dry_run"})
    ops, dry_run = fields["ops"], fields.get("dry_run", False)
    if not isinstance(dry_run, bool):
        raise RequestError("dry_run must be a boolean")
    if not isinstance(ops, list) or not 1 <= len(ops) <= MAX_OPS:
        raise RequestError(f"ops must be a list of 1 to {MAX_OPS} items")
    parsed = []
    for index, raw in enumerate(ops):
        try:
            parsed.append(_op(raw))
        except RequestError as exc:
            raise RequestError(f"op {index}: {exc}") from None
    return parsed, dry_run


async def write_route(request: Request) -> JSONResponse:
    """Apply a batch of fact ops all or nothing; `dry_run` validates and returns the plan."""
    try:
        ops, dry_run = parse_batch(await request_body(request))
    except RequestError as exc:
        return error(str(exc))
    try:
        results = await store.apply(ops, request.state.author, dry_run)
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse({"results": results})


async def card_route(request: Request) -> JSONResponse:
    """An entity's registry row with the facts it holds and the facts that point at it."""
    params = request.query_params
    flag = params.get("include_past", "false")
    if flag not in ("true", "false"):
        return error("include_past must be true or false")
    try:
        requested = normalize_namespaces(params.getlist("namespace") or None)
    except ValueError as exc:
        return error(str(exc))
    card = await store.card(request.path_params["slug"], flag == "true", requested)
    if card is None:
        return error("unknown entity", 404)
    entity, facts, incoming = card
    return JSONResponse({"entity": entity_row(entity), "facts": facts, "incoming": incoming})
