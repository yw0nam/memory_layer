"""REST routes for the entity registry: request validation, authority, and status codes.

Authority comes from the key's authors: a write names the author it acts as, which must be
one of them. The `import` consent method is for an admin key carrying the `user` author.
"""

from __future__ import annotations

import re
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.serve.access.keys import AUTHOR_RE
from memory_base.serve.common.consent import CONSENT_METHODS
from memory_base.serve.common.http import (
    RequestError,
    bounded_text,
    error,
    iso,
    object_fields,
    request_body,
)
from memory_base.serve.entities import store

KINDS = ("person", "agent", "project", "org", "place", "thing")
USER_AUTHOR = "user"
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
MAX_ITEMS = 50
MAX_NAME_CHARS = 200
MAX_ALIASES = 20
MAX_DESCRIPTION_CHARS = 1000
ITEM_FIELDS = {"slug", "kind", "name", "aliases", "description"}
TIMESTAMPS = ("consented_at", "created_at", "updated_at", "archived_at")


def _slug(value: Any) -> str:
    if not isinstance(value, str) or not SLUG_RE.fullmatch(value):
        raise RequestError("slug must match ^[a-z0-9][a-z0-9-]{0,39}$")
    return value


def _item(raw: Any) -> dict[str, Any]:
    fields = object_fields(raw, {"slug"}, ITEM_FIELDS - {"slug"})
    item = {"slug": _slug(fields["slug"])}
    if "kind" in fields:
        if fields["kind"] not in KINDS:
            raise RequestError(f"kind must be one of {KINDS}")
        item["kind"] = fields["kind"]
    if "name" in fields:
        item["name"] = bounded_text(fields["name"], "name", 1, MAX_NAME_CHARS)
    if "aliases" in fields:
        aliases = fields["aliases"]
        if not isinstance(aliases, list) or len(aliases) > MAX_ALIASES:
            raise RequestError(f"aliases must be a list of at most {MAX_ALIASES} strings")
        item["aliases"] = [bounded_text(alias, "alias", 1, MAX_NAME_CHARS) for alias in aliases]
    if "description" in fields:
        item["description"] = bounded_text(
            fields["description"], "description", 0, MAX_DESCRIPTION_CHARS
        )
    return item


def _consent(fields: dict[str, Any]) -> tuple[str, str, bool]:
    author, method = fields["author"], fields["consent_method"]
    dry_run = fields.get("dry_run", False)
    if not isinstance(dry_run, bool):
        raise RequestError("dry_run must be a boolean")
    if not isinstance(author, str) or not AUTHOR_RE.fullmatch(author):
        raise RequestError("author must be an author slug")
    if method not in CONSENT_METHODS:
        raise RequestError(f"consent_method must be one of {CONSENT_METHODS}")
    return author, method, dry_run


def parse_upsert(body: Any) -> tuple[list[dict[str, Any]], str, str, bool]:
    """`{entities, author, consent_method, dry_run?}` → (items, author, method, dry_run)."""
    fields = object_fields(body, {"entities", "author", "consent_method"}, {"dry_run"})
    items = fields["entities"]
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise RequestError(f"entities must be a list of 1 to {MAX_ITEMS} items")
    return [_item(item) for item in items], *_consent(fields)


def _denied(key: Any, author: str, method: str) -> JSONResponse | None:
    if author not in key.authors:
        return error(f"author {author!r} is not one of this key's authors", 403)
    if method == "import" and not (key.is_admin and USER_AUTHOR in key.authors):
        return error("consent_method import needs an admin key carrying the 'user' author", 403)
    return None


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, **{name: iso(row[name]) for name in TIMESTAMPS}}


async def list_route(request: Request) -> JSONResponse:
    """The registry ordered by slug; archived entities only with include_archived=true."""
    flag = request.query_params.get("include_archived", "false")
    if flag not in ("true", "false"):
        return error("include_archived must be true or false")
    return JSONResponse(
        {"entities": [_public(row) for row in await store.list_entities(flag == "true")]}
    )


async def upsert_route(request: Request) -> JSONResponse:
    """Create or edit entities as one batch; 409 names the entity that holds a colliding value.

    With `dry_run`, every check runs and nothing is written."""
    try:
        items, author, method, dry_run = parse_upsert(await request_body(request))
    except RequestError as exc:
        return error(str(exc))
    if (denied := _denied(request.state.key, author, method)) is not None:
        return denied
    try:
        rows = await store.upsert(items, author, method, dry_run)
    except store.Collision as exc:
        return JSONResponse(
            {"error": str(exc), "holder": exc.holder, "value": exc.value}, status_code=409
        )
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse({"entities": [_public(row) for row in rows]})


async def archive_route(request: Request) -> JSONResponse:
    """Archive one active entity; `dry_run` checks without writing."""
    try:
        slug = _slug(request.path_params["slug"])
        fields = object_fields(
            await request_body(request), {"author", "consent_method"}, {"dry_run"}
        )
        author, method, dry_run = _consent(fields)
    except RequestError as exc:
        return error(str(exc))
    if (denied := _denied(request.state.key, author, method)) is not None:
        return denied
    try:
        row = await store.archive(slug, author, method, dry_run)
    except ValueError as exc:
        return error(str(exc))
    return JSONResponse({"entity": _public(row)})
