"""REST routes for key author allowlists and the namespace registry."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.serve.access import keys, namespaces
from memory_base.serve.common.http import error, json_body


async def authors_route(request: Request) -> JSONResponse:
    """Report a label's author allowlist; a non-admin key may read only its own label."""
    key = request.state.key
    label = request.path_params["label"]
    if not (key.is_admin or key.label == label):
        return error("not permitted to read another label's authors", 403)
    authors = await keys.get_authors(label)
    if authors is None:
        return JSONResponse({"error": f"unknown key label: {label}"}, status_code=404)
    return JSONResponse({"label": label, "authors": authors})


async def authors_put_route(request: Request) -> JSONResponse:
    """Replace a label's author allowlist; admin keys only."""
    if not request.state.key.is_admin:
        return error("admin key required", 403)
    label = request.path_params["label"]
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    try:
        authors = keys.validate_authors(body.get("authors"))
    except keys.AuthorError as exc:
        return error(str(exc))
    stored = await keys.set_authors(label, authors)
    if stored is None:
        return JSONResponse({"error": f"unknown key label: {label}"}, status_code=404)
    return JSONResponse({"label": label, "authors": stored})


async def namespaces_create_route(request: Request) -> JSONResponse:
    """Register a new namespace; 400 on a bad slug, 409 on a duplicate name.

    A private namespace records the caller's key label as owner.
    """
    key = request.state.key
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    visibility = body.get("visibility", "public")
    owner = key.label if visibility == "private" else None
    try:
        result = await namespaces.create_namespace(body.get("name"), visibility, owner)
    except namespaces.NamespaceExistsError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except namespaces.NamespaceError as exc:
        return error(str(exc))
    return JSONResponse(result, status_code=201)


async def namespaces_list_route(request: Request) -> JSONResponse:
    """List the caller's allowed namespaces (every namespace for an admin key)."""
    key = request.state.key
    rows = await namespaces.list_namespaces()
    if not key.is_admin:
        rows = [row for row in rows if row["name"] in key.allowed]
    return JSONResponse(rows)


async def namespaces_delete_route(request: Request) -> JSONResponse:
    """Unregister a namespace: 400 reserved, 404 unknown, 403 non-owner, 409 non-empty."""
    key = request.state.key
    name = request.path_params["name"]
    if name == namespaces.DEFAULT_NAMESPACE:
        return error("the 'default' namespace is reserved and cannot be deleted")
    ns = await namespaces.get_namespace(name)
    if ns is None:
        return JSONResponse({"error": f"unknown namespace: {name}"}, status_code=404)
    if not (key.is_admin or ns["owner"] == key.label):
        return error(f"not permitted to delete namespace: {name}", 403)
    try:
        await namespaces.delete_namespace(name)
    except namespaces.NamespaceReservedError as exc:
        return error(str(exc))
    except namespaces.NamespaceNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except namespaces.NamespaceNotEmptyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse({"deleted": name})
