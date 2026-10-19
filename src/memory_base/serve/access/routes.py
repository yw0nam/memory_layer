"""REST routes for the namespace registry."""

from __future__ import annotations

from starlette.requests import Request
from starlette.responses import JSONResponse

from memory_base.serve.access import namespaces
from memory_base.serve.common.http import error, json_body


async def namespaces_create_route(request: Request) -> JSONResponse:
    """Register a new namespace; 400 on a bad slug, 409 on a duplicate name."""
    try:
        body = await json_body(request)
    except Exception as exc:
        return error(f"invalid JSON body: {exc}")
    if "author" in body:
        return error("author comes from the X-Author header")
    try:
        result = await namespaces.create_namespace(body.get("name"))
    except namespaces.NamespaceExistsError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    except namespaces.NamespaceError as exc:
        return error(str(exc))
    return JSONResponse(result, status_code=201)


async def namespaces_list_route(request: Request) -> JSONResponse:
    """List every registered namespace."""
    del request
    return JSONResponse(await namespaces.list_namespaces())


async def namespaces_delete_route(request: Request) -> JSONResponse:
    """Unregister a namespace: 400 reserved, 404 unknown, 409 non-empty."""
    name = request.path_params["name"]
    try:
        await namespaces.delete_namespace(name)
    except namespaces.NamespaceReservedError as exc:
        return error(str(exc))
    except namespaces.NamespaceNotFoundError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except namespaces.NamespaceNotEmptyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse({"deleted": name})
