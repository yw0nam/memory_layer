"""Starlette REST API for memory search and storage."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route

from memory_base.core import db
from memory_base.core.logger import setup_logging
from memory_base.serve import health
from memory_base.serve.access import routes as access_routes
from memory_base.serve.access.auth import ApiKeyAuthMiddleware
from memory_base.serve.messages import routes as message_routes
from memory_base.serve.notes import routes as note_routes
from memory_base.serve.search import routes as search_routes

setup_logging()
logging.getLogger("uvicorn.access").addFilter(health.HealthAccessFilter())


@asynccontextmanager
async def lifespan(app: Starlette):
    """Close the pool on shutdown."""
    del app
    try:
        yield
    finally:
        await db.close_pool()


app = Starlette(
    lifespan=lifespan,
    middleware=[Middleware(ApiKeyAuthMiddleware)],
    routes=[
        Route("/health", health.health, methods=["GET"]),
        Route("/health/services", health.health_services, methods=["GET"]),
        Route("/search", search_routes.search_route, methods=["POST"]),
        Route("/save_memory", note_routes.save_route, methods=["POST"]),
        Route("/notes", note_routes.list_route, methods=["GET"]),
        Route("/messages", message_routes.send_route, methods=["POST"]),
        Route("/messages", message_routes.list_route, methods=["GET"]),
        Route("/messages/{message_id}/claim", message_routes.claim_route, methods=["POST"]),
        Route("/messages/{message_id}", message_routes.cancel_route, methods=["DELETE"]),
        Route("/namespaces", access_routes.namespaces_create_route, methods=["POST"]),
        Route("/namespaces", access_routes.namespaces_list_route, methods=["GET"]),
        Route("/namespaces/{name}", access_routes.namespaces_delete_route, methods=["DELETE"]),
        Route("/admin/notes/delete", note_routes.delete_route, methods=["POST"]),
        Route("/admin/archive", note_routes.archive_route, methods=["POST"]),
        Route("/admin/messages/purge", message_routes.purge_route, methods=["POST"]),
    ],
)
