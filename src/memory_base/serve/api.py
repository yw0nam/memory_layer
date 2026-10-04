"""Starlette REST API for memory search and storage."""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route

from memory_base.core import db
from memory_base.core.logger import setup_logging
from memory_base.serve import health
from memory_base.serve.access import routes as access_routes
from memory_base.serve.access.auth import ApiKeyAuthMiddleware
from memory_base.serve.consolidation import routes as consolidation_routes
from memory_base.serve.documents import pipeline as document_pipeline
from memory_base.serve.documents import routes as document_routes
from memory_base.serve.messages import routes as message_routes
from memory_base.serve.notes import routes as note_routes
from memory_base.serve.profiles import routes as profile_routes
from memory_base.serve.repos import cache as repo_cache
from memory_base.serve.repos import routes as repo_routes
from memory_base.serve.search import access_log
from memory_base.serve.search import routes as search_routes
from memory_base.serve.tables import routes as table_routes

setup_logging()
logging.getLogger("uvicorn.access").addFilter(health.HealthAccessFilter())


# Background work as (start, stop) pairs: started in order, stopped in reverse.
BACKGROUND = (
    (document_pipeline.start, document_pipeline.stop),
    (repo_cache.start, repo_cache.stop),
    (access_log.start, access_log.stop),
)


@asynccontextmanager
async def lifespan(app: Starlette):
    """Start background work, stop what started in reverse, and close the pools last.

    A failing start() still stops everything started before it.
    """
    del app
    async with AsyncExitStack() as stack:
        stack.push_async_callback(db.close_pool)
        stack.push_async_callback(db.close_table_query_pool)
        for start, stop in BACKGROUND:
            stack.push_async_callback(stop, await start())
        yield


app = Starlette(
    lifespan=lifespan,
    middleware=[Middleware(ApiKeyAuthMiddleware)],
    routes=[
        Route("/health", health.health, methods=["GET"]),
        Route("/health/services", health.health_services, methods=["GET"]),
        Route("/search", search_routes.search_route, methods=["POST"]),
        Route("/save_memory", note_routes.save_route, methods=["POST"]),
        Route("/notes", note_routes.list_route, methods=["GET"]),
        Route("/profiles", profile_routes.profile_route, methods=["GET"]),
        Route("/profiles/self", profile_routes.self_route, methods=["PUT"]),
        Route("/profiles/versions", profile_routes.versions_route, methods=["GET"]),
        Route("/profiles/user/proposals", profile_routes.propose_route, methods=["POST"]),
        Route("/profiles/user/proposals", profile_routes.proposals_route, methods=["GET"]),
        Route(
            "/profiles/user/proposals/{proposal_id}", profile_routes.proposal_route, methods=["GET"]
        ),
        Route(
            "/profiles/user/proposals/{proposal_id}/approve",
            profile_routes.approve_route,
            methods=["POST"],
        ),
        Route(
            "/profiles/user/proposals/{proposal_id}/reject",
            profile_routes.reject_route,
            methods=["POST"],
        ),
        Route("/messages", message_routes.send_route, methods=["POST"]),
        Route("/messages", message_routes.list_route, methods=["GET"]),
        Route("/messages/{message_id}/claim", message_routes.claim_route, methods=["POST"]),
        Route("/messages/{message_id}", message_routes.cancel_route, methods=["DELETE"]),
        Route("/tables/query", table_routes.query_route, methods=["POST"]),
        Route("/ingest/document", document_routes.ingest_route, methods=["POST"]),
        Route("/ingest/jobs", document_routes.jobs_route, methods=["GET"]),
        Route("/ingest/jobs/{job_id}", document_routes.job_route, methods=["GET"]),
        Route("/ingest/documents/{document_id}", document_routes.remove_route, methods=["DELETE"]),
        Route("/repos", repo_routes.ingest_route, methods=["POST"]),
        Route("/repos", repo_routes.list_route, methods=["GET"]),
        Route("/repos/jobs/{job_id}", repo_routes.job_route, methods=["GET"]),
        Route("/repos/{name}", repo_routes.remove_route, methods=["DELETE"]),
        Route("/keys/{label}/authors", access_routes.authors_route, methods=["GET"]),
        Route("/keys/{label}/authors", access_routes.authors_put_route, methods=["PUT"]),
        Route("/namespaces", access_routes.namespaces_create_route, methods=["POST"]),
        Route("/namespaces", access_routes.namespaces_list_route, methods=["GET"]),
        Route("/namespaces/{name}", access_routes.namespaces_delete_route, methods=["DELETE"]),
        Route("/admin/notes", note_routes.old_notes_route, methods=["GET"]),
        Route("/admin/notes/delete", note_routes.delete_route, methods=["POST"]),
        Route("/admin/notes/move", note_routes.move_route, methods=["POST"]),
        Route("/admin/duplicates", note_routes.duplicates_route, methods=["GET"]),
        Route("/admin/consolidate/groups", consolidation_routes.groups_route, methods=["GET"]),
        Route("/admin/consolidate/verdicts", consolidation_routes.verdicts_route, methods=["POST"]),
        Route("/admin/consolidate/undo", consolidation_routes.undo_route, methods=["POST"]),
        Route("/admin/consolidate/actions", consolidation_routes.actions_route, methods=["GET"]),
        Route("/admin/archive", note_routes.archive_route, methods=["POST"]),
        Route("/admin/messages/purge", message_routes.purge_route, methods=["POST"]),
        Route("/admin/restore", note_routes.restore_route, methods=["POST"]),
    ],
)
