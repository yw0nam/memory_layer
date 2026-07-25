"""Shared Redis-backed durability mirror for job registries.

Not a source of truth: the in-memory registries stay the working state
(queue caps, semaphores). Redis only has to answer "what happened to job X"
after a restart. A store outage must degrade to today's in-memory-only
behaviour, never take a job down.
"""

from __future__ import annotations

import dataclasses
import json
import os

import redis.asyncio as redis_asyncio
from loguru import logger

JOB_TTL_SECONDS = 24 * 60 * 60
SOCKET_TIMEOUT_SECONDS = 3

_client: redis_asyncio.Redis | None = None
_client_initialized = False


def job_key(kind: str, job_id: str) -> str:
    return f"job:{kind}:{job_id}"


async def get_client() -> redis_asyncio.Redis | None:
    """Cached Redis client, or None if it could not be built."""
    global _client, _client_initialized
    if not _client_initialized:
        _client_initialized = True
        try:
            _client = redis_asyncio.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                decode_responses=True,
                socket_connect_timeout=SOCKET_TIMEOUT_SECONDS,
                socket_timeout=SOCKET_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            # broad catch: a store fault must never take a job down
            logger.warning("could not build redis client: {}", exc)
            _client = None
    return _client


async def save(kind: str, job: object) -> None:
    """Mirror a job to Redis with a TTL; never raises."""
    try:
        client = await get_client()
        if client is None:
            return
        payload = json.dumps(dataclasses.asdict(job))
        await client.set(job_key(kind, job.job_id), payload, ex=JOB_TTL_SECONDS)
    except Exception as exc:
        # broad catch: a store fault must never take a job down
        logger.warning("failed to persist {} job {}: {}", kind, job.job_id, exc)


async def load(kind: str, job_id: str, cls: type, terminal: frozenset[str]) -> object | None:
    """Read back a job; a non-terminal status means it died mid-flight."""
    try:
        client = await get_client()
        if client is None:
            return None
        raw = await client.get(job_key(kind, job_id))
        if raw is None:
            return None
        job = cls(**json.loads(raw))
    except Exception as exc:
        # broad catch: a store fault must never take a job down
        logger.warning("failed to load {} job {}: {}", kind, job_id, exc)
        return None
    if job.status not in terminal:
        # ponytail: single-process judgement — non-terminal and not in memory
        # means dead; with multiple workers this could misjudge a job running
        # on another worker, upgrade path is a short-TTL heartbeat key.
        job.error = "job state lost to a process restart"
        job.status = "failed"
    return job
