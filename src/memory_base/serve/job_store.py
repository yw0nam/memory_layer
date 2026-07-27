"""Shared job registry and Redis-backed durability mirror.

Not a source of truth: the in-memory registry stays the working state
(queue caps, semaphores). Redis only has to answer "what happened to job X"
after a restart. A store outage must degrade to today's in-memory-only
behaviour, never take a job down.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis_asyncio
from loguru import logger

JOB_TTL_SECONDS = 24 * 60 * 60
MAX_COMPLETED_JOBS = 100
SOCKET_TIMEOUT_SECONDS = 3

_client: redis_asyncio.Redis | None = None
_client_initialized = False


def _iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


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
    """Read back a job; a non-terminal status means no outcome was recorded."""
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
        # ponytail: single-process judgement can misread a job running on another worker
        job.error = "no terminal state was recorded (a restart or a lost final write)"
        job.status = "failed"
    return job


class JobRegistry:
    """Bounded job state for one job kind, run under a concurrency semaphore."""

    def __init__(
        self,
        *,
        kind: str,
        job_cls: type,
        terminal_statuses: frozenset[str],
        queue_full_message: str,
        max_queued: int,
        max_concurrent: int,
        ttl_seconds: int = JOB_TTL_SECONDS,
        max_completed: int = MAX_COMPLETED_JOBS,
    ) -> None:
        self.kind = kind
        self.job_cls = job_cls
        self.terminal_statuses = terminal_statuses
        self.queue_full_message = queue_full_message
        self.max_queued = max_queued
        self.ttl_seconds = ttl_seconds
        self.max_completed = max_completed
        self.jobs: OrderedDict[str, Any] = OrderedDict()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: set[asyncio.Task[None]] = set()

    def cleanup(self, now: float | None = None) -> None:
        current = time.time() if now is None else now
        expired = [
            job_id
            for job_id, job in self.jobs.items()
            if job.status in self.terminal_statuses and current - job.updated_at >= self.ttl_seconds
        ]
        for job_id in expired:
            self.jobs.pop(job_id, None)

        completed = [
            job_id for job_id, job in self.jobs.items() if job.status in self.terminal_statuses
        ]
        for job_id in completed[: max(0, len(completed) - self.max_completed)]:
            self.jobs.pop(job_id, None)

    def queued_count(self) -> int:
        return sum(job.status == "queued" for job in self.jobs.values())

    def has_capacity(self) -> bool:
        self.cleanup()
        return self.queued_count() < self.max_queued

    def register(self, job: Any) -> None:
        """Run the shared capacity check and store an already-built job."""
        self.cleanup()
        if not self.has_capacity():
            raise OverflowError(self.queue_full_message)
        self.jobs[job.job_id] = job

    def start(self, job: Any, runner: Callable[[], Awaitable[None]]) -> None:
        async def run() -> None:
            await save(self.kind, job)
            async with self._semaphore:
                job.mark_running()
                await save(self.kind, job)
                try:
                    await runner()
                    job.mark_succeeded()
                except Exception as exc:
                    job.error = str(exc) or type(exc).__name__
                    job.mark_failed()
                finally:
                    await save(self.kind, job)
                    self.cleanup()

        task = asyncio.create_task(run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def get(self, job_id: str) -> Any | None:
        self.cleanup()
        job = self.jobs.get(job_id)
        if job is not None:
            return job
        return await load(self.kind, job_id, self.job_cls, self.terminal_statuses)
