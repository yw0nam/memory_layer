"""Buffered persistence of retrieval activity, flushed off the request path."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Sequence

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.retrieval.search import Hit

LOGGER = logging.getLogger(__name__)
RETRIEVAL_LOG_RETENTION_DAYS = int(os.getenv("RETRIEVAL_LOG_RETENTION_DAYS", "180"))
HIT_FLUSH_INTERVAL_SECONDS = float(os.getenv("HIT_FLUSH_INTERVAL_SECONDS", "30"))
RETENTION_INTERVAL_SECONDS = 3600.0
# Bounds how many log rows a failed flush carries over, so a long DB outage cannot grow the buffer.
MAX_PENDING_LOGS = 10_000

# Counters feed lifecycle decisions only, so an unclean shutdown may lose one interval of them.
_pending_logs: list[tuple[str, str, list[str], float]] = []
_pending_hits: dict[str, tuple[int, float]] = {}
_last_retention = 0.0


def record_retrieval(
    query: str, source: str, hits: Sequence[Hit], now: float | None = None
) -> None:
    """Buffer a search's log row and per-chunk hit counts, touching no database."""
    if now is None:
        now = time.time()
    hit_ids = [hit.ref if hit.source == "code" else hit.meta.get("id", hit.ref) for hit in hits]
    _pending_logs.append((query, source, hit_ids, now))
    for hit in hits:
        if hit.source == "code":
            continue
        chunk_id = hit.meta.get("id")
        if not chunk_id:
            continue
        count, _ = _pending_hits.get(chunk_id, (0, now))
        _pending_hits[chunk_id] = (count + 1, now)


async def flush(now: float | None = None) -> None:
    """Persist the buffered rows in batched statements, pruning retention at most hourly."""
    global _last_retention
    if now is None:
        now = time.time()
    logs = _pending_logs[:]
    _pending_logs.clear()
    hits = dict(_pending_hits)
    _pending_hits.clear()
    retention_due = now - _last_retention >= RETENTION_INTERVAL_SECONDS
    if not logs and not hits and not retention_due:
        return
    try:
        async with db.acquire() as conn:
            if logs:
                await conn.executemany(
                    f'INSERT INTO "{PG_SCHEMA}".retrieval_log(query, source, hit_ids, ts) '
                    "VALUES($1,$2,$3,$4)",
                    logs,
                )
            if hits:
                await conn.execute(
                    f'UPDATE "{PG_SCHEMA}".memory_chunks AS m '
                    "SET hit_count = m.hit_count + v.hits, last_hit_at = v.ts "
                    "FROM unnest($1::text[], $2::bigint[], $3::double precision[]) "
                    "AS v(id, hits, ts) WHERE m.id = v.id",
                    list(hits),
                    [count for count, _ in hits.values()],
                    [ts for _, ts in hits.values()],
                )
            if retention_due:
                await conn.execute(
                    f'DELETE FROM "{PG_SCHEMA}".retrieval_log '
                    "WHERE ts < $1 - $2::double precision * 86400",
                    now,
                    RETRIEVAL_LOG_RETENTION_DAYS,
                )
                _last_retention = now
    except Exception:
        LOGGER.exception("failed to persist buffered retrieval access; retrying next interval")
        _pending_logs[:0] = logs
        del _pending_logs[:-MAX_PENDING_LOGS]
        for chunk_id, (count, ts) in hits.items():
            newer_count, newer_ts = _pending_hits.get(chunk_id, (0, ts))
            _pending_hits[chunk_id] = (count + newer_count, max(ts, newer_ts))


async def flusher_loop(stop: asyncio.Event | None = None) -> None:
    """Prune retention at startup, then flush the buffer on the configured interval."""
    stop = stop or asyncio.Event()
    await flush()
    while not stop.is_set():
        await asyncio.sleep(HIT_FLUSH_INTERVAL_SECONDS)
        await flush()


def start_flusher() -> asyncio.Task[None]:
    return asyncio.create_task(flusher_loop())


async def stop_flusher(task: asyncio.Task[None]) -> None:
    """Cancel the flusher, then persist whatever the buffer still holds."""
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await flush()
