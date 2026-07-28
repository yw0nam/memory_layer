"""Best-effort persistence of retrieval activity."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.retrieval.search import Hit

LOGGER = logging.getLogger(__name__)


async def log_retrieval(
    query: str, source: str, hits: Sequence[Hit], now: float | None = None
) -> None:
    """Record returned hit identifiers and update memory chunk access counters."""
    if now is None:
        now = time.time()
    hit_ids = [hit.ref if hit.source == "code" else hit.meta.get("id", hit.ref) for hit in hits]
    memory_ids = [hit.meta["id"] for hit in hits if hit.meta.get("id")]
    try:
        async with db.acquire() as conn:
            await conn.execute(
                f'INSERT INTO "{PG_SCHEMA}".retrieval_log(query, source, hit_ids, ts) '
                "VALUES($1,$2,$3,$4)",
                query,
                source,
                hit_ids,
                now,
            )
            if memory_ids:
                await conn.execute(
                    f'UPDATE "{PG_SCHEMA}".memory_chunks '
                    "SET hit_count = hit_count + 1, last_hit_at = $2 "
                    "WHERE id = ANY($1::text[])",
                    memory_ids,
                    now,
                )
    except Exception:
        LOGGER.exception("failed to record retrieval access")
