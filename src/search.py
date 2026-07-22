"""Hybrid search: FTS + vector + time-decay signals fused with RRF, then reranked.

CLI: uv run python src/search.py "your query" [--source code|history|all]
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import asyncpg
from openai import AsyncOpenAI

from common import DB_URL, PG_SCHEMA, RERANK_MODEL, VllmEmbedder

RRF_K = 60
CANDIDATES_PER_SIGNAL = 50
PER_FILE_CAP = 3
FUSED_TOP = 20
RERANK_TOP = 10
TIME_DECAY_HALF_LIFE_DAYS = 90.0


@dataclass
class Hit:
    source: str  # "code" | "history"
    ref: str  # filename:lines or session id
    text: str  # text shown to reranker / synthesis
    ts: float  # recency timestamp (epoch sec)
    rrf: float = 0.0
    rerank_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _rrf_fuse(lists: list[list[Any]]) -> dict[Any, float]:
    """score(d) = sum over lists of 1/(k + rank)."""
    scores: dict[Any, float] = {}
    for ranked in lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def _time_decay_list(rows: dict[Any, float]) -> list[Any]:
    """Candidates ranked purely by recency — one more voter in the RRF fusion."""
    return [k for k, _ in sorted(rows.items(), key=lambda kv: kv[1], reverse=True)]


async def _search_code(conn: asyncpg.Connection, query: str, qvec_lit: str) -> list[Hit]:
    tbl = f'"{PG_SCHEMA}"."code_chunks"'
    vec_rows = await conn.fetch(
        f"SELECT id, filename, code, start_line, end_line, mtime "
        f"FROM {tbl} ORDER BY embedding <=> $1::halfvec LIMIT {CANDIDATES_PER_SIGNAL}",
        qvec_lit,
    )
    fts_rows = await conn.fetch(
        f"SELECT id, filename, code, start_line, end_line, mtime FROM {tbl} "
        f"WHERE to_tsvector('simple', code) @@ websearch_to_tsquery('simple', $1) "
        f"ORDER BY ts_rank_cd(to_tsvector('simple', code), websearch_to_tsquery('simple', $1)) DESC "
        f"LIMIT {CANDIDATES_PER_SIGNAL}",
        query,
    )
    by_id = {r["id"]: r for r in [*vec_rows, *fts_rows]}
    recency = _time_decay_list({r["id"]: r["mtime"] for r in by_id.values()})
    scores = _rrf_fuse([[r["id"] for r in vec_rows], [r["id"] for r in fts_rows], recency])

    hits = []
    for cid, score in scores.items():
        r = by_id[cid]
        hits.append(
            Hit(
                source="code",
                ref=f'{r["filename"]}:L{r["start_line"]}-L{r["end_line"]}',
                text=r["code"],
                ts=r["mtime"],
                rrf=score,
                meta={"id": cid, "filename": r["filename"], "start_line": r["start_line"]},
            )
        )
    return hits


async def _search_history(conn: asyncpg.Connection, query: str, qvec_lit: str) -> list[Hit]:
    tbl = f'"{PG_SCHEMA}"."memory_chunks"'
    exists = await conn.fetchval(
        "SELECT to_regclass($1) IS NOT NULL", f"{PG_SCHEMA}.memory_chunks"
    )
    if not exists:
        return []
    vec_rows = await conn.fetch(
        f"SELECT id, source_ref, distilled, content_raw, ts_last_active, idf_score "
        f"FROM {tbl} ORDER BY embedding <=> $1::halfvec LIMIT {CANDIDATES_PER_SIGNAL}",
        qvec_lit,
    )
    fts_rows = await conn.fetch(
        f"SELECT id, source_ref, distilled, content_raw, ts_last_active, idf_score FROM {tbl} "
        f"WHERE to_tsvector('simple', content_raw) @@ websearch_to_tsquery('simple', $1) "
        f"ORDER BY ts_rank_cd(to_tsvector('simple', content_raw), websearch_to_tsquery('simple', $1)) DESC "
        f"LIMIT {CANDIDATES_PER_SIGNAL}",
        query,
    )
    by_id = {r["id"]: r for r in [*vec_rows, *fts_rows]}
    recency = _time_decay_list({r["id"]: r["ts_last_active"] for r in by_id.values()})
    idf = _time_decay_list({r["id"]: r["idf_score"] or 0.0 for r in by_id.values()})
    scores = _rrf_fuse(
        [[r["id"] for r in vec_rows], [r["id"] for r in fts_rows], recency, idf]
    )

    hits = []
    for cid, score in scores.items():
        r = by_id[cid]
        hits.append(
            Hit(
                source="history",
                ref=r["source_ref"],
                text=r["distilled"] or r["content_raw"],
                ts=r["ts_last_active"],
                rrf=score,
                meta={"id": cid, "raw": r["content_raw"]},
            )
        )
    return hits


def _apply_time_decay(hits: list[Hit]) -> None:
    """Multiply RRF score by exp decay on age — old answers lose to fresh ones."""
    now = time.time()
    for h in hits:
        age_days = max(0.0, (now - h.ts) / 86400.0)
        h.rrf *= math.pow(0.5, age_days / TIME_DECAY_HALF_LIFE_DAYS)


def _dedup_cap(hits: list[Hit]) -> list[Hit]:
    """Per-file/session cap for diversity, then take fused top."""
    hits.sort(key=lambda h: h.rrf, reverse=True)
    counts: dict[str, int] = {}
    out = []
    for h in hits:
        key = h.meta.get("filename") or h.ref
        if counts.get(key, 0) >= PER_FILE_CAP:
            continue
        counts[key] = counts.get(key, 0) + 1
        out.append(h)
        if len(out) >= FUSED_TOP:
            break
    return out


async def _rerank(query: str, hits: list[Hit]) -> list[Hit]:
    import httpx

    if not hits:
        return hits
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            os.environ["RERANK_URL"].rstrip("/") + "/rerank",
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": [h.text[:4000] for h in hits],
            },
        )
        r.raise_for_status()
    for item in r.json()["results"]:
        hits[item["index"]].rerank_score = item["relevance_score"]
    hits.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
    return hits[:RERANK_TOP]


async def _restore_context(conn: asyncpg.Connection, hits: list[Hit]) -> None:
    """Re-attach neighboring code chunks so cut-off premises come back."""
    tbl = f'"{PG_SCHEMA}"."code_chunks"'
    for h in hits:
        if h.source != "code":
            continue
        rows = await conn.fetch(
            f"SELECT code, start_line FROM {tbl} WHERE filename = $1 "
            f"AND id <> $2 AND abs(start_line - $3) <= 40 ORDER BY start_line LIMIT 2",
            h.meta["filename"],
            h.meta["id"],
            h.meta["start_line"],
        )
        if rows:
            h.meta["context"] = "\n...\n".join(r["code"] for r in rows)


async def search(query: str, source: str = "all", rerank: bool = True) -> list[Hit]:
    embedder = VllmEmbedder()
    qvec = await embedder.embed(query, query=True)
    qvec_lit = "[" + ",".join(f"{x:.6f}" for x in qvec.astype(float)) + "]"

    conn = await asyncpg.connect(DB_URL)
    try:
        hits: list[Hit] = []
        if source in ("code", "all"):
            hits += await _search_code(conn, query, qvec_lit)
        if source in ("history", "all"):
            hits += await _search_history(conn, query, qvec_lit)
        _apply_time_decay(hits)
        hits = _dedup_cap(hits)
        if rerank:
            hits = await _rerank(query, hits)
        await _restore_context(conn, hits)
        return hits
    finally:
        await conn.close()


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--source", choices=["code", "history", "all"], default="all")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    for h in await search(args.query, source=args.source, rerank=not args.no_rerank):
        score = h.rerank_score if h.rerank_score is not None else h.rrf
        print(f"[{score:.4f}] ({h.source}) {h.ref}")
        print("    " + h.text[:300].replace("\n", "\n    "))
        print("---")


if __name__ == "__main__":
    asyncio.run(_main())
