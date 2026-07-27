"""Hybrid search: FTS + vector + time-decay signals fused with RRF, then reranked.

CLI: uv run python -m memory_base.retrieval.search "your query" [--source code|memory|all]
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

from memory_base.core.config import (
    DB_URL,
    OVERSAMPLE_FACTOR,
    PG_SCHEMA,
    RERANK_MODEL,
    SERVICE_TIMEOUT_SECONDS,
    VllmEmbedder,
    require_env,
    vector_literal,
)

RRF_K = 60
CANDIDATES_PER_SIGNAL = 50
PER_FILE_CAP = 3
FUSED_TOP = 20
RERANK_TOP = 10
TIME_DECAY_HALF_LIFE_DAYS = 90.0
ATOM_RETRIEVE_K = int(os.getenv("ATOM_RETRIEVE_K", "8"))
SEARCH_KINDS = ("doc", "note", "decision")
# Reranker input budget; the API's TEXT_LIMIT is separate and bounds only the response.
RERANK_TEXT_LIMIT = 4000
NEIGHBOR_LINE_WINDOW = 40
NEIGHBOR_LIMIT = 2


@dataclass
class Hit:
    source: str  # "code" | "memory"
    ref: str  # filename:lines or session id
    text: str  # text shown to reranker / synthesis
    ts: float  # recency timestamp (epoch sec)
    rrf: float = 0.0
    rerank_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf


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


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json

        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _normalize_tags(tags: Any) -> list[str] | None:
    if tags is None:
        return None
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("tags must be a non-empty list of strings")
    normalized = list(dict.fromkeys(tag.strip().lower() for tag in tags if tag.strip()))
    if not normalized:
        raise ValueError("tags must be a non-empty list of strings")
    return normalized


def validate_search_options(
    source: str,
    kind: Any,
    tags: Any,
) -> tuple[str | None, list[str] | None]:
    """Validate source-specific filters and return normalized values."""
    if kind is not None and kind not in SEARCH_KINDS:
        raise ValueError(f"kind must be one of {SEARCH_KINDS}")
    normalized_tags = _normalize_tags(tags)
    if (kind is not None or normalized_tags is not None) and source != "memory":
        raise ValueError('kind and tags filters require source="memory"')
    return kind, normalized_tags


def _history_predicates(
    *,
    include_archived: bool,
    kind: str | None,
    tags: list[str] | None,
    alias: str = "",
) -> tuple[str, list[Any]]:
    prefix = f"{alias}." if alias else ""
    clauses = [f"{prefix}chunk_kind <> 'atom'"]
    args: list[Any] = []
    if not include_archived:
        clauses.append(f"{prefix}archived_at IS NULL")
    if kind is not None:
        args.append(kind)
        clauses.append(f"{prefix}chunk_kind = ${len(args) + 1}")
    if tags is not None:
        args.append(tags)
        clauses.append(f"{prefix}metadata->'tags' ?| ${len(args) + 1}::text[]")
    return " AND ".join(clauses), args


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
                ref=f"{r['filename']}:L{r['start_line']}-L{r['end_line']}",
                text=r["code"],
                ts=r["mtime"],
                rrf=score,
                meta={"id": cid, "filename": r["filename"], "start_line": r["start_line"]},
            )
        )
    return hits


async def _search_memory(
    conn: asyncpg.Connection,
    query: str,
    qvec_lit: str,
    include_archived: bool = False,
    kind: str | None = None,
    tags: list[str] | None = None,
) -> list[Hit]:
    tbl = f'"{PG_SCHEMA}"."memory_chunks"'
    exists = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", f"{PG_SCHEMA}.memory_chunks")
    if not exists:
        return []
    predicates, filter_args = _history_predicates(
        include_archived=include_archived,
        kind=kind,
        tags=tags,
    )
    columns = (
        "id, source_ref, chunk_kind, metadata, distilled, content_raw, ts_last_active, idf_score"
    )
    vec_rows = await conn.fetch(
        f"SELECT {columns} FROM {tbl} WHERE {predicates} "
        f"ORDER BY embedding <=> $1::halfvec LIMIT {CANDIDATES_PER_SIGNAL}",
        qvec_lit,
        *filter_args,
    )
    fts_rows = await conn.fetch(
        f"SELECT {columns} FROM {tbl} WHERE {predicates} AND "
        "to_tsvector('simple', content_raw) @@ websearch_to_tsquery('simple', $1) "
        f"ORDER BY ts_rank_cd(to_tsvector('simple', content_raw), websearch_to_tsquery('simple', $1)) DESC "
        f"LIMIT {CANDIDATES_PER_SIGNAL}",
        query,
        *filter_args,
    )
    by_id = {r["id"]: r for r in [*vec_rows, *fts_rows]}
    recency = _time_decay_list({r["id"]: r["ts_last_active"] for r in by_id.values()})
    idf = _time_decay_list({r["id"]: r["idf_score"] or 0.0 for r in by_id.values()})
    scores = _rrf_fuse([[r["id"] for r in vec_rows], [r["id"] for r in fts_rows], recency, idf])

    hits = []
    for cid, score in scores.items():
        r = by_id[cid]
        metadata = _metadata_dict(r["metadata"])
        hits.append(
            Hit(
                source="memory",
                ref=metadata.get("search_ref") or r["source_ref"],
                text=r["distilled"] or r["content_raw"],
                ts=r["ts_last_active"],
                rrf=score,
                meta={
                    "id": cid,
                    "raw": r["content_raw"],
                    "kind": r["chunk_kind"],
                    "tags": metadata.get("tags", []),
                    "source_ref": r["source_ref"],
                },
            )
        )
    return hits


async def _search_atoms(
    conn: asyncpg.Connection,
    qvec_lit: str,
    include_archived: bool = False,
    kind: str | None = None,
    tags: list[str] | None = None,
) -> list[Hit]:
    tbl = f'"{PG_SCHEMA}"."memory_chunks"'
    predicates, filter_args = _history_predicates(
        include_archived=include_archived,
        kind=kind,
        tags=tags,
        alias="parent",
    )
    rows = await conn.fetch(
        f"""
        SELECT atom.id AS atom_id,
               coalesce(atom.distilled, atom.content_raw) AS matched_question,
               1 - (atom.embedding <=> $1::halfvec) AS atom_cosine,
               parent.id, parent.source_ref, parent.chunk_kind, parent.content_raw,
               parent.distilled, parent.ts_last_active, parent.metadata
        FROM {tbl} AS atom
        JOIN {tbl} AS parent ON parent.id = atom.metadata->>'parent_id'
        WHERE atom.chunk_kind = 'atom' AND {predicates}
        ORDER BY atom.embedding <=> $1::halfvec
        LIMIT {OVERSAMPLE_FACTOR * ATOM_RETRIEVE_K}
        """,
        qvec_lit,
        *filter_args,
    )
    best_by_parent: dict[str, Any] = {}
    for row in rows:
        current = best_by_parent.get(row["id"])
        if current is None or row["atom_cosine"] > current["atom_cosine"]:
            best_by_parent[row["id"]] = row
    collapsed = sorted(
        best_by_parent.values(),
        key=lambda row: row["atom_cosine"],
        reverse=True,
    )[:ATOM_RETRIEVE_K]
    hits = []
    for row in collapsed:
        metadata = _metadata_dict(row["metadata"])
        hits.append(
            Hit(
                source="memory",
                ref=metadata.get("search_ref") or row["source_ref"],
                text=row["distilled"] or row["content_raw"],
                ts=row["ts_last_active"],
                meta={
                    "id": row["id"],
                    "raw": row["content_raw"],
                    "kind": row["chunk_kind"],
                    "tags": metadata.get("tags", []),
                    "source_ref": row["source_ref"],
                    "atom_id": row["atom_id"],
                    "atom_question": row["matched_question"],
                    "atom_cosine": row["atom_cosine"],
                },
            )
        )
    return hits


def _merge_atom_hits(baseline: list[Hit], atom_hits: list[Hit]) -> list[Hit]:
    baseline_by_id = {hit.meta.get("id"): hit for hit in baseline if hit.meta.get("id")}
    new_hits = []
    for atom_hit in atom_hits:
        existing = baseline_by_id.get(atom_hit.meta["id"])
        if existing is None:
            new_hits.append(atom_hit)
            continue
        for key in ("atom_id", "atom_question", "atom_cosine"):
            existing.meta[key] = atom_hit.meta[key]
    return [*baseline, *new_hits]


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
        key = (
            h.meta.get("source_ref", h.ref)
            if h.source == "memory"
            else h.meta.get("filename") or h.ref
        )
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
    async with httpx.AsyncClient(timeout=SERVICE_TIMEOUT_SECONDS) as client:
        r = await client.post(
            require_env("RERANK_URL").rstrip("/") + "/rerank",
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": [h.text[:RERANK_TEXT_LIMIT] for h in hits],
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
            f"AND id <> $2 AND abs(start_line - $3) <= {NEIGHBOR_LINE_WINDOW} "
            f"ORDER BY start_line LIMIT {NEIGHBOR_LIMIT}",
            h.meta["filename"],
            h.meta["id"],
            h.meta["start_line"],
        )
        if rows:
            h.meta["context"] = "\n...\n".join(r["code"] for r in rows)


async def search(
    query: str,
    source: str = "all",
    rerank: bool = True,
    include_archived: bool = False,
    kind: str | None = None,
    tags: list[str] | None = None,
    include_atoms: bool | None = None,
) -> list[Hit]:
    kind, tags = validate_search_options(source, kind, tags)
    if include_atoms is not None and not isinstance(include_atoms, bool):
        raise ValueError("include_atoms must be a boolean")
    if include_atoms is None:
        include_atoms = os.getenv("ATOMS_RETRIEVE", "true").lower() in ("1", "true", "yes", "on")

    embedder = VllmEmbedder()
    qvec = await embedder.embed(query, query=True)
    qvec_lit = vector_literal(qvec)

    conn = await asyncpg.connect(DB_URL)
    try:
        hits: list[Hit] = []
        if source in ("code", "all"):
            hits += await _search_code(conn, query, qvec_lit)
        if source in ("memory", "all"):
            hits += await _search_memory(
                conn,
                query,
                qvec_lit,
                include_archived=include_archived,
                kind=kind,
                tags=tags,
            )
        if not include_archived:
            # Archival recall asks for old rows; recency decay would bury them.
            _apply_time_decay(hits)
        hits = _dedup_cap(hits)
        if include_atoms and source in ("memory", "all"):
            atom_hits = await _search_atoms(
                conn,
                qvec_lit,
                include_archived=include_archived,
                kind=kind,
                tags=tags,
            )
            hits = _merge_atom_hits(hits, atom_hits)
        if rerank:
            hits = await _rerank(query, hits)
        await _restore_context(conn, hits)
        return hits
    finally:
        await conn.close()


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--source", choices=["code", "memory", "all"], default="all")
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()

    for h in await search(args.query, source=args.source, rerank=not args.no_rerank):
        print(f"[{h.score:.4f}] ({h.source}) {h.ref}")
        print("    " + h.text[:300].replace("\n", "\n    "))
        print("---")


if __name__ == "__main__":
    asyncio.run(_main())
