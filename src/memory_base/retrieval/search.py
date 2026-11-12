"""Hybrid search over notes: FTS + vector + recency signals fused with RRF, then reranked."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import asyncpg

from memory_base.core import db
from memory_base.core.config import (
    PG_SCHEMA,
    QUERY_TIMEOUT_SECONDS,
    RERANK_CONNECT_TIMEOUT_SECONDS,
    VllmEmbedder,
    rerank_model,
    require_env,
    vector_literal,
)

RRF_K = 60
CANDIDATES_PER_SIGNAL = 50
FUSED_TOP = 20
BUDGET_FUSED_TOP = 40
RERANK_TOP = 10
SEARCH_KINDS = ("personal", "work")
RERANK_TEXT_LIMIT = 4000
MEMORY_BM25_INDEX = "memory_chunks_bm25"
# Reranker relevance floor (0..1, rerank-only): above junk (<=~0.01), below weak true hits (~0.29).
MIN_SCORE = 0.25
FTS_RRF_WEIGHT = 0.2
# Recency is a tie-breaker ("when relevance is otherwise equal, the newer wins").
TIEBREAK_RRF_WEIGHT = 0.25
# BM25 index scans cap candidates at bm25_catalog.bm25_limit (100 by default) before filters.

# vLLM's official Qwen3 reranker example; byte-exact.
QWEN3_RERANK_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN3_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN3_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class UpstreamUnavailable(RuntimeError):
    """A vLLM endpoint the query path depends on is unreachable."""

    def __init__(self, service: str) -> None:
        super().__init__(f"the {service} service is not responding")
        self.service = service


@dataclass
class Hit:
    ref: str  # note ref
    text: str  # text shown to reranker / synthesis
    ts: float  # recency timestamp (epoch sec)
    rrf: float = 0.0
    rerank_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf


def rrf_fuse(lists: list[list[Any]], weights: list[float] | None = None) -> dict[Any, float]:
    """Fuse ranked lists with score(d) = sum(weight / (k + rank))."""
    if weights is None:
        weights = [1.0] * len(lists)
    scores: dict[Any, float] = {}
    for ranked, weight in zip(lists, weights, strict=True):
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank)
    return scores


def time_decay_list(rows: dict[Any, float]) -> list[Any]:
    """Candidates ranked purely by recency — one more voter in the RRF fusion."""
    return [k for k, _ in sorted(rows.items(), key=lambda kv: kv[1], reverse=True)]


def metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    return {}


def normalize_tags(tags: Any, *, allow_empty: bool = False) -> list[str] | None:
    """Validate and normalize a tags list: strip, lowercase, dedupe.

    Search filters (allow_empty=False, the default) treat an absent value as "no
    filter" (None) and reject an explicit empty or all-blank list as ambiguous with
    that. Stored data (allow_empty=True, e.g. note tags) has no such
    ambiguity: an absent or empty value both normalize to "no tags" ([]).
    """
    if tags is None:
        return [] if allow_empty else None
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        msg = (
            "tags must be a list of strings"
            if allow_empty
            else "tags must be a non-empty list of strings"
        )
        raise ValueError(msg)
    normalized = list(dict.fromkeys(tag.strip().lower() for tag in tags if tag.strip()))
    if not normalized and not allow_empty:
        raise ValueError("tags must be a non-empty list of strings")
    return normalized


def normalize_namespaces(namespaces: Any) -> list[str] | None:
    """Normalize an optional namespace filter; omitted or empty means every namespace."""
    if namespaces is None:
        return None
    if not isinstance(namespaces, list) or any(not isinstance(ns, str) for ns in namespaces):
        raise ValueError("namespaces must be a list of strings")
    normalized = list(dict.fromkeys(ns.strip() for ns in namespaces if ns.strip()))
    return normalized or None


def parse_time_bound(value: Any, *, end: bool = False) -> float:
    """Epoch seconds for an ISO 8601 date or datetime; naive values are read as UTC.

    A bare date names a whole day, so as an end bound it resolves to the next
    midnight; a datetime is the exact instant either way.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("since/until must be ISO 8601 date or datetime strings")
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"invalid ISO 8601 timestamp: {value!r}") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    is_bare_date = True
    try:
        date.fromisoformat(text)
    except ValueError:
        is_bare_date = False
    epoch = parsed.timestamp()
    return epoch + 86400.0 if end and is_bare_date else epoch


def normalize_time_range(since: Any, until: Any) -> tuple[float | None, float | None]:
    """Epoch bounds for the optional since/until filter; omitted bounds stay None."""
    since_ts = parse_time_bound(since) if since is not None else None
    until_ts = parse_time_bound(until, end=True) if until is not None else None
    if since_ts is not None and until_ts is not None and since_ts >= until_ts:
        raise ValueError("since must be earlier than until")
    return since_ts, until_ts


def validate_search_options(
    kind: Any,
    tags: Any,
    since: Any = None,
    until: Any = None,
    authors: Any = None,
) -> tuple[str | None, list[str] | None, float | None, float | None]:
    """Validate the filters and return normalized kind, tags, and time bounds.

    `authors` needs no normalization, so it is validated here but not returned.
    """
    if kind is not None and kind not in SEARCH_KINDS:
        raise ValueError(f"kind must be one of {SEARCH_KINDS}")
    normalized_tags = normalize_tags(tags)
    since_ts, until_ts = normalize_time_range(since, until)
    if authors is not None and (
        not isinstance(authors, list)
        or not authors
        or not all(isinstance(a, str) and a.strip() for a in authors)
    ):
        raise ValueError("authors must be a non-empty list of non-empty strings")
    return kind, normalized_tags, since_ts, until_ts


def visible_clause(alias: str, index: int) -> str:
    """The clause hiding another author's private notes, `$index` being the reader."""
    prefix = f"{alias}." if alias else ""
    return (
        f"({prefix}source_type <> 'agent_note' "
        f"OR {prefix}metadata->>'author' = ${index} "
        f"OR {prefix}metadata @> '{{\"shared\": true}}'::jsonb)"
    )


def history_predicates(
    *,
    include_archived: bool,
    kind: str | None,
    tags: list[str] | None,
    alias: str = "",
    namespaces: list[str] | None = None,
    since: float | None = None,
    until: float | None = None,
    authors: list[str] | None = None,
    visible_to: str | None = None,
    start: int = 1,
) -> tuple[str, list[Any]]:
    """Filter clauses over memory_chunks; placeholders start after `$start`.

    `start` is the index of the caller's last already-bound argument ($1 by default).
    since/until bound when the remembered event happened, falling back to the save time.
    `visible_to` limits agent notes to that author's own and shared ones.
    """
    prefix = f"{alias}." if alias else ""
    event_time = f"COALESCE({prefix}occurred_at, {prefix}ts_last_active)"
    clauses: list[str] = []
    args: list[Any] = []
    if not include_archived:
        clauses.append(f"{prefix}archived_at IS NULL")
    if kind is not None:
        args.append(kind)
        clauses.append(f"{prefix}chunk_kind = ${start + len(args)}")
    if tags is not None:
        args.append(tags)
        clauses.append(f"{prefix}metadata->'tags' ?| ${start + len(args)}::text[]")
    if namespaces:
        args.append(namespaces)
        clauses.append(f"{prefix}namespace = ANY(${start + len(args)}::text[])")
    if since is not None:
        args.append(since)
        clauses.append(f"{event_time} >= ${start + len(args)}")
    if until is not None:
        args.append(until)
        clauses.append(f"{event_time} < ${start + len(args)}")
    if authors is not None:
        args.append(authors)
        clauses.append(f"{prefix}metadata->>'author' = ANY(${start + len(args)}::text[])")
    if visible_to is not None:
        args.append(visible_to)
        clauses.append(visible_clause(alias, start + len(args)))
    return " AND ".join(clauses) or "true", args


async def _search_memory(
    conn: asyncpg.Connection,
    query: str,
    qvec_lit: str,
    include_archived: bool = False,
    kind: str | None = None,
    tags: list[str] | None = None,
    namespaces: list[str] | None = None,
    since: float | None = None,
    until: float | None = None,
    authors: list[str] | None = None,
    visible_to: str | None = None,
) -> list[Hit]:
    tbl = f'"{PG_SCHEMA}"."memory_chunks"'
    bm25_index = f"{PG_SCHEMA}.{MEMORY_BM25_INDEX}"
    predicates, filter_args = history_predicates(
        include_archived=include_archived,
        kind=kind,
        tags=tags,
        namespaces=namespaces,
        since=since,
        until=until,
        authors=authors,
        visible_to=visible_to,
    )
    columns = (
        "id, source_ref, chunk_kind, metadata, distilled, content_raw, ts_last_active, "
        "archived_at, occurred_at"
    )
    try:
        vec_rows = await conn.fetch(
            f"SELECT {columns} FROM {tbl} WHERE {predicates} "
            f"ORDER BY embedding <=> $1::halfvec LIMIT {CANDIDATES_PER_SIGNAL}",
            qvec_lit,
            *filter_args,
        )
        fts_rows = await conn.fetch(
            f"SELECT {columns} FROM {tbl} WHERE {predicates} AND "
            f"(content_raw <@> to_bm25query($1, '{bm25_index}')) < 0 "
            f"ORDER BY content_raw <@> to_bm25query($1, '{bm25_index}') "
            f"LIMIT {CANDIDATES_PER_SIGNAL}",
            query,
            *filter_args,
        )
    except asyncpg.exceptions.UndefinedTableError:
        # ensure_schema creates memory_chunks up front; missing means an unprovisioned schema.
        return []
    by_id = {r["id"]: r for r in [*vec_rows, *fts_rows]}
    recency = time_decay_list({r["id"]: r["ts_last_active"] for r in by_id.values()})
    scores = rrf_fuse(
        [[r["id"] for r in vec_rows], [r["id"] for r in fts_rows], recency],
        [1.0, FTS_RRF_WEIGHT, TIEBREAK_RRF_WEIGHT],
    )

    hits = []
    for cid, score in scores.items():
        r = by_id[cid]
        metadata = metadata_dict(r["metadata"])
        hit_meta = {
            "id": cid,
            "raw": r["content_raw"],
            "kind": r["chunk_kind"],
            "tags": metadata.get("tags", []),
            "author": metadata.get("author"),
            "shared": metadata.get("shared") is True,
            "supersedes": metadata.get("supersedes"),
            "archived": r["archived_at"] is not None,
            "occurred_at": r["occurred_at"],
        }
        hits.append(
            Hit(
                ref=r["source_ref"],
                text=r["distilled"] or r["content_raw"],
                ts=r["ts_last_active"],
                rrf=score,
                meta=hit_meta,
            )
        )
    return hits


def _apply_min_score(hits: list[Hit], min_score: float | None, rerank: bool) -> list[Hit]:
    """Drop hits scoring below the floor; None resolves to MIN_SCORE only when rerank ran."""
    floor = min_score if min_score is not None else (MIN_SCORE if rerank else None)
    if floor is None:
        return hits
    return [h for h in hits if h.score >= floor]


def estimate_tokens(hit: Hit) -> int:
    """Token estimate of what a hit returns: its text."""
    # ponytail: chars/4, no tokenizer dependency.
    return max(1, len(hit.text) // 4)


def _pack_budget(hits: list[Hit], budget_tokens: int) -> list[Hit]:
    """Hits in rank order until the next one would push the estimate past the budget."""
    packed: list[Hit] = []
    used = 0
    for h in hits:
        used += estimate_tokens(h)
        if used > budget_tokens:
            break
        packed.append(h)
    return packed


def rerank_payload(model: str, query: str, texts: list[str]) -> dict:
    """Build the /rerank request body, templated for Qwen3 rerankers."""
    truncated = [text[:RERANK_TEXT_LIMIT] for text in texts]
    if "qwen3-reranker" not in model.lower():
        return {"model": model, "query": query, "documents": truncated}
    templated_query = (
        f"{QWEN3_RERANK_PREFIX}<Instruct>: {QWEN3_RERANK_INSTRUCTION}\n<Query>: {query}\n"
    )
    templated_docs = [f"<Document>: {text}{QWEN3_RERANK_SUFFIX}" for text in truncated]
    return {"model": model, "query": templated_query, "documents": templated_docs}


async def _embed_query(query: str) -> str:
    """Embed the query as a vector literal, or report the embedder as unavailable."""
    import openai

    try:
        vector = await asyncio.wait_for(
            VllmEmbedder().embed(query, query=True), timeout=QUERY_TIMEOUT_SECONDS
        )
    except (TimeoutError, openai.APIError) as exc:
        raise UpstreamUnavailable("embedding") from exc
    return vector_literal(vector)


async def _rerank(query: str, hits: list[Hit]) -> list[Hit]:
    import httpx

    if not hits:
        return hits
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(QUERY_TIMEOUT_SECONDS, connect=RERANK_CONNECT_TIMEOUT_SECONDS),
        transport=httpx.AsyncHTTPTransport(retries=1),
    ) as client:
        try:
            r = await client.post(
                require_env("RERANK_URL").rstrip("/") + "/rerank",
                json=rerank_payload(rerank_model(), query, [h.text for h in hits]),
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise UpstreamUnavailable("reranking") from exc
    for item in r.json()["results"]:
        hits[item["index"]].rerank_score = item["relevance_score"]
    hits.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
    return hits


async def search(
    query: str,
    rerank: bool = True,
    include_archived: bool = False,
    kind: str | None = None,
    tags: list[str] | None = None,
    namespaces: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    min_score: float | None = None,
    authors: list[str] | None = None,
    visible_to: str | None = None,
    budget_tokens: int | None = None,
) -> list[Hit]:
    """Return ranked note hits.

    Without budget_tokens: the top RERANK_TOP of FUSED_TOP reranked candidates, above the
    score floor.
    With it: BUDGET_FUSED_TOP candidates in reranked order until the token estimate
    of the returned text would exceed the budget; the floor and RERANK_TOP do not apply.
    """
    kind, tags, since_ts, until_ts = validate_search_options(kind, tags, since, until, authors)
    namespaces = normalize_namespaces(namespaces)

    qvec_lit = await _embed_query(query)

    async with db.acquire() as conn:
        hits = await _search_memory(
            conn,
            query,
            qvec_lit,
            include_archived=include_archived,
            kind=kind,
            tags=tags,
            namespaces=namespaces,
            since=since_ts,
            until=until_ts,
            authors=authors,
            visible_to=visible_to,
        )
    hits = sorted(hits, key=lambda h: h.rrf, reverse=True)[
        : FUSED_TOP if budget_tokens is None else BUDGET_FUSED_TOP
    ]
    if rerank:
        hits = await _rerank(query, hits)
    if budget_tokens is None:
        hits = _apply_min_score(hits[:RERANK_TOP] if rerank else hits, min_score, rerank)
    else:
        hits = _pack_budget(hits, budget_tokens)
    return hits
