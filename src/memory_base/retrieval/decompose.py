"""Knowledge-aware decomposition: multi-hop retrieval over memory atoms."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import asyncpg

from memory_base.common import (
    DB_URL,
    LLM_MODEL,
    OVERSAMPLE_FACTOR,
    PG_SCHEMA,
    SERVICE_TIMEOUT_SECONDS,
    VllmEmbedder,
    llm_client,
    vector_literal,
)
from memory_base.retrieval.search import (
    CANDIDATES_PER_SIGNAL,
    PER_FILE_CAP,
    TIME_DECAY_HALF_LIFE_DAYS,
    _history_predicates,
    _metadata_dict,
    _rrf_fuse,
    _time_decay_list,
)

DEEP_MAX_HOPS = int(os.getenv("DEEP_MAX_HOPS", "3"))
DEEP_TIMEOUT_SECONDS = int(os.getenv("DEEP_TIMEOUT_SECONDS", "180"))
HOP_CANDIDATE_CAP = 8
SUB_QUESTION_CAP = 3
PARENT_SNIPPET_LEN = 500


@dataclass
class EvidenceEntry:
    id: str
    ref: str
    text: str
    kind: str
    tags: list[str]
    date: float
    hop: int
    atom_question: str | None


@dataclass
class TraceEntry:
    hop: int
    sub_questions: list[str]
    selected_ref: str | None


@dataclass
class DeepResult:
    evidence: list[EvidenceEntry]
    trace: list[TraceEntry]
    hops_used: int
    stopped_reason: str


@dataclass
class _Candidate:
    parent_id: str
    ref: str
    text: str
    ts: float
    kind: str
    tags: list[str]
    source_ref: str
    atom_question: str | None


class _LLMError(Exception):
    pass


class _Timeout(Exception):
    pass


_PROPOSE_SYSTEM = "You are a helpful AI assistant good at question decomposition."

_PROPOSE_USER = """\
# Task
Analyse the providing context then raise atomic sub-questions for the knowledge \
that can help you answer the question better. Think in different ways and raise as \
many diverse questions as possible. At most {max_sub} sub-questions.

# Output Format
Output JSON with keys:
- "continue": boolean, true if more evidence is needed. Set it to false only when \
the evidence already covers every entity and fact the question asks about; if any \
part of the question is still uncovered, set it to true.
- "sub_questions": list of up to {max_sub} English sub-questions

# Context (evidence gathered so far)
{evidence_text}

# Question
{question}"""

_SELECT_SYSTEM = "You are a helpful AI assistant on question answering."

_SELECT_USER = """\
# Task
Analyse the providing context then decide which candidate may be useful to answer \
the question. Prefer the candidate that fills a part of the question the context \
does not yet cover. Return null only when no candidate adds any missing \
information.

# Output Format
Output JSON with key:
- "selected": 1-based integer index, or null if none are relevant

# Context (evidence gathered so far)
{evidence_text}

# Candidates
{candidate_list}

# Question
{question}"""


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _call_timeout(deadline: float) -> float:
    return min(SERVICE_TIMEOUT_SECONDS, _remaining(deadline))


def _evidence_text(evidence: list[EvidenceEntry]) -> str:
    if not evidence:
        return "(none)"
    parts = []
    for i, entry in enumerate(evidence, 1):
        parts.append(f"[{i}] (hop {entry.hop}) {entry.text[:PARENT_SNIPPET_LEN]}")
    return "\n".join(parts)


def _candidate_list(candidates: list[_Candidate]) -> str:
    parts = []
    for i, c in enumerate(candidates, 1):
        snippet = c.text[:PARENT_SNIPPET_LEN]
        if c.atom_question:
            parts.append(f"Candidate {i}: Atom question: {c.atom_question}\nSnippet: {snippet}")
        else:
            parts.append(f"Candidate {i}: {snippet}")
    return "\n".join(parts)


def _parse_proposal(text: str) -> tuple[bool, list[str]]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("proposal must be a JSON object")
    continue_flag = data.get("continue")
    if not isinstance(continue_flag, bool):
        raise ValueError("continue must be a boolean")
    sub_questions = data.get("sub_questions")
    if not isinstance(sub_questions, list):
        raise ValueError("sub_questions must be a list")
    filtered = [sq for sq in sub_questions if isinstance(sq, str) and sq.strip()]
    return continue_flag, filtered[:SUB_QUESTION_CAP]


def _parse_selection(text: str, n_candidates: int) -> int | None:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("selection must be a JSON object")
    selected = data.get("selected")
    if selected is None:
        return None
    if not isinstance(selected, int):
        raise ValueError("selected must be an integer or null")
    if selected < 1 or selected > n_candidates:
        raise ValueError(f"selected index {selected} out of range 1..{n_candidates}")
    return selected - 1


async def _llm_complete(llm: Any, messages: list[dict], deadline: float) -> str:
    remaining = _remaining(deadline)
    if remaining <= 0:
        raise _Timeout()
    try:
        response = await asyncio.wait_for(
            llm.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            ),
            timeout=min(SERVICE_TIMEOUT_SECONDS, remaining),
        )
        return response.choices[0].message.content
    except asyncio.TimeoutError:
        if _remaining(deadline) <= 0:
            raise _Timeout()
        raise _LLMError()
    except Exception:
        raise _LLMError()


async def _propose(
    llm: Any,
    question: str,
    evidence: list[EvidenceEntry],
    deadline: float,
) -> tuple[bool, list[str]]:
    messages = [
        {"role": "system", "content": _PROPOSE_SYSTEM},
        {
            "role": "user",
            "content": _PROPOSE_USER.format(
                max_sub=SUB_QUESTION_CAP,
                evidence_text=_evidence_text(evidence),
                question=question,
            ),
        },
    ]
    for attempt in range(2):
        try:
            text = await _llm_complete(llm, messages, deadline)
            return _parse_proposal(text)
        except _Timeout:
            raise
        except (_LLMError, json.JSONDecodeError, ValueError, KeyError):
            if attempt == 1:
                raise _LLMError()
    raise _LLMError()


async def _select(
    llm: Any,
    question: str,
    evidence: list[EvidenceEntry],
    candidates: list[_Candidate],
    deadline: float,
) -> int | None:
    messages = [
        {"role": "system", "content": _SELECT_SYSTEM},
        {
            "role": "user",
            "content": _SELECT_USER.format(
                evidence_text=_evidence_text(evidence),
                candidate_list=_candidate_list(candidates),
                question=question,
            ),
        },
    ]
    for attempt in range(2):
        try:
            text = await _llm_complete(llm, messages, deadline)
            return _parse_selection(text, len(candidates))
        except _Timeout:
            raise
        except (_LLMError, json.JSONDecodeError, ValueError, KeyError):
            if attempt == 1:
                raise _LLMError()
    raise _LLMError()


async def _atom_rows(
    conn: Any,
    qvec_lit: str,
    excluded: list[str],
    kind: str | None,
    tags: list[str] | None,
    include_archived: bool,
) -> list[Any]:
    tbl = f'"{PG_SCHEMA}"."memory_chunks"'
    predicates, filter_args = _history_predicates(
        include_archived=include_archived,
        kind=kind,
        tags=tags,
        alias="parent",
    )
    excl_idx = len(filter_args) + 2
    excl_clause = f"parent.id != ALL(${excl_idx}::text[])"
    args = [qvec_lit, *filter_args, excluded]
    return await conn.fetch(
        f"""
        SELECT atom.id AS atom_id,
               coalesce(atom.distilled, atom.content_raw) AS matched_question,
               1 - (atom.embedding <=> $1::halfvec) AS atom_cosine,
               parent.id, parent.source_ref, parent.chunk_kind, parent.content_raw,
               parent.distilled, parent.ts_last_active, parent.metadata
        FROM {tbl} AS atom
        JOIN {tbl} AS parent ON parent.id = atom.metadata->>'parent_id'
        WHERE atom.chunk_kind = 'atom' AND {predicates} AND {excl_clause}
        ORDER BY atom.embedding <=> $1::halfvec
        LIMIT {OVERSAMPLE_FACTOR * HOP_CANDIDATE_CAP}
        """,
        *args,
    )


def _collapse_atoms(rows: list[Any]) -> list[_Candidate]:
    best_by_parent: dict[str, Any] = {}
    for row in rows:
        current = best_by_parent.get(row["id"])
        if current is None or row["atom_cosine"] > current["atom_cosine"]:
            best_by_parent[row["id"]] = row
    collapsed = sorted(
        best_by_parent.values(),
        key=lambda row: row["atom_cosine"],
        reverse=True,
    )[:HOP_CANDIDATE_CAP]
    candidates = []
    for row in collapsed:
        metadata = _metadata_dict(row["metadata"])
        candidates.append(
            _Candidate(
                parent_id=row["id"],
                ref=metadata.get("search_ref") or row["source_ref"],
                text=row["distilled"] or row["content_raw"],
                ts=row["ts_last_active"],
                kind=row["chunk_kind"],
                tags=metadata.get("tags", []),
                source_ref=row["source_ref"],
                atom_question=row["matched_question"],
            )
        )
    return candidates


async def _memory_backup(
    conn: Any,
    query: str,
    qvec_lit: str,
    excluded: list[str],
    kind: str | None,
    tags: list[str] | None,
    include_archived: bool,
) -> list[_Candidate]:
    tbl = f'"{PG_SCHEMA}"."memory_chunks"'
    exists = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", f"{PG_SCHEMA}.memory_chunks")
    if not exists:
        return []
    predicates, filter_args = _history_predicates(
        include_archived=include_archived,
        kind=kind,
        tags=tags,
    )
    excl_idx = len(filter_args) + 2
    excl_clause = f"id != ALL(${excl_idx}::text[])"
    columns = (
        "id, source_ref, chunk_kind, metadata, distilled, content_raw, ts_last_active, idf_score"
    )
    vec_args = [qvec_lit, *filter_args, excluded]
    vec_rows = await conn.fetch(
        f"SELECT {columns} FROM {tbl} WHERE {predicates} AND {excl_clause} "
        f"ORDER BY embedding <=> $1::halfvec LIMIT {CANDIDATES_PER_SIGNAL}",
        *vec_args,
    )
    fts_args = [query, *filter_args, excluded]
    fts_rows = await conn.fetch(
        f"SELECT {columns} FROM {tbl} WHERE {predicates} AND {excl_clause} AND "
        "to_tsvector('simple', content_raw) @@ websearch_to_tsquery('simple', $1) "
        f"ORDER BY ts_rank_cd(to_tsvector('simple', content_raw), "
        f"websearch_to_tsquery('simple', $1)) DESC "
        f"LIMIT {CANDIDATES_PER_SIGNAL}",
        *fts_args,
    )
    by_id = {r["id"]: r for r in [*vec_rows, *fts_rows]}
    recency = _time_decay_list({r["id"]: r["ts_last_active"] for r in by_id.values()})
    idf = _time_decay_list({r["id"]: r["idf_score"] or 0.0 for r in by_id.values()})
    scores = _rrf_fuse([[r["id"] for r in vec_rows], [r["id"] for r in fts_rows], recency, idf])
    now = time.time()
    for cid in scores:
        age_days = max(0.0, (now - by_id[cid]["ts_last_active"]) / 86400.0)
        scores[cid] *= math.pow(0.5, age_days / TIME_DECAY_HALF_LIFE_DAYS)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    counts: dict[str, int] = {}
    candidates = []
    for cid, _score in ranked:
        r = by_id[cid]
        metadata = _metadata_dict(r["metadata"])
        key = metadata.get("source_ref", r["source_ref"])
        if counts.get(key, 0) >= PER_FILE_CAP:
            continue
        counts[key] = counts.get(key, 0) + 1
        candidates.append(
            _Candidate(
                parent_id=cid,
                ref=metadata.get("search_ref") or r["source_ref"],
                text=r["distilled"] or r["content_raw"],
                ts=r["ts_last_active"],
                kind=r["chunk_kind"],
                tags=metadata.get("tags", []),
                source_ref=r["source_ref"],
                atom_question=None,
            )
        )
        if len(candidates) >= HOP_CANDIDATE_CAP:
            break
    return candidates


async def _hop_retrieve(
    conn: Any,
    embedder: Any,
    sub_questions: list[str],
    question: str,
    qvec_lit: str,
    excluded: list[str],
    kind: str | None,
    tags: list[str] | None,
    include_archived: bool,
    deadline: float,
) -> list[_Candidate]:
    all_rows: list[Any] = []
    for sq in sub_questions:
        if _remaining(deadline) <= 0:
            raise _Timeout()
        try:
            sq_vec = await asyncio.wait_for(
                embedder.embed(sq, query=True),
                timeout=_call_timeout(deadline),
            )
        except asyncio.TimeoutError:
            if _remaining(deadline) <= 0:
                raise _Timeout()
            continue
        except Exception:
            continue
        sq_lit = vector_literal(sq_vec)
        rows = await _atom_rows(conn, sq_lit, excluded, kind, tags, include_archived)
        all_rows.extend(rows)
    candidates = _collapse_atoms(all_rows)
    if candidates:
        return candidates

    if _remaining(deadline) <= 0:
        raise _Timeout()
    rows = await _atom_rows(conn, qvec_lit, excluded, kind, tags, include_archived)
    candidates = _collapse_atoms(rows)
    if candidates:
        return candidates

    if _remaining(deadline) <= 0:
        raise _Timeout()
    return await _memory_backup(
        conn,
        question,
        qvec_lit,
        excluded,
        kind,
        tags,
        include_archived,
    )


async def _deep_loop(
    question: str,
    conn: Any,
    embedder: Any,
    llm: Any,
    max_hops: int,
    kind: str | None,
    tags: list[str] | None,
    include_archived: bool,
    deadline: float,
) -> DeepResult:
    evidence: list[EvidenceEntry] = []
    chosen_parent_ids: set[str] = set()
    trace: list[TraceEntry] = []

    if _remaining(deadline) <= 0:
        return DeepResult(evidence, trace, 0, "timeout")
    try:
        qvec = await asyncio.wait_for(
            embedder.embed(question, query=True),
            timeout=_call_timeout(deadline),
        )
    except asyncio.TimeoutError:
        return DeepResult(evidence, trace, 0, "timeout")
    except Exception:
        return DeepResult(evidence, trace, 0, "llm_error")
    qvec_lit = vector_literal(qvec)

    for hop in range(1, max_hops + 1):
        if _remaining(deadline) <= 0:
            return DeepResult(evidence, trace, len(evidence), "timeout")

        try:
            continue_flag, sub_questions = await _propose(
                llm,
                question,
                evidence,
                deadline,
            )
        except _Timeout:
            trace.append(TraceEntry(hop, [], None))
            return DeepResult(evidence, trace, len(evidence), "timeout")
        except _LLMError:
            trace.append(TraceEntry(hop, [], None))
            return DeepResult(evidence, trace, len(evidence), "llm_error")

        if not continue_flag:
            trace.append(TraceEntry(hop, sub_questions, None))
            return DeepResult(evidence, trace, len(evidence), "done")

        excluded = list(chosen_parent_ids)
        try:
            candidates = await _hop_retrieve(
                conn,
                embedder,
                sub_questions,
                question,
                qvec_lit,
                excluded,
                kind,
                tags,
                include_archived,
                deadline,
            )
        except _Timeout:
            trace.append(TraceEntry(hop, sub_questions, None))
            return DeepResult(evidence, trace, len(evidence), "timeout")

        if not candidates:
            trace.append(TraceEntry(hop, sub_questions, None))
            return DeepResult(evidence, trace, len(evidence), "no_candidates")

        try:
            selected_idx = await _select(
                llm,
                question,
                evidence,
                candidates,
                deadline,
            )
        except _Timeout:
            trace.append(TraceEntry(hop, sub_questions, None))
            return DeepResult(evidence, trace, len(evidence), "timeout")
        except _LLMError:
            trace.append(TraceEntry(hop, sub_questions, None))
            return DeepResult(evidence, trace, len(evidence), "llm_error")

        if selected_idx is None:
            trace.append(TraceEntry(hop, sub_questions, None))
            return DeepResult(evidence, trace, len(evidence), "no_selection")

        chosen = candidates[selected_idx]
        evidence.append(
            EvidenceEntry(
                id=chosen.parent_id,
                ref=chosen.ref,
                text=chosen.text,
                kind=chosen.kind,
                tags=chosen.tags,
                date=chosen.ts,
                hop=hop,
                atom_question=chosen.atom_question,
            )
        )
        chosen_parent_ids.add(chosen.parent_id)
        trace.append(TraceEntry(hop, sub_questions, chosen.ref))

    return DeepResult(evidence, trace, len(evidence), "max_hops")


async def deep_search(
    query: str,
    *,
    max_hops: int | None = None,
    kind: str | None = None,
    tags: list[str] | None = None,
    include_archived: bool = False,
) -> DeepResult:
    if max_hops is None:
        max_hops = DEEP_MAX_HOPS
    if not isinstance(max_hops, int) or not (1 <= max_hops <= DEEP_MAX_HOPS):
        raise ValueError(f"max_hops must be between 1 and {DEEP_MAX_HOPS}")
    if kind is not None and kind not in ("doc", "note", "decision"):
        raise ValueError("kind must be one of ('doc', 'note', 'decision')")
    if tags is not None:
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ValueError("tags must be a list of strings")
        tags = list(dict.fromkeys(t.strip().lower() for t in tags if t.strip()))
        if not tags:
            raise ValueError("tags must be a non-empty list of strings")

    deadline = time.monotonic() + DEEP_TIMEOUT_SECONDS
    embedder = VllmEmbedder()
    llm = llm_client()
    conn = await asyncpg.connect(DB_URL)
    try:
        return await _deep_loop(
            query,
            conn,
            embedder,
            llm,
            max_hops,
            kind,
            tags,
            include_archived,
            deadline,
        )
    finally:
        await conn.close()
