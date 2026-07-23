"""Query -> Planner(source selection) -> Executor(hybrid search) -> Synthesis(cited answer) CLI.

uv run python -m memory_base.serve.answer "question" [--source auto|code|memory|all] [--deep]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone

from memory_base.common import LLM_MODEL, llm_client
from memory_base.retrieval.decompose import EvidenceEntry, deep_search
from memory_base.retrieval.search import Hit, search

LOGGER = logging.getLogger("answer")

EVIDENCE_TEXT_LIMIT = 2000

PLANNER_SYSTEM_PROMPT = """You are a search planner. Select which source to search and
produce one to three search-friendly queries for the user's question.

- Use source="code" for questions about code structure or implementation locations.
- Use source="memory" for questions about stored knowledge — past decisions, saved
  notes, ingested documents, or any curated memory.
- Use source="all" when both sources apply or the choice is ambiguous.
- Search queries are ALWAYS English, regardless of the question language. Translate the
  user's intent into English because the FTS index contains English text.

Return only this JSON format: {"source": "code"|"memory"|"all", "queries": ["...", ...]}
"""

SYNTHESIS_SYSTEM_PROMPT = """You are an assistant that answers questions from the
provided evidence.

Rules:
- Use only the provided evidence. Do not speculate beyond it.
- Cite the supporting evidence number after each claim in the form [1][2].
- Explicitly state when the evidence is insufficient.
- When older and newer evidence conflict, prefer the newer evidence and note the conflict
  as a caution.
- Always answer in English.
"""


def dedup_sort_hits(hits: list[Hit], top_k: int = 10) -> list[Hit]:
    """Dedup by (source, ref), keep the best-scoring copy, sort desc, take top_k."""
    best: dict[tuple[str, str], Hit] = {}
    for h in hits:
        key = (h.source, h.ref)
        existing = best.get(key)
        if existing is None or h.score > existing.score:
            best[key] = h
    ordered = sorted(best.values(), key=lambda h: h.score, reverse=True)
    return ordered[:top_k]


def _fmt_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def format_evidence_block(hits: list[Hit]) -> str:
    """Render numbered evidence blocks for the synthesis user message."""
    parts = []
    for i, h in enumerate(hits, start=1):
        text = h.text[:EVIDENCE_TEXT_LIMIT]
        block = [
            f"[{i}] source={h.source} ref={h.ref} date={_fmt_date(h.ts)}",
            text,
        ]
        context = h.meta.get("context") if h.source == "code" else None
        if context:
            block.append(f"(Surrounding context)\n{context}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def format_references_section(hits: list[Hit]) -> str:
    lines = ["References:"]
    for i, h in enumerate(hits, start=1):
        lines.append(f"[{i}] {h.ref}")
    return "\n".join(lines)


async def plan(query: str) -> tuple[str, list[str]]:
    """LLM call: decide source + up to 3 search-friendly query variants."""
    client = llm_client()
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content or "")
        source = data.get("source", "all")
        if source not in ("code", "memory", "all"):
            source = "all"
        queries = data.get("queries") or [query]
    except (json.JSONDecodeError, TypeError, KeyError):
        LOGGER.warning("planner returned invalid JSON; falling back to trivial plan")
        source = "all"
        queries = [query]
    return source, queries[:3]


async def execute(source: str, queries: list[str]) -> list[Hit]:
    """Run search.search() for each query in parallel, then dedup/sort/top-10."""
    results = await asyncio.gather(*(search(q, source=source) for q in queries))
    all_hits = [h for r in results for h in r]
    return dedup_sort_hits(all_hits, top_k=10)


async def synthesize(query: str, hits: list[Hit]) -> str:
    """LLM call: cited English answer from evidence blocks."""
    client = llm_client()
    user_content = f"{format_evidence_block(hits)}\n\nQuestion: {query}"
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    answer = resp.choices[0].message.content
    return f"{answer}\n\n{format_references_section(hits)}"


def _evidence_to_hits(evidence: list[EvidenceEntry]) -> list[Hit]:
    hits = []
    for entry in evidence:
        hits.append(
            Hit(
                source="memory",
                ref=entry.ref,
                text=entry.text,
                ts=entry.date,
                meta={
                    "id": entry.id,
                    "kind": entry.kind,
                    "tags": entry.tags,
                    "hop": entry.hop,
                    "atom_question": entry.atom_question,
                },
            )
        )
    return hits


async def answer(query: str, source: str = "auto", deep: bool = False) -> str:
    if deep:
        result = await deep_search(query)
        hits = _evidence_to_hits(result.evidence)
        if not hits:
            return "No relevant evidence was found."
        return await synthesize(query, hits)

    if source == "auto":
        decided_source, queries = await plan(query)
    else:
        decided_source, queries = source, [query]

    hits = await execute(decided_source, queries)
    if not hits:
        return "No relevant evidence was found."
    return await synthesize(query, hits)


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--source", choices=["auto", "code", "memory", "all"], default="auto")
    ap.add_argument("--deep", action="store_true")
    args = ap.parse_args()

    if args.deep and args.source not in ("auto", "memory"):
        ap.error("--deep requires --source memory (or no --source)")

    print(await answer(args.query, source=args.source, deep=args.deep))


if __name__ == "__main__":
    asyncio.run(_main())
