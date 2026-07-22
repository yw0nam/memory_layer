"""Query -> Planner(source selection) -> Executor(hybrid search) -> Synthesis(cited answer) CLI.

uv run python src/answer.py "질문" [--source auto|code|history|all]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone

from common import LLM_MODEL, llm_client
from search import Hit, search

EVIDENCE_TEXT_LIMIT = 2000

PLANNER_SYSTEM_PROMPT = """당신은 검색 플래너입니다. 사용자 질의를 보고 어떤 소스를 검색할지,
그리고 검색에 쓸 질의 1~3개를 정합니다.

- 질의가 코드 구조/구현 위치를 묻는 질문이면 source="code".
- "예전에", "그때", "어떻게 풀었지" 류의 과거 대화/작업 회고 질문이면 source="history".
- 둘 다 해당하거나 애매하면 source="all".
- queries는 원 질의를 검색 친화적으로 변형한 것들이며, 원 질의 자체도 반드시 포함해야 합니다.

다음 JSON 형식으로만 답하세요: {"source": "code"|"history"|"all", "queries": ["...", ...]}
"""

SYNTHESIS_SYSTEM_PROMPT = """당신은 제공된 증거를 바탕으로 질문에 답하는 어시스턴트입니다.

원칙:
- 제공된 증거만 사용하여 답변할 것. 증거에 없는 내용을 추측하지 말 것.
- 각 주장 뒤에는 근거가 된 증거 번호를 [1][2] 형태로 인용할 것.
- 증거가 부족하면 부족하다고 명시할 것.
- 오래된 정보와 최신 정보가 충돌하면 최신 정보를 우선하고, 그 사실을 주의사항으로 표기할 것.
- 반드시 한국어로 답변할 것.
"""


def dedup_sort_hits(hits: list[Hit], top_k: int = 10) -> list[Hit]:
    """Dedup by (source, ref), keep the best-scoring copy, sort desc, take top_k."""
    best: dict[tuple[str, str], Hit] = {}
    for h in hits:
        key = (h.source, h.ref)
        score = h.rerank_score if h.rerank_score is not None else h.rrf
        existing = best.get(key)
        if existing is None:
            best[key] = h
        else:
            existing_score = existing.rerank_score if existing.rerank_score is not None else existing.rrf
            if score > existing_score:
                best[key] = h
    ordered = sorted(
        best.values(),
        key=lambda h: h.rerank_score if h.rerank_score is not None else h.rrf,
        reverse=True,
    )
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
            block.append(f"(주변 맥락)\n{context}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def format_references_section(hits: list[Hit]) -> str:
    lines = ["참조:"]
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
    data = json.loads(resp.choices[0].message.content)
    source = data.get("source", "all")
    if source not in ("code", "history", "all"):
        source = "all"
    queries = data.get("queries") or [query]
    if query not in queries:
        queries = [query, *queries]
    return source, queries[:3]


async def execute(source: str, queries: list[str]) -> list[Hit]:
    """Run search.search() for each query in parallel, then dedup/sort/top-10."""
    results = await asyncio.gather(*(search(q, source=source) for q in queries))
    all_hits = [h for r in results for h in r]
    return dedup_sort_hits(all_hits, top_k=10)


async def synthesize(query: str, hits: list[Hit]) -> str:
    """LLM call: cited Korean answer from evidence blocks."""
    client = llm_client()
    user_content = f"{format_evidence_block(hits)}\n\n질문: {query}"
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    answer = resp.choices[0].message.content
    return f"{answer}\n\n{format_references_section(hits)}"


async def answer(query: str, source: str = "auto") -> str:
    if source == "auto":
        decided_source, queries = await plan(query)
    else:
        decided_source, queries = source, [query]

    hits = await execute(decided_source, queries)
    if not hits:
        return "관련 증거를 찾지 못했습니다."
    return await synthesize(query, hits)


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--source", choices=["auto", "code", "history", "all"], default="auto")
    args = ap.parse_args()

    print(await answer(args.query, source=args.source))


if __name__ == "__main__":
    asyncio.run(_main())
