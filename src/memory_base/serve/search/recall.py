"""The two branches of POST /recall: ranked notes and the caller's active decisions."""

from __future__ import annotations

from typing import Any

from memory_base.retrieval.search import Hit, rerank_scores, search
from memory_base.serve.decisions import store as decision_store

DECISION_CANDIDATES = 200


async def recall_notes(query: str, author: str, top_k: int, min_score: float) -> list[Hit]:
    """The caller's visible notes above the reranked-score floor, best first."""
    return (await search(query, visible_to=author, min_score=min_score))[:top_k]


def _decision_text(row: dict[str, Any]) -> str:
    parts = [f"D-{row['number']} {row['title']}", row["decision"], row["why"]]
    return "\n\n".join(part for part in parts if part)


async def recall_decisions(
    query: str, scope: str, author: str, top_k: int, min_score: float
) -> list[dict[str, Any]]:
    """The caller's newest active decisions of the scope, reranked, above the floor."""
    rows = await decision_store.active_decisions(
        author=author, scope=scope, limit=DECISION_CANDIDATES
    )
    if not rows:
        return []
    scores = await rerank_scores(query, [_decision_text(row) for row in rows])
    scored = [{**row, "score": score} for row, score in zip(rows, scores, strict=True)]
    kept = [row for row in scored if row["score"] >= min_score]
    kept.sort(key=lambda row: (-row["score"], -row["number"]))
    return kept[:top_k]
