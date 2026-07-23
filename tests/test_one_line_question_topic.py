"""Integration test: one_line_question must capture the session's central topic,
not the opening message (GitHub issue #27).

Requires a live vLLM endpoint. Skipped when the endpoint is unreachable.
"""

from __future__ import annotations

import asyncio

import pytest

from memory_base.adapters.base import Message, Session
from memory_base.common import LLM_MODEL, llm_client
from memory_base.ingest.history import Distillation, _distill, build_transcript


def _require_llm() -> None:
    async def _check() -> None:
        await asyncio.wait_for(
            llm_client().chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            ),
            timeout=10,
        )

    try:
        asyncio.run(_check())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"LLM not reachable; skipping integration test: {exc}")


def _run_sync(coro):
    return asyncio.run(coro)


def _topic_pivot_session() -> Session:
    """Session that opens with a trivial question but pivots to a substantial
    topic: designing a hybrid retrieval pipeline with FTS + vector search."""
    messages = [
        Message(
            role="user",
            text="Is the honcho Docker container currently running?",
            timestamp=1_700_100_000.0,
            session_id="s-topic-pivot",
            cwd="/repo",
            git_branch="main",
        ),
        Message(
            role="assistant",
            text=(
                "Let me check. Yes, the container is running on port 8080. "
                "The health endpoint returns 200 OK."
            ),
            timestamp=1_700_100_010.0,
            session_id="s-topic-pivot",
        ),
        Message(
            role="user",
            text=(
                "Good. Now I want to design the retrieval pipeline for our memory layer. "
                "We need full-text search combined with vector similarity search. "
                "The idea is to use pgvector for the vector index and PostgreSQL's "
                "built-in tsvector for FTS. We should merge results using reciprocal "
                "rank fusion (RRF) and then optionally apply a cross-encoder reranker "
                "to improve precision on the top candidates."
            ),
            timestamp=1_700_100_020.0,
            session_id="s-topic-pivot",
        ),
        Message(
            role="assistant",
            text=(
                "That sounds like a solid architecture. Here is how I would structure it: "
                "1) Build a hybrid search function that runs both FTS and vector KNN queries "
                "in parallel against the memory_chunks table. 2) Fuse the two result sets "
                "using RRF with k=60 as the default constant. 3) Apply a time-decay factor "
                "so recent sessions are boosted. 4) Optionally pass the top-20 fused results "
                "through a cross-encoder reranker for final reordering. "
                "The pgvector halfvec_cosine_ops operator class works well for cosine similarity, "
                "and we can use an HNSW index with ef_search=100 for good recall at low latency."
            ),
            timestamp=1_700_100_030.0,
            session_id="s-topic-pivot",
        ),
        Message(
            role="user",
            text=(
                "Great. Let us also add IDF-based weighting so that common tokens across "
                "sessions do not dominate the relevance score. We can compute document "
                "frequencies across all stored transcripts and use inverse document frequency "
                "to down-weight ubiquitous terms."
            ),
            timestamp=1_700_100_040.0,
            session_id="s-topic-pivot",
        ),
        Message(
            role="assistant",
            text=(
                "Agreed. We can build a corpus-level DF table that gets updated incrementally "
                "as new sessions are ingested. The mean-IDF score per burst can serve as an "
                "additional signal in the burst gate, filtering out low-information passages "
                "before they are even stored. This keeps the retrieval pipeline focused on "
                "high-signal content."
            ),
            timestamp=1_700_100_050.0,
            session_id="s-topic-pivot",
        ),
        Message(
            role="user",
            text=(
                "Perfect. Let us finalize the design: hybrid FTS+vector search with RRF fusion, "
                "IDF weighting in the burst gate, time decay, and optional reranking. "
                "We will store everything in the memory_chunks table with halfvec embeddings."
            ),
            timestamp=1_700_100_060.0,
            session_id="s-topic-pivot",
        ),
    ]
    return Session(
        session_id="s-topic-pivot",
        messages=messages,
        transcript=build_transcript(messages),
        ts_last_active=messages[-1].timestamp,
    )


@pytest.mark.integration
def test_distill_one_line_question_captures_central_topic_not_opener():
    """A session that opens with a trivial Docker question but pivots to
    designing a retrieval pipeline must produce a one_line_question about
    the retrieval pipeline, not the Docker container."""
    _require_llm()

    async def _run() -> Distillation:
        semaphore = asyncio.Semaphore(1)
        return await _distill(_topic_pivot_session(), semaphore)

    distillation = _run_sync(_run())
    question = distillation.one_line_question.lower()

    docker_keywords = ("docker", "container running", "honcho")
    topic_keywords = ("retrieval", "pipeline", "search", "vector", "fts", "rrf", "hybrid", "rank fusion")

    matches_docker = any(kw in question for kw in docker_keywords)
    matches_topic = any(kw in question for kw in topic_keywords)

    assert not matches_docker, (
        f"one_line_question anchors on the trivial opener: {distillation.one_line_question!r}"
    )
    assert matches_topic, (
        f"one_line_question does not capture the central topic: {distillation.one_line_question!r}"
    )
