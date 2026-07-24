"""English-only policy (Issue #20).

Two flavors of test live here:

- Pure guards and literal pins that require no services. They assert the repo
  stores/emits English and keeps Korean **input** handling intact.
- LLM integration tests (marker ``integration``) that drive the real
  distillation and sub-question paths and assert English output.

The Hangul range boundaries are written with unicode escapes so this test file
itself stays ASCII in its logic. Korean string literals appear only as
deliberate input-fixture data, each flagged with an English comment.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from memory_base.adapters.base import Burst, Message, Session, SourceAdapter
from memory_base.common import LLM_MODEL, llm_client
from memory_base.retrieval.decompose import _propose
from memory_base.ingest.history import (
    Distillation,
    build_rows,
    build_transcript,
    parse_distillation,
    tokenize,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Hangul syllables, Hangul Jamo, Hangul compatibility Jamo, and Kana ranges,
# as (low, high) codepoint bounds so this file stays ASCII in its logic.
_HANGUL_KANA_RANGES = (
    (0xAC00, 0xD7AF),
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0x3040, 0x30FF),
)


def _is_hangul_or_kana(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _HANGUL_KANA_RANGES)


def _scan_for_hangul_kana(paths: list[Path]) -> list[str]:
    """Return ``file:line:snippet`` for every line containing a flagged glyph."""
    hits: list[str] = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if any(_is_hangul_or_kana(char) for char in line):
                rel = path.relative_to(REPO_ROOT)
                hits.append(f"{rel}:{lineno}:{line.strip()}")
    return hits


# ---- permanent guards --------------------------------------------------


def test_src_has_no_hangul_or_kana():
    hits = _scan_for_hangul_kana(list((REPO_ROOT / "src").rglob("*.py")))
    assert not hits, "Hangul/Kana found in src/:\n" + "\n".join(hits)


# ---- literal pins ------------------------------------------------------


def test_parse_distillation_missing_resolution_defaults_to_unresolved():
    distillation = parse_distillation(
        {"one_line_question": "How is X wired?", "summary": "A short English summary."}
    )
    assert distillation.resolution == "unresolved"


class _EnglishAdapter(SourceAdapter):
    source_type = "fake"

    def discover(self):
        return []

    def parse(self, text, file):
        return []

    def has_social(self, burst: Burst) -> bool:
        return False


def _english_session() -> Session:
    messages = [
        Message(
            role="user",
            text="How should we set the burst gate threshold for short replies?",
            timestamp=1_700_000_000.0,
            session_id="s-en",
            cwd="/repo",
            git_branch="main",
        ),
        Message(
            role="assistant",
            text="Use a weighted signal sum and require it to clear a fixed floor.",
            timestamp=1_700_000_010.0,
            session_id="s-en",
        ),
    ]
    return Session(
        session_id="s-en",
        messages=messages,
        transcript=build_transcript(messages),
        ts_last_active=messages[-1].timestamp,
    )


def test_decision_row_uses_english_decision_prefix():
    distillation = parse_distillation(
        {
            "one_line_question": "How should the burst gate treat short bursts?",
            "summary": "Weighted-sum gate over signal features.",
            "resolution": "Adopt a fixed floor on the weighted sum.",
            "decisions": ["Set the burst gate floor at four."],
        }
    )
    rows = build_rows(_english_session(), distillation, _EnglishAdapter(), dfs={}, n=1)
    decision_rows = [row for row in rows if row["kind"] == "decision"]
    assert len(decision_rows) == 1
    distilled = decision_rows[0]["distilled"]
    assert "Decision:" in distilled
    # Fixtures are all-English, so any flagged glyph is a static-part leak.
    assert not any(_is_hangul_or_kana(char) for char in distilled)


def test_tokenize_still_splits_hangul_input():
    # Korean input fixture: pins that the Hangul token branch survives the
    # English-only sweep (input handling stays multilingual).
    assert tokenize("메모리 시스템 설계") == {"메모리", "시스템", "설계"}


# ---- integration (real LLM) --------------------------------------------


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


def _hangul_ratio(text: str) -> float:
    if not text:
        return 0.0
    hangul = sum(_is_hangul_or_kana(char) for char in text)
    return hangul / len(text)


def _korean_session() -> Session:
    # Korean transcript fixture: exercises English-output distillation of
    # non-English input. Concrete topic: pgvector HNSW index tuning.
    messages = [
        Message(
            role="user",
            text="pgvector HNSW 인덱스에서 ef_search 값을 어떻게 정하는 게 좋을까? 검색 정확도가 낮아.",
            timestamp=1_700_000_000.0,
            session_id="s-ko",
            cwd="/repo",
            git_branch="main",
        ),
        Message(
            role="assistant",
            text=(
                "ef_search를 키우면 recall이 올라가지만 지연이 늘어난다. "
                "기본값 40에서 시작해서 100까지 올려가며 지연과 정확도를 함께 측정하는 게 좋다."
            ),
            timestamp=1_700_000_010.0,
            session_id="s-ko",
        ),
        Message(
            role="user",
            text="그럼 m 값이랑 ef_construction은 인덱스를 다시 만들어야 바뀌는 거지?",
            timestamp=1_700_000_020.0,
            session_id="s-ko",
        ),
        Message(
            role="assistant",
            text=(
                "맞다. m과 ef_construction은 빌드 타임 파라미터라 CREATE INDEX를 다시 해야 한다. "
                "halfvec(2048) 컬럼에 halfvec_cosine_ops로 hnsw 인덱스를 재생성해야 반영된다."
            ),
            timestamp=1_700_000_030.0,
            session_id="s-ko",
        ),
        Message(
            role="user",
            text="좋아, 그러면 ef_search만 런타임에 조정하고 나머지는 빌드 시점에 정하는 걸로 하자.",
            timestamp=1_700_000_040.0,
            session_id="s-ko",
        ),
    ]
    return Session(
        session_id="s-ko",
        messages=messages,
        transcript=build_transcript(messages),
        ts_last_active=messages[-1].timestamp,
    )


@pytest.mark.integration
def test_distillation_of_korean_transcript_is_english():
    _require_llm()
    from memory_base.ingest.history import _distill

    async def _run() -> Distillation:
        semaphore = asyncio.Semaphore(1)
        return await _distill(_korean_session(), semaphore)

    distillation = _run_sync(_run())
    combined = (
        f"{distillation.one_line_question}\n{distillation.summary}\n{distillation.resolution}"
    )
    ratio = _hangul_ratio(combined)
    assert ratio < 0.05, f"distillation not English (Hangul ratio {ratio:.3f}): {combined!r}"


@pytest.mark.integration
def test_sub_questions_are_english_for_korean_question():
    _require_llm()
    # Korean question fixture: cross-lingual recall must emit English sub-questions.
    _continue, sub_questions = _run_sync(
        _propose(
            llm_client(), "예전에 burst gate 임계값을 어떻게 정했지?", [], time.monotonic() + 120
        )
    )
    assert sub_questions
    for question in sub_questions:
        assert question.isprintable(), f"unprintable query: {question!r}"
        leaked = [char for char in question if _is_hangul_or_kana(char)]
        assert not leaked, f"non-English query: {question!r}"


def _run_sync(coro):
    return asyncio.run(coro)
