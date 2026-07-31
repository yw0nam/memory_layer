"""English-only policy (Issue #20).

Two flavors of test live here:

- A pure guard that requires no services: src/ stores/emits English only.
- An LLM integration test (marker ``integration``) that drives the real
  sub-question path and asserts English output.

The Hangul range boundaries are written with unicode escapes so this test file
itself stays ASCII in its logic. Korean string literals appear only as
deliberate input-fixture data, each flagged with an English comment.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from memory_base.core.config import llm_client, llm_model
from memory_base.retrieval.decompose import _propose

REPO_ROOT = Path(__file__).resolve().parents[2]

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


# ---- integration (real LLM) --------------------------------------------


def _require_llm() -> None:
    async def _check() -> None:
        await asyncio.wait_for(
            llm_client().chat.completions.create(
                model=llm_model(),
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            ),
            timeout=10,
        )

    try:
        asyncio.run(_check())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"LLM not reachable; skipping integration test: {exc}")


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
