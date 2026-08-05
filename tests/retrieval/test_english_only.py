"""English-only policy (Issue #20).

A pure guard that requires no services: src/ stores/emits English only.

The Hangul range boundaries are written with unicode escapes so this test file
itself stays ASCII in its logic.
"""

from __future__ import annotations

from pathlib import Path

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
