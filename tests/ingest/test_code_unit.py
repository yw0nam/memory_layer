"""Unit tests for the multi-repo code indexer's pure path helper."""

from __future__ import annotations

import asyncio

import numpy as np

from memory_base.ingest import code


def test_cache_rel_builds_repo_prefixed_path():
    path = code.CACHE_ROOT / "repo-a" / "pkg" / "mod.py"
    assert code._cache_rel(path) == "repo-a/pkg/mod.py"


def test_cache_rel_is_unique_across_repos_for_same_relative_path():
    a = code.CACHE_ROOT / "repo-a" / "src" / "main.py"
    b = code.CACHE_ROOT / "repo-b" / "src" / "main.py"
    assert code._cache_rel(a) != code._cache_rel(b)
    assert code._cache_rel(a) == "repo-a/src/main.py"
    assert code._cache_rel(b) == "repo-b/src/main.py"


def test_is_minified_flags_long_average_line_length():
    # 3 lines, ~500KB each: mirrors a real minified swagger-ui-bundle.js.
    minified = "\n".join("x" * 500_000 for _ in range(3))
    assert code._is_minified(minified, "swagger-ui-bundle.js") is True


def test_is_minified_spares_large_normally_formatted_source():
    # 17k lines averaging ~32 chars/line, like an unminified lodash.js.
    normal = "\n".join(f"function fn{i}(a, b) {{ return a + b; }}" for i in range(17_000))
    assert code._is_minified(normal, "lodash.js") is False


def test_is_minified_exempts_markdown_regardless_of_line_length():
    # One long paragraph-per-line, and a table using <br><br> instead of newlines:
    # legitimate markdown, not a generated bundle.
    prose = "word " * 300 + "\n" + "<br><br>".join(f"cell {i}" for i in range(100))
    assert code._is_minified(prose, "swagger-ui-bundle.js") is True
    assert code._is_minified(prose, "notes.md") is False


def test_is_minified_empty_file_is_not_minified():
    assert code._is_minified("", "empty.py") is False


def test_is_minified_no_trailing_newline():
    assert code._is_minified("x" * (code._MAX_AVG_LINE_LENGTH + 1), "one_line.js") is True


def test_is_minified_all_blank_lines_is_not_minified():
    assert code._is_minified("\n\n\n\n", "blank.py") is False


class _FakeEmbedder:
    def __init__(self):
        self.batch_sizes = []

    async def embed_many(self, texts):
        self.batch_sizes.append(len(texts))
        return [np.zeros(1, dtype=np.float16) for _ in texts]


def test_batched_embed_collapses_concurrent_calls_into_few_batch_requests():
    embedder = _FakeEmbedder()
    n = 200

    async def run():
        return await asyncio.gather(*(code._batched_embed(f"text-{i}", embedder) for i in range(n)))

    results = asyncio.run(run())

    assert len(results) == n
    assert sum(embedder.batch_sizes) == n  # every text embedded exactly once
    assert len(embedder.batch_sizes) <= 5  # ceil(200 / 64) == 4, +1 slack for scheduling
    assert max(embedder.batch_sizes) <= code._EMBED_BATCH_SIZE
