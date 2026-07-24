"""Unit tests for the multi-repo code indexer's pure path helper."""

from __future__ import annotations

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
