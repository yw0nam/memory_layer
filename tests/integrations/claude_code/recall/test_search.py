"""Unit tests for the prompt recall query and its one note search — HTTP faked."""

from __future__ import annotations

import threading
import time

import pytest

from recall import search


def _hit(note_id, score):
    return {"id": note_id, "kind": "work", "tags": ["t"], "date": "2026-11-01", "score": score}


def test_the_query_drops_complete_memory_fences_and_is_cut_to_2000_chars():
    prompt = (
        "Why does this fail?\n"
        "<MEMORY-CONTEXT>\nMemory: old handoffs\n</memory-context>\n"
        "<memory-recall>\nNote: stale\n</ memory-recall>\n"
        "and a bare <memory-recall tag stays " + "z" * 3000
    )

    text = search.query(prompt)

    assert text.startswith("Why does this fail?")
    assert "and a bare <memory-recall tag stays" in text
    assert "old handoffs" not in text and "stale" not in text
    assert len(text) == 2000


def test_one_search_keeps_only_hits_at_or_above_the_floor():
    calls = []

    def post(path, body, timeout):
        calls.append((path, body))
        return [_hit("n1", 0.9), _hit("n2", 0.5), _hit("n3", 0.3)]

    hits = search.search(post, "how do we retopologize the hair", time.monotonic() + 2)

    assert [hit["id"] for hit in hits] == ["n1", "n2"]
    assert calls == [
        ("/search", {"query": "how do we retopologize the hair", "top_k": 3, "min_score": 0.5})
    ]


@pytest.mark.parametrize(
    "answer", [OSError("connection refused"), {"detail": "not a list"}], ids=["down", "malformed"]
)
def test_a_failed_search_gives_no_hits(answer):
    def post(path, body, timeout):
        if isinstance(answer, Exception):
            raise answer
        return answer

    assert search.search(post, "how do we retopologize the hair", time.monotonic() + 2) == []


def test_a_search_still_running_at_the_cutoff_gives_no_hits():
    release = threading.Event()

    def post(path, body, timeout):
        release.wait(10)
        return [_hit("n1", 0.9)]

    started = time.monotonic()
    try:
        hits = search.search(post, "how do we retopologize the hair", started + 0.2)
    finally:
        release.set()

    assert hits == []
    assert time.monotonic() - started < 1.0
