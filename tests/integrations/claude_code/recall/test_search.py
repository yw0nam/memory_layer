"""Unit tests for the prompt recall query and its one /recall call — HTTP faked."""

from __future__ import annotations

import threading
import time

import pytest

from recall import search


def test_the_query_is_the_previous_text_a_blank_line_and_the_prompt_without_fences_cut_to_2000():
    prompt = (
        "Why does this fail?\n"
        "<MEMORY-CONTEXT>\nMemory: old handoffs\n</memory-context>\n"
        "<memory-recall>\nNote: stale\n</ memory-recall>\n"
        "and a bare <memory-recall tag stays " + "z" * 3000
    )

    text = search.query(prompt, "I will retopologize the hair.")

    assert text.startswith("I will retopologize the hair.\n\nWhy does this fail?")
    assert "and a bare <memory-recall tag stays" in text
    assert "old handoffs" not in text and "stale" not in text
    assert len(text) == len("I will retopologize the hair.\n\n") + 2000


@pytest.mark.parametrize(
    ("prompt", "previous", "expected"),
    [
        ("/review the whole branch against the spec", "A long answer.", ""),
        ("/compact", "", ""),
        ("그래", "", ""),
        ("<memory-recall>\nNote: a fenced block\n</memory-recall>", "A long answer.", ""),
        ("그래", "Shall I use the old rig?", "Shall I use the old rig?\n\n그래"),
        ("fix the typo in the readme", "", "fix the typo in the readme"),
    ],
    ids=["slash", "slash-alone", "short-alone", "fence-only", "short-reply", "long-alone"],
)
def test_the_query_skips_slash_and_unanswered_short_prompts(prompt, previous, expected):
    assert search.query(prompt, previous) == expected


def _reply(**lanes):
    return {"notes": [], "decisions": [], "errors": [], **lanes}


def test_one_recall_asks_both_lanes_with_the_scope_and_floors():
    calls = []

    def post(path, body, timeout):
        calls.append((path, body))
        return _reply(notes=[{"id": "n1"}], decisions=[{"number": 3}], errors=["notes"])

    result = search.search(post, "the query", "repo:github.com/o/r", time.monotonic() + 2)

    assert result == ([{"id": "n1"}], [{"number": 3}], ["notes"])
    assert calls == [
        (
            "/recall",
            {
                "query": "the query",
                "scope": "repo:github.com/o/r",
                "notes": {"top_k": 5, "min_score": search.NOTES_FLOOR},
                "decisions": {"top_k": 5, "min_score": search.DECISIONS_FLOOR},
            },
        )
    ]
    assert (search.NOTES_FLOOR, search.DECISIONS_FLOOR) == (0.5, 0.5)


def test_a_prompt_without_a_scope_sends_none():
    calls = []

    def post(path, body, timeout):
        calls.append(body)
        return _reply()

    search.search(post, "the query", None, time.monotonic() + 2)

    assert "scope" not in calls[0]


@pytest.mark.parametrize(
    "answer", [OSError("connection refused"), ["not", "an", "object"]], ids=["down", "malformed"]
)
def test_a_failed_recall_gives_no_hits(answer):
    def post(path, body, timeout):
        if isinstance(answer, Exception):
            raise answer
        return answer

    notes, decisions, _ = search.search(post, "the query", None, time.monotonic() + 2)

    assert (notes, decisions) == ([], [])


def test_a_recall_still_running_at_the_cutoff_gives_no_hits():
    release = threading.Event()

    def post(path, body, timeout):
        release.wait(10)
        return _reply(notes=[{"id": "n1"}])

    started = time.monotonic()
    try:
        notes, decisions, _ = search.search(post, "the query", None, started + 0.2)
    finally:
        release.set()

    assert (notes, decisions) == ([], [])
    assert time.monotonic() - started < 1.0
