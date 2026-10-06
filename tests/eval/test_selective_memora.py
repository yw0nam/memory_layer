"""Unit coverage for the Memora weekly converter."""

from __future__ import annotations

from pathlib import Path

from memory_base.eval.longmemeval import read_jsonl
from memory_base.eval.selective import memora

FIXTURE = Path(__file__).parent / "fixtures" / "memora_weekly"


def test_a_weekly_persona_converts_to_sessions_facts_and_questions(tmp_path):
    memora.convert_weekly(FIXTURE, tmp_path)
    out = tmp_path / "memora" / "tester"

    sessions = read_jsonl(out / "sessions.jsonl")
    assert [s["session_id"] for s in sessions] == [1, 2, 3]
    assert sessions[0] == {
        "session_id": 1,
        "date": "2025-06-01",
        "turns": [
            {"role": "user", "text": "Hi there."},
            {"role": "assistant", "text": "Hello!\nWhat is on your mind?"},
            {"role": "user", "text": "Just saying hello."},
        ],
    }

    added, updated = read_jsonl(out / "facts.jsonl")
    assert added["fact_id"] == "tester:2"
    assert (added["operation"], added["category"], added["update_of"]) == ("add", "actors", None)
    assert added["evidence"] == "I really like Joan Crawford."
    assert updated["fact_id"] == "tester:3"
    assert (updated["session_id"], updated["date"]) == (3, "2025-06-02")
    assert (updated["operation"], updated["update_of"]) == ("update", "tester:2")
    assert updated["value"]["old_item"] == "Joan Crawford"

    [question] = read_jsonl(out / "questions.jsonl")
    assert question["question"] == "Can you suggest me a movie?"
    assert question["evidence_session_ids"] == [3]
    assert question["forgetting"] == [{"value": "Joan Crawford", "session_id": 2}]
    assert [c["expected_answer"] for c in question["checks"]] == ["yes", "no"]
