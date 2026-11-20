"""scripts/probe_recall.py: the split, the hit rule, the floor choice and the parsers on stubs.

No test here reaches a server or starts a hook process: HTTP is a fake inside the process.
"""

from __future__ import annotations

import json

import pytest

import probe_recall as probe
from recall import search as hook_search


def _case(case_id, kind="decision", gold=()):
    return {"id": case_id, "kind": kind, "gold": list(gold)}


def test_split_puts_odd_numeric_suffixes_in_calibration_and_even_in_held_out():
    cases = [_case(f"c{n:02d}") for n in (1, 2, 3, 10, 11)]
    calibration, held_out = probe.split_cases(cases)
    assert [c["id"] for c in calibration] == ["c01", "c03", "c11"]
    assert [c["id"] for c in held_out] == ["c02", "c10"]


def test_map_parsing_keeps_only_arrow_lines():
    text = "Author: claude-code\n#12 -> D-3\n#5 skipped (already imported)\n#7 -> D-?\n#9 -> D-11\nrecorded 2, skipped 1\n"
    assert probe.parse_map(text) == {12: 3, 9: 11}


def test_gold_of_an_unmapped_issue_is_an_error():
    with pytest.raises(ValueError, match="#99"):
        probe.gold_numbers(_case("c01", gold=[99]), {12: 3})
    assert probe.gold_numbers(_case("c01", gold=[12, 9]), {12: 3, 9: 11}) == {3, 11}


def test_the_tail_lengths_swept_are_0_200_400_and_800():
    assert probe.TAILS == (0, 200, 400, 800)


def test_query_holds_the_last_l_chars_of_the_tail_whatever_the_hooks_own_cut():
    tail = "".join(chr(65 + n % 26) for n in range(1500))
    prompt = "x" * 2500
    assert probe.build_query(prompt, tail, 800) == tail[-800:] + "\n\n" + "x" * 2000
    assert probe.build_query(prompt, tail, 200) == hook_search.query(prompt, tail)
    assert hook_search.TAIL_QUERY_CHARS == 200
    assert probe.build_query("그래 그렇게 하자 지금 바로", tail, 0) == ""
    assert (
        probe.build_query("a prompt of over twenty chars", tail, 0)
        == "a prompt of over twenty chars"
    )


def _rows(*pairs):
    return [{"number": n, "score": s} for n, s in pairs]


def test_hit_needs_every_gold_among_the_top_three_decisions_at_or_above_the_floor():
    rows = _rows((1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6))
    assert probe.emitted_decisions(rows, 0.5) == [1, 2, 3]
    assert probe.emitted_decisions(rows, 0.85) == [1]
    assert probe.is_hit({1, 2}, [1, 2])
    assert not probe.is_hit({1, 3}, [1, 2])
    assert not probe.is_hit({1}, [])


def test_choice_maximises_hits_under_the_false_positive_cap_and_ties_go_to_the_higher_floor():
    cases = [_case(f"c{n:02d}", gold=[n]) for n in (1, 3, 5)]
    cases += [_case(f"c{n:02d}", kind="unrelated") for n in range(7, 27, 2)]  # 10 unrelated
    issue_map = {1: 1, 3: 3, 5: 5}
    scores = {}
    for length in (0, 800):
        for case in cases:
            if case["kind"] == "unrelated":
                # one unrelated turn scores 0.60 (decision 9) at 800 and 0.30 at 0
                top = 0.60 if (length == 800 and case["id"] == "c07") else 0.30
                scores[case["id"], length] = {"decisions": _rows((9, top))}
            else:
                n = case["gold"][0]
                gold = 0.9 if length == 800 else 0.4
                scores[case["id"], length] = {"decisions": _rows((n, gold))}
    length, floor, hits = probe.choose(cases, scores, issue_map, (0, 800), (0.35, 0.45, 0.65, 0.85))
    # 800 @ 0.65 and 0.85 both hit 3 with 0 false positives; the higher floor wins.
    assert (length, floor, hits) == (800, 0.85, 3)
    # at 800 and floor 0.45, c07 (0.60) is 1/10 false positives, still within the 10% cap
    assert probe.summarize(
        cases,
        {c["id"]: probe.emitted_decisions(scores[c["id"], 800]["decisions"], 0.45) for c in cases},
        issue_map,
    ) == {
        "decision_cases": 3,
        "hits": 3,
        "unrelated_cases": 10,
        "false_positives": 1,
    }


def test_choice_refuses_when_no_floor_meets_the_cap():
    cases = [_case("c01", gold=[1])] + [_case(f"c{n:02d}", kind="unrelated") for n in (3, 5)]
    scores = {(c["id"], 0): {"decisions": _rows((1, 0.99))} for c in cases}
    with pytest.raises(ValueError, match="false-positive"):
        probe.choose(cases, scores, {1: 1}, (0,), (0.5, 0.95))


def test_hook_output_parser_reads_decision_and_note_headers():
    context = (
        "<memory-recall>\nMemory recall: data.\n\n"
        "Related recorded decisions (titles only; read one with get_decision):\n"
        "D-12 Keep lanes apart (2026-01-05)\nD-3 Another (2026-01-06)\n\n"
        "Note 4f2a-91 (2026-02-01; tags: a, b)\nsome tip\n</memory-recall>"
    )
    stdout = json.dumps(
        {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}
    )
    assert probe.parse_hook_output(stdout) == ([12, 3], ["4f2a-91"])
    assert probe.parse_hook_output("") == ([], [])


def test_percentile_is_nearest_rank():
    assert probe.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert probe.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert probe.percentile([], 50) is None


def test_seed_notes_copies_each_note_with_allow_similar_and_prints_counts_only(monkeypatch, capsys):
    monkeypatch.setenv("MEMORY_BASE_API_KEY", "k")
    monkeypatch.setenv("MEMORY_BASE_AUTHOR", "a")
    monkeypatch.setenv("MEMORY_BASE_TO_API_KEY", "target-key")
    listing = [
        {
            "id": "1",
            "kind": "work",
            "text": "secret one",
            "tags": ["t"],
            "namespace": "default",
            "date": "2026-01-02",
        },
        {
            "id": "2",
            "kind": "personal",
            "text": "secret two",
            "tags": ["u"],
            "namespace": "default",
            "date": "2026-01-03",
            "shared": True,
        },
    ]
    calls = []

    keys = []

    def fake_rest(url, *, api_key, author, payload=None):
        calls.append((url, payload))
        keys.append(api_key)
        if payload is None:
            return listing
        if payload["content"] == "secret two":
            raise probe.RequestFailed("409")
        return {"id": "new"}

    monkeypatch.setattr(probe, "rest", fake_rest)
    assert (
        probe.main(["seed-notes", "--from", "http://a/", "--to", "http://b"]) == 1
    )  # one note failed
    assert calls[0][0] == "http://a/notes?limit=200"
    first = calls[1]
    assert first[0] == "http://b/save_memory"
    assert first[1] == {
        "content": "secret one",
        "kind": "work",
        "tags": ["t"],
        "namespace": "default",
        "occurred_at": "2026-01-02",
        "allow_similar": True,
    }
    assert calls[2][1]["shared"] is True
    assert keys == ["k", "target-key", "target-key"]
    out = capsys.readouterr().out
    assert "copied 1" in out and "failed 1" in out and "secret" not in out


def _seed_env(monkeypatch, to_key="target-key"):
    monkeypatch.setenv("MEMORY_BASE_API_KEY", "k")
    monkeypatch.setenv("MEMORY_BASE_AUTHOR", "a")
    if to_key:
        monkeypatch.setenv("MEMORY_BASE_TO_API_KEY", to_key)
    else:
        monkeypatch.delenv("MEMORY_BASE_TO_API_KEY", raising=False)


@pytest.mark.parametrize("target", ["http://a", "http://a/"])
def test_seed_notes_refuses_to_copy_a_server_onto_itself(monkeypatch, target):
    _seed_env(monkeypatch)
    monkeypatch.setattr(probe, "rest", lambda *a, **k: pytest.fail("no request expected"))

    assert probe.main(["seed-notes", "--from", "http://a", "--to", target]) == 2


def test_seed_notes_needs_the_target_key_from_the_environment(monkeypatch):
    _seed_env(monkeypatch, to_key=None)
    monkeypatch.setattr(probe, "rest", lambda *a, **k: pytest.fail("no request expected"))

    assert probe.main(["seed-notes", "--from", "http://a", "--to", "http://b"]) == 2
    with pytest.raises(SystemExit):
        probe.main(["seed-notes", "--from", "http://a", "--to", "http://b", "--to-key", "x"])


def test_seed_notes_warns_when_a_full_page_came_back(monkeypatch, capsys):
    _seed_env(monkeypatch)
    note = {"text": "t", "kind": "work", "tags": [], "namespace": "default", "date": "2026-01-02"}
    monkeypatch.setattr(
        probe,
        "rest",
        lambda url, *, api_key, author, payload=None: [note] * 200 if payload is None else {},
    )

    assert probe.main(["seed-notes", "--from", "http://a", "--to", "http://b"]) == 0
    captured = capsys.readouterr()
    assert "copied 200" in captured.out
    assert "200" in captured.err and "more" in captured.err


def test_the_scope_and_origin_come_from_the_repo_argument():
    assert probe.scope_of("acme/app") == "repo:github.com/acme/app"
    assert probe.origin_of("acme/app") == "https://github.com/acme/app.git"


def test_the_sweep_asks_recall_for_the_repos_scope(monkeypatch):
    bodies = []

    def fake_rest(url, *, api_key, author, payload=None):
        bodies.append(payload)
        return {"decisions": [], "notes": []}

    monkeypatch.setattr(probe, "rest", fake_rest)
    case = {
        "id": "c01",
        "kind": "decision",
        "gold": [],
        "prompt": "a prompt of over twenty chars",
        "assistant_tail": "x",
    }

    probe.recall_sweep("http://a", [case], "repo:github.com/acme/app", api_key="k", author="a")

    assert {body["scope"] for body in bodies} == {"repo:github.com/acme/app"}
