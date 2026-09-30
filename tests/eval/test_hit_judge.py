"""Unit coverage for the per-hit judge prompt, its parser, and the frontier report."""

from __future__ import annotations

import json

import pytest

from memory_base.eval import hit_judge

D = "2023/05/01 (Mon) 10:00"


def reply(labels, min_prefix):
    return json.dumps({"labels": labels, "min_prefix": min_prefix})


def test_a_valid_reply_parses_to_labels_in_index_order():
    text = reply({"1": "related", "0": "useful", "2": "misleading"}, 1)
    assert hit_judge.parse_hit_judgment(text, 3) == (["useful", "related", "misleading"], 1)
    fenced = "```json\n" + reply({"0": "unrelated"}, None) + "\n```"
    assert hit_judge.parse_hit_judgment(fenced, 1) == (["unrelated"], None)


@pytest.mark.parametrize(
    "text",
    [
        reply({"0": "useful"}, 1),
        reply({"0": "useful", "1": "useful", "2": "useful"}, 1),
        reply(["useful", "useful"], 1),
        reply({"0": "useful", "1": "helpful"}, 1),
        reply({"0": "useful", "1": "useful"}, 0),
        reply({"0": "useful", "1": "useful"}, 3),
        reply({"0": "useful", "1": "useful"}, True),
        reply({"0": "useful", "1": "useful"}, "1"),
        "not json",
        json.dumps([]),
    ],
)
def test_a_malformed_reply_is_refused(text):
    with pytest.raises(ValueError):
        hit_judge.parse_hit_judgment(text, 2)


def test_the_system_prompt_defines_the_four_labels_verbatim():
    for definition in (
        "useful: contains information a careful answerer would use to produce the reference "
        "answer (a fact the answer states, a date needed to compute it, or a fact that rules "
        "out a wrong answer).",
        "unrelated: about a different subject than the question.",
    ):
        assert definition in hit_judge.HIT_JUDGE_SYSTEM_PROMPT
    assert hit_judge.LABELS == ("useful", "related", "misleading", "unrelated")


def test_the_prompt_numbers_the_first_ten_hits_with_their_dates_and_the_reference():
    hits = [{"date": f"2023/05/{i + 1:02d} (Mon) 10:00", "text": f"fact {i}"} for i in range(12)]
    packet = {"question": "How many?", "question_date": "2023/06/01 (Thu) 10:00", "hits": hits}
    question = {"answer": "twenty dozen"}
    prompt = hit_judge.hit_judge_prompt(packet, question)
    assert "How many?" in prompt and "2023/06/01 (Thu) 10:00" in prompt
    assert "twenty dozen" in prompt
    assert "[0] (2023/05/01 (Mon) 10:00) fact 0" in prompt
    assert "[9] (2023/05/10 (Mon) 10:00) fact 9" in prompt
    assert "fact 10" not in prompt
    assert prompt.index("[0]") < prompt.index("[1]") < prompt.index("[9]")


def hit(sid, score):
    return {"id": sid, "date": D, "score": score, "text": f"fact {sid}", "sessions": [[sid, D]]}


def question(qid, qtype, answers):
    return {"question_id": qid, "question_type": qtype, "answer_session_ids": answers}


def judged(packet, labels, min_prefix, **extra):
    texts = [h["text"] for h in packet["hits"][:10]]
    return {"question_id": packet["question_id"], "texts": texts, "labels": labels,
            "min_prefix": min_prefix, **extra}  # fmt: skip


def fixture():
    qa = {"question_id": "qa", "hits": [hit("s1", 0.9), hit("s9", 0.45), hit("s2", 0.3)]}
    qa["hits"].append(hit("s8", 0.03))
    qb = {"question_id": "qb", "hits": [hit("s5", 0.6), hit("s6", 0.2)]}
    qc = {"question_id": "qc", "hits": [hit("s7", 0.08)]}
    qd = {"question_id": "qd_abs", "hits": [hit("s3", 0.9)]}
    qe = {"question_id": "qe", "hits": [hit("s4", 0.9)]}
    qf = {"question_id": "qf", "hits": [hit("s0", 0.9)]}
    packets = [qa, qb, qc, qd, qe, qf]
    judgments = [
        {"question_id": "qa", "texts": [], "error": "timeout"},
        judged(qa, ["useful", "misleading", "useful", "unrelated"], 3),
        judged(qb, ["useful", "related"], 1),
        judged(qc, ["useful"], 1),
        judged(qd, ["useful"], 1),
        {**judged(qe, ["useful"], 1), "texts": ["an older packet's note"]},
        {"question_id": "qf", "texts": ["fact s0"], "error": "malformed"},
    ]
    questions = {
        "qa": question("qa", "multi-session", ["s1", "s2"]),
        "qb": question("qb", "single-session-user", ["s5"]),
        "qc": question("qc", "knowledge-update", ["s7"]),
        "qd_abs": question("qd_abs", "multi-session", ["s3"]),
        "qe": question("qe", "temporal-reasoning", ["s4"]),
        "qf": question("qf", "multi-session", ["s0"]),
    }
    return packets, judgments, questions


def cell(report, top_k, floor):
    return next(c for c in report["cells"] if c["top_k"] == top_k and c["floor"] == floor)


def test_the_frontier_counts_exclusions_and_computes_each_cell():
    report = hit_judge.frontier(*fixture())
    assert report["questions"] == 3
    assert report["excluded"] == {"no_judgment": 1, "stale_judgment": 1, "abstention": 1}
    assert len(report["cells"]) == 30

    full = cell(report, 10, 0)
    assert full["coverage"] == pytest.approx(1.0)
    assert full["coverage_agg"] == pytest.approx(1.0)
    assert full["coverage_lookup"] == pytest.approx(1.0)
    assert full["junk"] == pytest.approx(2 / 7)
    assert full["misleading"] == pytest.approx(1 / 7)
    assert full["related"] == pytest.approx(1 / 7)
    assert full["recall_all"] == pytest.approx(1.0)
    assert full["hits_per_question"] == pytest.approx(7 / 3)

    floored = cell(report, 10, 0.25)
    assert floored["coverage"] == pytest.approx(2 / 3)
    assert floored["coverage_agg"] == pytest.approx(0.5)
    assert floored["coverage_lookup"] == pytest.approx(1.0)
    assert floored["junk"] == pytest.approx(1 / 4)
    assert floored["misleading"] == pytest.approx(1 / 4)
    assert floored["related"] == pytest.approx(0.0)
    assert floored["recall_all"] == pytest.approx(2 / 3)
    assert floored["hits_per_question"] == pytest.approx(4 / 3)

    top1 = cell(report, 1, 0)
    assert top1["coverage"] == pytest.approx(2 / 3)
    assert top1["junk"] == pytest.approx(0.0)
    assert top1["recall_all"] == pytest.approx(2 / 3)
    assert top1["hits_per_question"] == pytest.approx(1.0)


def test_the_frontier_sorts_by_junk_then_coverage_then_hits_and_names_the_best_cell():
    report = hit_judge.frontier(*fixture())
    keys = [
        (c["junk"] is None, c["junk"] or 0.0, -c["coverage"], c["hits_per_question"])
        for c in report["cells"]
    ]
    assert keys == sorted(keys)
    assert (report["cells"][0]["top_k"], report["cells"][0]["floor"]) == (1, 0)
    assert report["best"] == report["cells"][0]
    rendered = hit_judge.render_frontier(report)
    assert "3 questions" in rendered
    lines = [line for line in rendered.splitlines() if line.startswith("| 1 |")]
    assert lines[0].startswith("| 1 | 0 |")
    assert "Best cell" in rendered and "top_k=1" in rendered


def test_without_a_qualifying_cell_there_is_no_best_and_empty_cells_sort_last():
    packet = {"question_id": "qa", "hits": [hit("s1", 0.3)]}
    report = hit_judge.frontier(
        [packet],
        [judged(packet, ["misleading"], None)],
        {"qa": question("qa", "multi-session", ["s1"])},
    )
    assert report["best"] is None
    assert all(c["junk"] is None for c in report["cells"][-10:])
    assert all(c["junk"] == pytest.approx(1.0) for c in report["cells"][:-10])
    assert cell(report, 1, 0.4)["coverage"] == 0.0
    assert cell(report, 1, 0.4)["hits_per_question"] == 0.0
    assert "no cell" in hit_judge.render_frontier(report)
