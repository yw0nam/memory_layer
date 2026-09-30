"""Per-hit judgments of a LongMemEval candidates run and the read-settings frontier over them.

The judge labels each of a packet's first ten hits useful, related, misleading or
unrelated to the question and names the shortest hit prefix that suffices for the
reference answer. The frontier cuts every top_k x floor cell from those labels and
reports evidence coverage, junk share and session recall per cell.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from memory_base.eval import longmemeval as lme
from memory_base.eval.claude_code import strip_fence

JUDGE_HITS_TOP = 10
LABELS = ("useful", "related", "misleading", "unrelated")
FRONTIER_TOP_KS = (1, 2, 3, 5, 10)
FRONTIER_FLOORS = (0, 0.05, 0.1, 0.25, 0.4, 0.5)
AGGREGATION_TYPES = ("multi-session", "temporal-reasoning", "knowledge-update")
BEST_MAX_JUNK = 0.10
HIT_JUDGE_SYSTEM_PROMPT = """\
You judge the notes a memory search returned for one question about a user. You receive \
the question with its date, the reference answer, and the notes in rank order, each \
numbered and dated. Label every note with exactly one of:
- useful: contains information a careful answerer would use to produce the reference answer (a fact the answer states, a date needed to compute it, or a fact that rules out a wrong answer).
- related: not needed for this answer, but about the user and the same subject the question asks about (the same entity, activity, or category), so it is sensible context that would not push the answerer toward a wrong answer.
- misleading: would push a careful answerer toward a wrong answer: it contradicts the reference answer, states a value the answer has superseded, or looks like it answers the question but does not (the wrong event, person, item, or time window).
- unrelated: about a different subject than the question.
Then give min_prefix: the smallest n such that notes [0..n-1] together are enough for a \
careful answerer to produce the reference answer, or null if all the notes together are \
not enough.
Reply with JSON only: {"labels": {"0": label, "1": label, ...}, "min_prefix": n | null}, \
where labels is an object keyed by the index of every note.
"""


def hit_judge_prompt(packet: dict[str, Any], question: dict[str, Any]) -> str:
    notes = "\n".join(
        f"[{index}] ({hit['date']}) {hit['text']}"
        for index, hit in enumerate(packet["hits"][:JUDGE_HITS_TOP])
    )
    return (
        f"Question date: {packet['question_date']}\n"
        f"Question: {packet['question']}\n"
        f"Reference answer: {question['answer']}\n\n"
        f"Notes:\n{notes}\n"
    )


def parse_hit_judgment(text: str, count: int) -> tuple[list[str], int | None]:
    """(labels in index order, min_prefix); ValueError when the reply is malformed."""
    reply = json.loads(strip_fence(text))
    if not isinstance(reply, dict):
        raise ValueError("the judgment is not a JSON object")
    labels = reply.get("labels")
    if not isinstance(labels, dict) or set(labels) != {str(i) for i in range(count)}:
        raise ValueError("labels must be an object keyed by every note index")
    ordered = [labels[str(i)] for i in range(count)]
    if any(label not in LABELS for label in ordered):
        raise ValueError(f"unknown label in {ordered}")
    min_prefix = reply.get("min_prefix")
    if min_prefix is not None and (
        isinstance(min_prefix, bool)
        or not isinstance(min_prefix, int)
        or not 1 <= min_prefix <= count
    ):
        raise ValueError(f"min_prefix out of range: {min_prefix!r}")
    return ordered, min_prefix


def judged_texts(packet: dict[str, Any]) -> list[str]:
    return [hit["text"] for hit in packet["hits"][:JUDGE_HITS_TOP]]


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _cell(top_k: int, floor: float, judged: Sequence[tuple[dict, dict, dict]]) -> dict[str, Any]:
    covered: dict[bool, list[bool]] = {True: [], False: []}
    counts = dict.fromkeys(LABELS, 0)
    recall, delivered_counts = [], []
    for packet, row, question in judged:
        delivered = [h for h in packet["hits"][:top_k] if h["score"] >= floor]
        n = len(delivered)
        delivered_counts.append(n)
        is_covered = row["min_prefix"] is not None and n >= row["min_prefix"]
        covered[question["question_type"] in AGGREGATION_TYPES].append(is_covered)
        for label in row["labels"][:n]:
            counts[label] += 1
        answers = set(question["answer_session_ids"])
        ranking = lme.session_ranking(delivered)
        recall.append(lme.recall_all_at_k(ranking, answers, k=len(ranking)) if n else 0.0)
    total = sum(delivered_counts)
    return {
        "top_k": top_k,
        "floor": floor,
        "coverage": lme._mean(covered[True] + covered[False]),
        "coverage_agg": lme._mean(covered[True]),
        "coverage_lookup": lme._mean(covered[False]),
        "junk": _share(counts["misleading"] + counts["unrelated"], total),
        "misleading": _share(counts["misleading"], total),
        "related": _share(counts["related"], total),
        "recall_all": lme._mean(recall),
        "hits_per_question": lme._mean(delivered_counts),
    }


def frontier(
    packets: Sequence[dict[str, Any]],
    judgments: Sequence[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Coverage, junk and recall of every top_k x floor cell over the judged questions."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in judgments:
        if "error" not in row:
            rows.setdefault(row["question_id"], []).append(row)
    excluded = {"no_judgment": 0, "stale_judgment": 0, "abstention": 0}
    judged = []
    for packet in packets:
        qid = packet["question_id"]
        if "_abs" in qid:
            excluded["abstention"] += 1
            continue
        if qid not in rows:
            excluded["no_judgment"] += 1
            continue
        texts = judged_texts(packet)
        current = [row for row in rows[qid] if row["texts"] == texts]
        if not current:
            excluded["stale_judgment"] += 1
            continue
        judged.append((packet, current[-1], questions[qid]))
    cells = [_cell(k, floor, judged) for k in FRONTIER_TOP_KS for floor in FRONTIER_FLOORS]
    cells.sort(
        key=lambda c: (
            c["junk"] is None,
            c["junk"] or 0.0,
            -(c["coverage"] or 0.0),
            c["hits_per_question"] or 0.0,
        )
    )
    best = next((c for c in cells if c["junk"] is not None and c["junk"] <= BEST_MAX_JUNK), None)
    return {"questions": len(judged), "excluded": excluded, "cells": cells, "best": best}


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:.3f}"


def render_frontier(report: dict[str, Any]) -> str:
    excluded = report["excluded"]
    lines = [
        f"Frontier over {report['questions']} questions (excluded: "
        f"{excluded['no_judgment']} without a judgment, {excluded['stale_judgment']} with a "
        f"stale judgment, {excluded['abstention']} abstention)",
        "",
        "| top_k | floor | coverage | agg | lookup | junk | misleading | related | recall_all "
        "| hits/q |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in report["cells"]:
        values = [
            c[name]
            for name in (
                "coverage", "coverage_agg", "coverage_lookup", "junk", "misleading",
                "related", "recall_all", "hits_per_question",
            )
        ]  # fmt: skip
        cells = " | ".join(_fmt(value) for value in values)
        lines.append(f"| {c['top_k']} | {c['floor']:g} | {cells} |")
    best = report["best"]
    lines.append("")
    if best is None:
        lines.append(f"Best cell (junk <= {BEST_MAX_JUNK:.2f}): no cell qualifies")
    else:
        lines.append(
            f"Best cell (junk <= {BEST_MAX_JUNK:.2f}): top_k={best['top_k']}, "
            f"floor={best['floor']:g}, coverage {_fmt(best['coverage'])}, "
            f"junk {_fmt(best['junk'])}, hits/q {_fmt(best['hits_per_question'])}"
        )
    return "\n".join(lines) + "\n"
