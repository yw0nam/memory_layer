"""Unit coverage for the two-judge keep relabel and the owner's Markdown round trip."""

from __future__ import annotations

import asyncio

from memory_base.eval.longmemeval import append_jsonl, read_jsonl
from memory_base.eval.selective import relabel


def fact(persona: str, session_id: int) -> dict:
    return {
        "fact_id": f"{persona}:{session_id}",
        "session_id": session_id,
        "date": "2025-06-01",
        "operation": "add",
        "category": "actors",
        "value": {"item": "Joan Crawford", "preference": "like"},
        "evidence": "I really like Joan | Crawford.",
        "update_of": None,
    }


def fill_owner(path, fact_id: str, decision: str) -> None:
    lines = path.read_text().splitlines()
    [row] = [i for i, line in enumerate(lines) if line.startswith(f"| {fact_id} |")]
    lines[row] = lines[row].removesuffix("|").rstrip() + f" {decision} |"
    path.write_text("\n".join(lines) + "\n")


def test_disagreements_and_an_agreement_sample_go_to_the_owner_and_come_back(tmp_path):
    dev = tmp_path / "memora" / "software_engineer"
    dev.mkdir(parents=True)
    append_jsonl(dev / "facts.jsonl", [fact("software_engineer", i) for i in range(11)])
    held_out = tmp_path / "memora" / "content_writer"
    held_out.mkdir()
    append_jsonl(held_out / "facts.jsonl", [fact("content_writer", 1)])
    calls = []

    async def keeps(f):
        calls.append(f["fact_id"])
        return {"label": "keep", "reason": "asked again later"}

    async def skips_first(f):
        calls.append(f["fact_id"])
        label = "skip" if f["fact_id"] == "software_engineer:0" else "keep"
        return {"label": label, "reason": "filler"}

    judges = {"sonnet": keeps, "glm": skips_first}
    asyncio.run(relabel.relabel(tmp_path, judges, contract="c1", seed=0))
    assert len(calls) == 22 and "content_writer:1" not in calls

    out = tmp_path / "relabel"
    disagreements = (out / "disagreements.md").read_text()
    assert "| software_engineer:0 |" in disagreements
    audit_rows = [
        line for line in (out / "audit.md").read_text().splitlines() if "| software_" in line
    ]
    assert len(audit_rows) == 1
    audited = audit_rows[0].split("|")[1].strip()

    relabel.finalize(tmp_path)
    labels = {row["fact_id"]: row for row in read_jsonl(out / "labels.jsonl")}
    assert len(labels) == 10 and "software_engineer:0" not in labels

    fill_owner(out / "disagreements.md", "software_engineer:0", "skip")
    fill_owner(out / "audit.md", audited, "skip")
    asyncio.run(relabel.relabel(tmp_path, judges, contract="c1", seed=0))
    assert len(calls) == 22

    relabel.finalize(tmp_path)
    labels = {row["fact_id"]: row for row in read_jsonl(out / "labels.jsonl")}
    assert len(labels) == 11
    assert labels["software_engineer:0"] == {
        "fact_id": "software_engineer:0", "label": "skip", "source": "owner", "contract": "c1"
    }  # fmt: skip
    assert (labels[audited]["label"], labels[audited]["source"]) == ("skip", "owner")
    others = [row for fid, row in labels.items() if fid not in {"software_engineer:0", audited}]
    assert {(row["label"], row["source"]) for row in others} == {("keep", "judges")}


def test_the_judge_sees_what_the_user_said_not_the_dataset_category():
    prompt = relabel.judge_prompt(fact("software_engineer", 1))
    assert "Joan | Crawford" in prompt and "actors" not in prompt
