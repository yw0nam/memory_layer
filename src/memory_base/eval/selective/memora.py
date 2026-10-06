"""Convert Memora's weekly timelines into writer sessions, gold facts, and questions.

Per persona under <root>/memora/<persona>/:
- sessions.jsonl: {session_id, date, turns: [{role: user|assistant, text}]} in date order,
  consecutive turns by one speaker merged. This is the only file a writer sees.
- facts.jsonl: one gold fact per memory session (add / update / delete) from
  operation_details, with the share-memory turns as evidence and `update_of` naming the
  latest earlier fact on the same item.
- questions.jsonl: question, answer (Memora's memory_evidence), evidence session ids,
  the forgetting evidence that must not be delivered as current, and the grading checks.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from memory_base.eval.longmemeval import update_manifest, write_jsonl_atomic
from memory_base.eval.selective import DEFAULT_ROOT, DEV_PERSONAS, manifest_path, source_commit

ROLES = {"user_agent": "user", "ai_agent": "assistant"}
VALUE_KEYS = (
    "item", "preference", "old_item", "old_preference",
    "content_data", "memory_updates", "memory_deletes",
)  # fmt: skip


def _turns(conversation: list[dict[str, Any]]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for turn in conversation:
        role = ROLES[turn["speaker"]]
        if turns and turns[-1]["role"] == role:
            turns[-1]["text"] += "\n" + turn["message"]
        else:
            turns.append({"role": role, "text": turn["message"]})
    return turns


def _item_key(details: dict[str, Any], item: Any) -> tuple[Any, ...]:
    if isinstance(item, dict):
        return (details["category"], item.get("description") or item.get("event_name"))
    return (details.get("subcategory"), item)


def _category(details: dict[str, Any]) -> str:
    return (
        details.get("category")
        or details.get("subcategory")
        or re.sub(r"_\d+$", "", details["item"])
    )


def _session_ids(node: Any) -> set[int]:
    if isinstance(node, dict):
        found = {node["session_id"]} if "session_id" in node else set()
        found |= set(node.get("session_history", []))
        return found.union(*(_session_ids(value) for value in node.values()))
    if isinstance(node, list):
        return set().union(*(_session_ids(value) for value in node))
    return set()


def convert_persona(persona_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    persona = persona_dir.name
    raw = [json.loads(p.read_text()) for p in (persona_dir / "conversations").glob("*.json")]
    raw.sort(key=lambda s: (s["date"], s["session_id"]))
    sessions, facts, latest = [], [], {}
    for session in raw:
        sid = session["session_id"]
        sessions.append(
            {"session_id": sid, "date": session["date"], "turns": _turns(session["conversation"])}
        )
        if session["operation"] is None:
            continue
        details = session["operation_details"]
        fact_id = f"{persona}:{sid}"
        key = _item_key(details, details["item"])
        old_key = _item_key(details, details.get("old_item", details["item"]))
        facts.append(
            {
                "fact_id": fact_id,
                "session_id": sid,
                "date": session["date"],
                "session_type": session["session_type"],
                "operation": session["operation"],
                "category": _category(details),
                "value": {k: details[k] for k in VALUE_KEYS if k in details},
                "evidence": "\n".join(
                    t["message"] for t in session["conversation"] if t["share_memory"]
                ),
                "update_of": None if session["operation"] == "add" else latest.get(old_key),
            }
        )
        latest[key] = fact_id
    gold = json.loads((persona_dir / f"evaluation_questions_{persona}.json").read_text())
    questions = [
        {
            "question_id": q["question_id"],
            "kind": kind,
            "question": q["question"],
            "question_date": q["question_date"],
            "answer": q["memory_evidence"],
            "evidence_session_ids": sorted(_session_ids(q["memory_evidence"])),
            "forgetting": (q.get("forgetting_evidence") or {}).get("forgotten_items", []),
            "checks": q["evaluation"]["evaluation_questions"],
        }
        for kind, group in gold["questions"].items()
        for q in group
    ]
    return sessions, facts, questions


def convert_weekly(weekly_dir: Path, root: Path) -> dict[str, dict[str, Any]]:
    """Convert every persona directory under `weekly_dir`; per-persona counts."""
    stats = {}
    for persona_dir in sorted(p for p in weekly_dir.iterdir() if p.is_dir()):
        sessions, facts, questions = convert_persona(persona_dir)
        out = root / "memora" / persona_dir.name
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(out / "sessions.jsonl", sessions)
        write_jsonl_atomic(out / "facts.jsonl", facts)
        write_jsonl_atomic(out / "questions.jsonl", questions)
        stats[persona_dir.name] = {
            "sessions": len(sessions),
            "turns": sum(len(s["turns"]) for s in sessions),
            "facts": dict(Counter(f["operation"] for f in facts)),
            "unlinked_changes": sum(f["operation"] != "add" and not f["update_of"] for f in facts),
            "questions": len(questions),
        }
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="Memora checkout")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    stats = convert_weekly(args.source / "data" / "weekly", args.root)
    personas = sorted(stats)
    update_manifest(
        manifest_path(args.root),
        "memora",
        {
            "source": "geniesinc/Memora",
            "commit": source_commit(args.source),
            "period": "weekly",
            "split": {
                "dev": list(DEV_PERSONAS),
                "held_out": [p for p in personas if p not in DEV_PERSONAS],
            },
            "counts": stats,
        },
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
