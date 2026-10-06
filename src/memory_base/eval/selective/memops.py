"""MemOps update chains and credential cases as a stale-value unit set.

Reads only the clean evidence conversations (generated_result/2-evidence_conversation), not
the UltraChat filler, and writes <root>/memops_updates.jsonl:
- kind "update_chain": one row per chain_id, steps in chain order with old and new values
  and validity, `current` = the last confirmed value, and the referenced segments in order.
- kind "credential": one row per remember/update whose target names a secret (a password,
  PIN, lock combination, pickup, verification or authentication code, gate code); memory_base must
  refuse these, so a chain on such a target is emitted only as credential cases.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from memory_base.eval.longmemeval import update_manifest, write_jsonl_atomic
from memory_base.eval.selective import DEFAULT_ROOT, manifest_path, source_commit

EVIDENCE_DIR = Path("generated_result") / "2-evidence_conversation"
CREDENTIAL = re.compile(
    r"password|passcode|\bpin\b|combination|pickup code|verification code|2fa code"
    r"|authentication code|gate code"
)


def _segments(evidence: dict[str, Any], indexes: set[int]) -> list[dict[str, Any]]:
    return [
        {
            "segment_index": seg["segment_index"],
            "turns": [{"role": t["role"], "text": t["content"]} for t in seg["dialogue"]],
        }
        for seg in sorted(evidence["conversations"], key=lambda s: s["segment_index"])
        if seg["segment_index"] in indexes
    ]


def _span(op: dict[str, Any]) -> dict[str, Any]:
    span = op["trigger_span"]
    return {k: span[k] for k in ("segment_index", "turn_index", "quote")}


def _is_credential(op: dict[str, Any]) -> bool:
    return CREDENTIAL.search(op["target"]["target_name"].lower()) is not None


def convert_file(path: Path) -> list[dict[str, Any]]:
    evidence = json.loads(path.read_text())
    ops = evidence["operations"]
    rows = []
    for chain_id in sorted({op["chain_id"] for op in ops if op.get("chain_id")}):
        steps = sorted(
            (op for op in ops if op.get("chain_id") == chain_id), key=lambda op: op["chain_step"]
        )
        if _is_credential(steps[0]):
            continue
        confirmed = [op["new_value"] for op in steps if op["validity"] == "confirmed"]
        rows.append(
            {
                "kind": "update_chain",
                "case_id": f"{path.stem}/{chain_id}",
                "target": steps[0]["target"]["target_name"],
                "steps": [
                    {
                        "step": op["chain_step"],
                        "type": op["type"],
                        "validity": op["validity"],
                        "old_value": op["old_value"],
                        "new_value": op["new_value"],
                        **_span(op),
                    }
                    for op in steps
                ],
                "current": confirmed[-1] if confirmed else None,
                "segments": _segments(
                    evidence, {op["trigger_span"]["segment_index"] for op in steps}
                ),
            }
        )
    for op in ops:
        if op["type"] in ("remember", "update") and op["new_value"] and _is_credential(op):
            rows.append(
                {
                    "kind": "credential",
                    "case_id": f"{path.stem}/{op['operation_id']}",
                    "target": op["target"]["target_name"],
                    "value": op["new_value"],
                    **_span(op),
                    "segments": _segments(evidence, {op["trigger_span"]["segment_index"]}),
                }
            )
    return rows


def convert(evidence_dir: Path) -> list[dict[str, Any]]:
    return [row for path in sorted(evidence_dir.glob("*.json")) for row in convert_file(path)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="MemOps checkout")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    rows = convert(args.source / EVIDENCE_DIR)
    args.root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.root / "memops_updates.jsonl", rows)
    counts = dict(Counter(row["kind"] for row in rows))
    update_manifest(
        manifest_path(args.root),
        "memops",
        {"source": "MemTensor/MemOps", "commit": source_commit(args.source), "counts": counts},
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
