#!/usr/bin/env python3
"""Real-turn probe of the per-prompt recall: seed a sandbox, calibrate on half, check the other half.

  probe_recall.py seed-notes --from URL --to URL
  probe_recall.py run --repo OWNER/NAME --url URL --cases cases.jsonl --map map.txt [--out report.json]

`seed-notes` copies the caller's notes into another server and prints counts only; it
refuses when --from and --to are the same server. `run` sweeps the tail length and the
decision floor on the calibration half through POST /recall, freezes the best pair, then
runs every case through the hook process (prompt_recall.py via a wrapper that sets the
frozen floor and tail length) and prints a Markdown summary. The case file stays local; the
report and the summary name case ids only.

Environment: MEMORY_BASE_API_KEY, MEMORY_BASE_AUTHOR (also the hook's key and author);
MEMORY_BASE_TO_API_KEY, the target server's key, for seed-notes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

from import_decision_issues import RequestFailed
from import_decision_issues import rest

HOOKS = Path(__file__).resolve().parents[1] / "integrations" / "claude_code" / "hooks"
sys.path.insert(0, str(HOOKS))

from recall import search as hook_search  # noqa: E402
from recall.choose import MAX_DECISIONS  # noqa: E402

TAILS = (0, 200, 400, 800)
FLOORS = tuple(round(step * 0.05, 2) for step in range(1, 20))
MAX_FALSE_POSITIVE = 0.10
LIST_LIMIT = 200
HOOK_TIMEOUT_S = 10
# Sets the frozen floor and tail length, then runs the hook's main() with the plugin's environment.
WRAPPER = (
    "import sys; sys.path.insert(0, sys.argv[1]); import prompt_recall; "
    "from recall import search; search.DECISIONS_FLOOR = float(sys.argv[2]); "
    "search.TAIL_QUERY_CHARS = int(sys.argv[3]) or search.TAIL_QUERY_CHARS; "
    "prompt_recall.main()"
)
_MAP_LINE = re.compile(r"^#(\d+) -> D-(\d+)$")
_DECISION_HEAD = re.compile(r"^D-(\d+) ", re.MULTILINE)
_NOTE_HEAD = re.compile(r"^Note (\S+) \(", re.MULTILINE)


# ---- pure rules ---------------------------------------------------------------


def split_cases(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """(calibration, held-out): an odd numeric suffix of the id calibrates, an even one is held out."""
    calibration, held_out = [], []
    for case in cases:
        suffix = re.search(r"(\d+)$", case["id"])
        if suffix is None:
            raise ValueError(f"case id {case['id']!r} has no numeric suffix")
        (calibration if int(suffix[1]) % 2 else held_out).append(case)
    return calibration, held_out


def parse_map(text: str) -> dict[int, int]:
    """issue number -> D-number from the importer's printed `#<issue> -> D-<n>` lines."""
    return {int(m[1]): int(m[2]) for line in text.splitlines() if (m := _MAP_LINE.match(line))}


def gold_numbers(case: dict, issue_map: dict[int, int]) -> set[int]:
    """The D-numbers of a case's gold issues."""
    missing = [issue for issue in case["gold"] if issue not in issue_map]
    if missing:
        raise ValueError(f"case {case['id']}: gold issue #{missing[0]} is not in the map")
    return {issue_map[issue] for issue in case["gold"]}


def scope_of(repo: str) -> str:
    return f"repo:github.com/{repo}"


def origin_of(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def build_query(prompt: str, tail: str, length: int) -> str:
    """The hook's query for the last `length` chars of the assistant tail and the prompt."""
    with mock.patch.object(hook_search, "TAIL_QUERY_CHARS", length or hook_search.TAIL_QUERY_CHARS):
        return hook_search.query(prompt, tail[-length:] if length else "")


def emitted_decisions(rows: list[dict], floor: float) -> list[int]:
    """The decision numbers the hook would show: the best three scoring at or above the floor."""
    kept = sorted(
        (r for r in rows if r["score"] >= floor), key=lambda r: (-r["score"], -r["number"])
    )
    return [r["number"] for r in kept[:MAX_DECISIONS]]


def is_hit(gold: set[int], emitted: list[int]) -> bool:
    return bool(gold) and gold <= set(emitted)


def summarize(cases: list[dict], emitted: dict[str, list[int]], issue_map: dict[int, int]) -> dict:
    """Gold hits and decision false positives of the cases given the decisions each emitted."""
    decision = [c for c in cases if c["kind"] == "decision"]
    unrelated = [c for c in cases if c["kind"] == "unrelated"]
    return {
        "decision_cases": len(decision),
        "hits": sum(is_hit(gold_numbers(c, issue_map), emitted[c["id"]]) for c in decision),
        "unrelated_cases": len(unrelated),
        "false_positives": sum(bool(emitted[c["id"]]) for c in unrelated),
    }


def sweep_grid(cases, scores, issue_map, tails=TAILS, floors=FLOORS) -> list[dict]:
    """One summary per (tail, floor) over the cases, from the stored /recall score table."""
    grid = []
    for length in tails:
        for floor in floors:
            emitted = {
                c["id"]: emitted_decisions(scores[c["id"], length]["decisions"], floor)
                for c in cases
            }
            grid.append({"tail": length, "floor": floor, **summarize(cases, emitted, issue_map)})
    return grid


def choose(cases, scores, issue_map, tails=TAILS, floors=FLOORS) -> tuple[int, float, int]:
    """(tail, floor, hits) with most hits at <= 10% unrelated cases emitting a decision.

    Ties go to the higher floor, then the shorter tail.
    """
    ok = [
        row
        for row in sweep_grid(cases, scores, issue_map, tails, floors)
        if row["false_positives"] <= MAX_FALSE_POSITIVE * row["unrelated_cases"]
    ]
    if not ok:
        raise ValueError("no tail/floor pair meets the false-positive cap")
    best = max(ok, key=lambda r: (r["hits"], r["floor"], -r["tail"]))
    return best["tail"], best["floor"], best["hits"]


def parse_hook_output(stdout: str) -> tuple[list[int], list[str]]:
    """(decision numbers, note ids) shown in the hook's stdout, in order."""
    if not stdout.strip():
        return [], []
    context = json.loads(stdout)["hookSpecificOutput"]["additionalContext"]
    # shortcut: a note body line shaped like a block header would be counted, fine for a probe
    return [int(n) for n in _DECISION_HEAD.findall(context)], _NOTE_HEAD.findall(context)


def percentile(values: list[float], q: int) -> float | None:
    """Nearest-rank percentile, or None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(q / 100 * len(ordered)) - 1, 0)]


# ---- seed-notes ---------------------------------------------------------------


def seed_notes(src: str, dst: str, *, from_key: str, to_key: str, author: str) -> tuple[int, int]:
    """Copy the caller's notes from src to dst; (copied, failed)."""
    rows = rest(f"{src}/notes?limit={LIST_LIMIT}", api_key=from_key, author=author)
    if len(rows) >= LIST_LIMIT:
        print(f"warning: {len(rows)} notes came back; there may be more to copy", file=sys.stderr)
    copied = failed = 0
    for row in rows:
        body = {
            "content": row["text"],
            "kind": row["kind"],
            "tags": row["tags"],
            "namespace": row["namespace"],
            "occurred_at": row["date"],
            "allow_similar": True,
        }
        if row.get("shared"):
            body["shared"] = True
        try:
            rest(f"{dst}/save_memory", api_key=to_key, author=author, payload=body)
            copied += 1
        except RequestFailed:
            failed += 1
    return copied, failed


# ---- run ----------------------------------------------------------------------


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recall_sweep(base: str, cases: list[dict], scope: str, *, api_key: str, author: str) -> dict:
    """The score table {(case id, tail): decisions, notes, ms} from POST /recall, no hook."""
    scores = {}
    for case in cases:
        for length in TAILS:
            query = build_query(case["prompt"], case["assistant_tail"], length)
            entry = {"decisions": [], "notes": [], "ms": None}
            if query:
                body = {
                    "query": query,
                    "scope": scope,
                    "notes": {"top_k": hook_search.ASK, "min_score": 0},
                    "decisions": {"top_k": hook_search.ASK, "min_score": 0},
                }
                started = time.monotonic()
                try:
                    reply = rest(f"{base}/recall", api_key=api_key, author=author, payload=body)
                    entry["ms"] = (time.monotonic() - started) * 1000
                    entry["decisions"] = [
                        {"number": d["number"], "score": d["score"]} for d in reply["decisions"]
                    ]
                    entry["notes"] = [{"id": n["id"], "score": n["score"]} for n in reply["notes"]]
                except RequestFailed:
                    entry["error"] = True
            scores[case["id"], length] = entry
    return scores


def run_hook(case: dict, length: int, floor: float, cwd: str, env: dict) -> dict:
    """One case through the real hook process in a fresh plugin data dir."""
    with tempfile.TemporaryDirectory() as data:
        session = f"probe-{case['id']}"
        tail = case["assistant_tail"][-length:] if length else ""
        if tail:
            (Path(data) / "sessions").mkdir()
            (Path(data) / "sessions" / f"{session}.tail").write_text(tail, encoding="utf-8")
        payload = json.dumps({"prompt": case["prompt"], "session_id": session, "cwd": cwd})
        started = time.monotonic()
        try:
            done = subprocess.run(
                [sys.executable, "-c", WRAPPER, str(HOOKS), str(floor), str(length)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=HOOK_TIMEOUT_S,
                env={**env, "CLAUDE_PLUGIN_DATA": data},
            )
            stdout = done.stdout
        except subprocess.TimeoutExpired:
            stdout = ""
        ms = (time.monotonic() - started) * 1000
        try:
            log = (Path(data) / "recall.log").read_text().splitlines()[-1]
        except (OSError, IndexError):
            log = ""  # the deadline killed the hook before it logged
        decisions, notes = parse_hook_output(stdout)
    return {
        "decisions": decisions,
        "notes": notes,
        "ms": ms,
        # a prompt the hook does not query leaves no log and is no cutoff
        "cutoff": bool(build_query(case["prompt"], tail, length))
        and (not log or "errors=unavailable" in log),
    }


def make_repo(root: str, repo: str) -> str:
    """A temp git repository whose origin is the probed repository."""
    for args in (["init", "-q"], ["remote", "add", "origin", origin_of(repo)]):
        subprocess.run(["git", "-C", root, *args], check=True, capture_output=True)
    return root


def timing(values: list[float]) -> dict:
    """Cold (first call), warm p50/p95 (the rest) and overall p50/p95, in ms."""
    warm = values[1:]
    return {
        "cold": values[0] if values else None,
        "warm_p50": percentile(warm, 50),
        "warm_p95": percentile(warm, 95),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def probe(
    base: str, repo: str, cases: list[dict], issue_map: dict[int, int], *, api_key: str, author: str
):
    """The whole report dict: sweep, frozen pair, hook run."""
    for case in cases:
        gold_numbers(case, issue_map)  # fails before any call
    calibration, held_out = split_cases(cases)
    scores = recall_sweep(base, cases, scope_of(repo), api_key=api_key, author=author)
    length, floor, calibration_hits = choose(calibration, scores, issue_map)
    env = {
        **os.environ,
        "CLAUDE_PLUGIN_OPTION_URL": base,
        "CLAUDE_PLUGIN_OPTION_API_KEY": api_key,
        "CLAUDE_PLUGIN_OPTION_AUTHOR": author,
    }
    with tempfile.TemporaryDirectory() as root:
        cwd = make_repo(root, repo)
        runs = {c["id"]: run_hook(c, length, floor, cwd, env) for c in cases}
    emitted = {case_id: run["decisions"] for case_id, run in runs.items()}
    held_ids = {c["id"] for c in held_out}
    notes = {
        kind: [len(r["notes"]) for c in cases if c["kind"] == kind for r in [runs[c["id"]]]]
        for kind in ("decision", "unrelated")
    }
    held = summarize(held_out, emitted, issue_map)
    everything = summarize(cases, emitted, issue_map)
    return {
        "chosen": {"tail": length, "decision_floor": floor, "calibration_hits": calibration_hits},
        "decisions_in_map": len(issue_map),
        "calibration": {"cases": len(calibration), **summarize(calibration, emitted, issue_map)},
        "held_out": held,
        "all": everything,
        "pass": held["hits"] >= 0.9 * held["decision_cases"]
        and everything["false_positives"] <= MAX_FALSE_POSITIVE * everything["unrelated_cases"],
        "notes_emitted": {
            kind: {"total": sum(counts), "cases_with_notes": sum(bool(n) for n in counts)}
            for kind, counts in notes.items()
        },
        "hook_cutoffs": sorted(case_id for case_id, run in runs.items() if run["cutoff"]),
        "hook_ms": timing([runs[c["id"]]["ms"] for c in cases]),
        "recall_ms": timing([e["ms"] for e in scores.values() if e["ms"] is not None]),
        "grid": sweep_grid(calibration, scores, issue_map),
        "cases": [
            {
                "id": c["id"],
                "kind": c["kind"],
                "split": "held_out" if c["id"] in held_ids else "calibration",
                "hit": is_hit(gold_numbers(c, issue_map), emitted[c["id"]]),
                "decisions": emitted[c["id"]],
                "notes": len(runs[c["id"]]["notes"]),
                "cutoff": runs[c["id"]]["cutoff"],
                "scores": {str(n): scores[c["id"], n] for n in TAILS},
            }
            for c in cases
        ],
    }


def _ms(stats: dict) -> str:
    fmt = lambda v: "n/a" if v is None else f"{v:.0f} ms"  # noqa: E731
    return (
        f"cold {fmt(stats['cold'])}, warm p50 {fmt(stats['warm_p50'])} / "
        f"p95 {fmt(stats['warm_p95'])}"
    )


def markdown(report: dict) -> str:
    chosen, held, everything = report["chosen"], report["held_out"], report["all"]
    rate = (
        everything["false_positives"] / everything["unrelated_cases"]
        if everything["unrelated_cases"]
        else 0
    )
    notes = report["notes_emitted"]
    return "\n".join(
        [
            f"# Recall probe: {'PASS' if report['pass'] else 'FAIL'}",
            "",
            f"- Frozen on the calibration half: tail {chosen['tail']} chars, decision floor "
            f"{chosen['decision_floor']} (calibration hits {chosen['calibration_hits']}/"
            f"{report['calibration']['decision_cases']}); notes floor 0.5",
            f"- Held-out gold hits: {held['hits']}/{held['decision_cases']}",
            f"- All cases gold hits: {everything['hits']}/{everything['decision_cases']}",
            f"- Unrelated false positives (decision emitted): {everything['false_positives']}/"
            f"{everything['unrelated_cases']} ({rate:.0%})",
            f"- Notes emitted: decision cases {notes['decision']['total']} "
            f"({notes['decision']['cases_with_notes']} cases), unrelated {notes['unrelated']['total']} "
            f"({notes['unrelated']['cases_with_notes']} cases)",
            f"- Hook process: {_ms(report['hook_ms'])}",
            f"- /recall round trip with {report['decisions_in_map']} decisions: {_ms(report['recall_ms'])}",
            f"- Hook cutoffs or no log (counted as misses): {', '.join(report['hook_cutoffs']) or 'none'}",
            "",
        ]
    )


# ---- cli ----------------------------------------------------------------------


def parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    seed = sub.add_parser("seed-notes", help="copy the caller's notes to another server")
    seed.add_argument("--from", dest="src", required=True, help="source REST base URL")
    seed.add_argument("--to", dest="dst", required=True, help="target REST base URL")
    run = sub.add_parser("run", help="calibrate, freeze and check the recall")
    run.add_argument("--repo", required=True, metavar="OWNER/NAME", help="GitHub repository")
    run.add_argument("--url", required=True, help="REST base URL of the sandbox server")
    run.add_argument("--cases", required=True, help="local JSONL case file")
    run.add_argument(
        "--map", required=True, help="the importer's printed '#<issue> -> D-<n>' lines"
    )
    run.add_argument("--out", default="probe-report.json", help="JSON report path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    author = os.environ.get("MEMORY_BASE_AUTHOR")
    api_key = os.environ.get("MEMORY_BASE_API_KEY")
    if not author or not api_key:
        print("error: MEMORY_BASE_API_KEY and MEMORY_BASE_AUTHOR must be set", file=sys.stderr)
        return 2
    try:
        if args.command == "seed-notes":
            src, dst = args.src.rstrip("/"), args.dst.rstrip("/")
            to_key = os.environ.get("MEMORY_BASE_TO_API_KEY")
            if src == dst or not to_key:
                print(
                    "error: --from and --to must differ, and MEMORY_BASE_TO_API_KEY must be set",
                    file=sys.stderr,
                )
                return 2
            copied, failed = seed_notes(src, dst, from_key=api_key, to_key=to_key, author=author)
            print(f"copied {copied}, failed {failed}")
            return 0 if not failed else 1
        issue_map = parse_map(Path(args.map).read_text())
        report = probe(
            args.url.rstrip("/"),
            args.repo,
            load_cases(args.cases),
            issue_map,
            api_key=api_key,
            author=author,
        )
    except (RequestFailed, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
