#!/usr/bin/env python3
"""Build and score a replay set of real logged memory queries.

`replay` fetches retrieval_log rows, samples 50 originally-empty and 50
originally-with-hits queries deterministically, replays each through hybrid
search, and writes the top-10 candidates per query to <out>/candidates.jsonl.
A reviewer then labels the relevant candidates in <out>/labels.jsonl, one line
per query: {"log_id": <id>, "relevant": [<1-based candidate numbers>]}.
`report` joins the two files into the eval fixture, a spot-check sheet, and
the A/B report.

Usage:
  uv run --env-file .env python scripts/label_notes_replay.py replay --out <dir>
  uv run python scripts/label_notes_replay.py report --out <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import asyncpg

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.eval.retrieval import classify_query_shape
from memory_base.retrieval import search as search_module

CUTOFF_EPOCH = datetime(2026, 8, 28, tzinfo=timezone.utc).timestamp()
TARGET_PER_HALF = 50
ROUND_ROBIN_CAP = 13
SHAPE_ORDER = ("cron_desire_tick", "cron_other", "keyword", "free_text")
CANDIDATE_TEXT_LIMIT = 600
SPOTCHECK_QUERY_LIMIT = 200
SPOTCHECK_TEXT_LIMIT = 160
SPOTCHECK_A = 5
SPOTCHECK_B = 5
SPOTCHECK_WITH_HITS = 10
TOP_K = 10
REPLAY_RETRY_ATTEMPTS = 3
REPLAY_RETRY_BACKOFF_SECONDS = 10.0
FIXTURE_PATH = Path("tests/fixtures/retrieval_eval_notes.jsonl")


@dataclass(frozen=True)
class LogRow:
    log_id: int
    query: str
    hit_ids: tuple[str, ...]
    namespaces: list[str] | None
    shape: str

    @property
    def original_empty(self) -> bool:
        return not self.hit_ids


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    score: float
    text: str


@dataclass(frozen=True)
class Judgment:
    row: LogRow
    candidates: tuple[Candidate, ...]
    relevant_ids: frozenset[str]

    @property
    def label_class(self) -> str:
        if not self.row.original_empty:
            return "with_hits"
        return "A" if self.relevant_ids else "B"

    @property
    def korean(self) -> bool:
        return contains_hangul(self.row.query)

    @property
    def original_hit_overlap(self) -> float:
        hits = set(self.row.hit_ids)
        if not hits:
            return 0.0
        return len(hits & set(self.relevant_ids)) / len(hits)


def contains_hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


async def fetch_rows() -> list[LogRow]:
    """Load source='memory'/'all' rows since the cutoff, deduped by query text."""
    conn = await asyncpg.connect(db_url(), timeout=10)
    try:
        rows = await conn.fetch(
            f'SELECT id, query, hit_ids, filters FROM "{PG_SCHEMA}".retrieval_log '
            "WHERE source IN ('memory', 'all') AND ts >= $1 ORDER BY md5(id::text)",
            CUTOFF_EPOCH,
        )
    finally:
        await conn.close()
    out: list[LogRow] = []
    seen: set[str] = set()
    for row in rows:
        if row["query"] in seen:
            continue
        seen.add(row["query"])
        filters = json.loads(row["filters"]) if isinstance(row["filters"], str) else row["filters"]
        namespaces = filters.get("namespaces") if isinstance(filters, dict) else None
        out.append(
            LogRow(
                log_id=row["id"],
                query=row["query"],
                hit_ids=tuple(row["hit_ids"]),
                namespaces=namespaces if isinstance(namespaces, list) and namespaces else None,
                shape=classify_query_shape(row["query"]),
            )
        )
    return out


def sample_half(rows: list[LogRow], target: int = TARGET_PER_HALF) -> list[LogRow]:
    """Round-robin up to ROUND_ROBIN_CAP rows per shape, then fill from leftover free_text."""
    queues = {shape: deque(r for r in rows if r.shape == shape) for shape in SHAPE_ORDER}
    taken = dict.fromkeys(SHAPE_ORDER, 0)
    picked: list[LogRow] = []
    while len(picked) < target:
        progressed = False
        for shape in SHAPE_ORDER:
            if len(picked) >= target:
                break
            if taken[shape] < ROUND_ROBIN_CAP and queues[shape]:
                picked.append(queues[shape].popleft())
                taken[shape] += 1
                progressed = True
        if not progressed:
            break
    while len(picked) < target and queues["free_text"]:
        picked.append(queues["free_text"].popleft())
    return picked


async def replay(row: LogRow) -> list[Candidate]:
    for attempt in range(1, REPLAY_RETRY_ATTEMPTS + 1):
        try:
            hits = await search_module.search(
                row.query, source="memory", rerank=True, min_score=0, namespaces=row.namespaces
            )
            break
        except search_module.UpstreamUnavailable:
            # ponytail: the rerank endpoint stalls a connection now and then; a pause and retry clears it.
            if attempt == REPLAY_RETRY_ATTEMPTS:
                raise
            await asyncio.sleep(REPLAY_RETRY_BACKOFF_SECONDS * attempt)
    return [
        Candidate(chunk_id, hit.score, hit.text[:CANDIDATE_TEXT_LIMIT])
        for hit in hits[:TOP_K]
        if isinstance((chunk_id := hit.meta.get("id")), str)
    ]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def row_from_record(record: dict) -> LogRow:
    return LogRow(
        log_id=record["log_id"],
        query=record["query"],
        hit_ids=tuple(record["hit_ids"]),
        namespaces=record["namespaces"],
        shape=record["shape"],
    )


def parse_label(payload: object, candidate_count: int) -> list[int] | None:
    """Return in-range candidate numbers from a label record, or None when malformed."""
    numbers = payload.get("relevant") if isinstance(payload, dict) else None
    if not isinstance(numbers, list):
        return None
    return [
        number
        for number in numbers
        if isinstance(number, int)
        and not isinstance(number, bool)
        and 1 <= number <= candidate_count
    ]


def join_labels(candidate_records: list[dict], label_records: list[dict]) -> list[Judgment]:
    """Pair every candidate row with its label; a missing or malformed label is an error."""
    labels = {record.get("log_id"): record for record in label_records}
    judgments: list[Judgment] = []
    problems: list[str] = []
    for record in candidate_records:
        candidates = tuple(Candidate(c["id"], c["score"], c["text"]) for c in record["candidates"])
        numbers = parse_label(labels.get(record["log_id"]), len(candidates))
        if numbers is None:
            problems.append(str(record["log_id"]))
            continue
        judgments.append(
            Judgment(
                row=row_from_record(record),
                candidates=candidates,
                relevant_ids=frozenset(candidates[n - 1].chunk_id for n in numbers),
            )
        )
    if problems:
        raise ValueError("missing or malformed labels for log_id: " + ", ".join(problems))
    return judgments


def write_fixture(judgments: list[Judgment], path: Path) -> int:
    lines = [
        json.dumps(
            {
                "query": judgment.row.query,
                "query_class": judgment.row.shape,
                "relevant_ids": sorted(judgment.relevant_ids),
            },
            ensure_ascii=False,
        )
        for judgment in judgments
        if judgment.relevant_ids
    ]
    lines.sort(key=lambda line: json.loads(line)["query"])
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def spotcheck_rows(judgments: list[Judgment]) -> list[Judgment]:
    class_a = [j for j in judgments if j.label_class == "A"]
    class_b = [j for j in judgments if j.label_class == "B"]
    with_hits = [j for j in judgments if j.label_class == "with_hits"]
    return class_a[:SPOTCHECK_A] + class_b[:SPOTCHECK_B] + with_hits[:SPOTCHECK_WITH_HITS]


def write_spotcheck(judgments: list[Judgment], path: Path) -> None:
    lines = ["# Notes replay spot check", ""]
    for judgment in spotcheck_rows(judgments):
        heading = f"## log {judgment.row.log_id} — shape={judgment.row.shape} class={judgment.label_class}"
        if judgment.label_class == "with_hits":
            heading += f" original_overlap={judgment.original_hit_overlap:.2f}"
        lines += [heading, f"query: {judgment.row.query[:SPOTCHECK_QUERY_LIMIT]}", "candidates:"]
        for index, candidate in enumerate(judgment.candidates, start=1):
            verdict = "RELEVANT" if candidate.chunk_id in judgment.relevant_ids else "not relevant"
            lines.append(
                f"  {index}. [{candidate.chunk_id}] {candidate.text[:SPOTCHECK_TEXT_LIMIT]} — {verdict}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.1f}" if denominator else "0.0"


def write_report(judgments: list[Judgment], path: Path) -> None:
    empty = [j for j in judgments if j.row.original_empty]
    with_hits = [j for j in judgments if not j.row.original_empty]
    class_a = [j for j in empty if j.relevant_ids]
    class_b = [j for j in empty if not j.relevant_ids]

    lines = ["# Notes replay labeling report", "", "## 1. Realized sample", ""]
    shapes = [shape for shape in SHAPE_ORDER if any(j.row.shape == shape for j in judgments)]
    lines.append("shape            empty  with_hits")
    for shape in shapes:
        e = sum(1 for j in empty if j.row.shape == shape)
        h = sum(1 for j in with_hits if j.row.shape == shape)
        lines.append(f"{shape:<16} {e:5d}  {h:9d}")
    lines.append(f"{'total':<16} {len(empty):5d}  {len(with_hits):9d}")

    lines += [
        "",
        "## 2. Empty queries: retrieval miss (A) vs coverage gap (B)",
        "",
        "class  count  % of empty",
        f"A      {len(class_a):5d}  {_pct(len(class_a), len(empty)):>10}",
        f"B      {len(class_b):5d}  {_pct(len(class_b), len(empty)):>10}",
        "",
        "per shape:",
        "shape            A   B    A%",
    ]
    for shape in shapes:
        shape_empty = [j for j in empty if j.row.shape == shape]
        a = sum(1 for j in shape_empty if j.relevant_ids)
        b = len(shape_empty) - a
        lines.append(f"{shape:<16} {a:3d} {b:3d}  {_pct(a, len(shape_empty)):>5}")

    korean_a = sum(1 for j in class_a if j.korean)
    korean_b = sum(1 for j in class_b if j.korean)
    mean_overlap = (
        f"{sum(j.original_hit_overlap for j in with_hits) / len(with_hits):.3f}"
        if with_hits
        else "n/a"
    )
    lines += [
        "",
        "## 3. Korean split",
        "",
        f"class A: korean {korean_a}/{len(class_a)} ({_pct(korean_a, len(class_a))}%), "
        f"non-korean {len(class_a) - korean_a}/{len(class_a)}",
        f"class B: korean {korean_b}/{len(class_b)} ({_pct(korean_b, len(class_b))}%), "
        f"non-korean {len(class_b) - korean_b}/{len(class_b)}",
        "",
        "## 4. With-hits queries",
        "",
        f"mean original_hits_judged_relevant: {mean_overlap}",
        f"queries with zero judged-relevant candidates: "
        f"{sum(1 for j in with_hits if not j.relevant_ids)}/{len(with_hits)}",
        "",
        "## 5. Harness output (python -m memory_base.eval.retrieval --notes)",
        "",
        "(appended after the harness run)",
        "",
        "## 6. Recommendation",
        "",
        "(appended after the harness run)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


async def run_replay(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = out_dir / "candidates.jsonl"
    done = {record["log_id"] for record in read_jsonl(candidates_path)}
    rows = await fetch_rows()
    sample = sample_half([r for r in rows if r.original_empty]) + sample_half(
        [r for r in rows if not r.original_empty]
    )
    realized = Counter((row.original_empty, row.shape) for row in sample)
    print(f"Pool: {len(rows)} unique queries; sampled {len(sample)}; already replayed {len(done)}")
    for (original_empty, shape), count in sorted(realized.items(), key=lambda item: item[0][1]):
        print(f"  {'empty' if original_empty else 'with_hits':<9} {shape:<16} {count}")
    with candidates_path.open("a", encoding="utf-8") as sink:
        for index, row in enumerate(sample, start=1):
            if row.log_id in done:
                continue
            candidates = await replay(row)
            record = {
                "log_id": row.log_id,
                "query": row.query,
                "shape": row.shape,
                "hit_ids": list(row.hit_ids),
                "namespaces": row.namespaces,
                "candidates": [
                    {"id": c.chunk_id, "score": c.score, "text": c.text} for c in candidates
                ],
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"{index:02d}/{len(sample):02d} log_id={row.log_id} shape={row.shape} "
                f"candidates={len(candidates)}"
            )
    print(f"candidates -> {candidates_path}; label them in {out_dir / 'labels.jsonl'}")


def run_report(out_dir: Path) -> None:
    judgments = join_labels(
        read_jsonl(out_dir / "candidates.jsonl"), read_jsonl(out_dir / "labels.jsonl")
    )
    fixture_count = write_fixture(judgments, FIXTURE_PATH)
    print(f"fixture: {fixture_count} labeled queries -> {FIXTURE_PATH}")
    write_spotcheck(judgments, out_dir / "spotcheck.md")
    write_report(judgments, out_dir / "report.md")
    print(f"spotcheck + report -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("replay", "report"))
    parser.add_argument(
        "--out", type=Path, required=True, help="directory for candidates, labels, and reports"
    )
    args = parser.parse_args()
    if args.command == "replay":
        asyncio.run(run_replay(args.out))
    else:
        run_report(args.out)


if __name__ == "__main__":
    main()
