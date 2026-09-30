"""Read-settings sweep: every top_k / min_score / budget_tokens point from one search per query.

Each query is searched once with a budget no packet reaches, which returns every fused
candidate in rerank order; each grid point is then cut from that list with the production
helpers (`_apply_min_score`, `_pack_budget`), so every point sees the same candidates and
scores. A budget point with a floor drops candidates below it before packing. Each read path
then delivers the cut hits the way its client does: `search` returns them all, `claude-code`
renders the prefetch hook's context block, and `hermes` renders the Hermes provider's prefetch.

Corpora: the LongMemEval personal notes (`longmemeval retrieve --read candidates --gate off`
writes the candidate packets; `lme-probe` searches the probe prompts in one question's
namespace), and the deployed corpus (`deployed` replays the labelled notes queries and the
probe prompts over a read-only connection).

CLI:
  uv run python -m memory_base.eval.read_sweep lme-probe --dataset PATH --data-dir DIR
  uv run python -m memory_base.eval.read_sweep deployed --out FILE
  uv run python -m memory_base.eval.read_sweep report --dataset PATH --data-dir DIR [--deployed FILE]
  uv run python -m memory_base.eval.read_sweep export --data-dir DIR --setting k5-f0.6 \
      --path search --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import importlib.util
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory_base.eval import longmemeval as lme
from memory_base.retrieval import search as search_module

REPO_ROOT = lme.REPO_ROOT
PROBES_PATH = REPO_ROOT / "tests" / "fixtures" / "read_sweep_probes.jsonl"
PREFETCH_HOOK_PATH = REPO_ROOT / "integrations" / "claude_code" / "prefetch_hook.py"
HERMES_CLIENT_PATH = REPO_ROOT / "integrations" / "hermes" / "memory_base" / "client.py"
CANDIDATES_BUDGET = lme.CANDIDATES_BUDGET
CANDIDATES_RUN = lme.run_name("baseline", "off", "candidates")
EXPORT_RUN = lme.run_name("baseline", "off")
PROBES_FILE = "probe-candidates.jsonl"
REPORT_FILE = "read-sweep.json"
TOP_KS = (3, 5, 10)
TOP_K_FLOORS = (0.0, 0.25, 0.4, 0.6)
BUDGETS = (800, 1500, 2500, 4000)
BUDGET_FLOORS = (0.0, 0.25, 0.4)
PATHS = ("search", "claude-code", "hermes")
INTENTS = ("off_topic", "memory")
# Every API hit date renders as YYYY-MM-DD, so a fixed date gives the clients' exact lengths.
DATE_PLACEHOLDER = "2000-01-01"
SETTING_RE = re.compile(r"^(k|b)(\d+)-f(\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class ReadSetting:
    top_k: int | None = None
    min_score: float = 0.0
    budget_tokens: int | None = None

    @property
    def name(self) -> str:
        if self.budget_tokens is None:
            return f"k{self.top_k}-f{self.min_score:g}"
        return f"b{self.budget_tokens}-f{self.min_score:g}"


def grid() -> list[ReadSetting]:
    top_k = [ReadSetting(top_k=k, min_score=f) for k in TOP_KS for f in TOP_K_FLOORS]
    budget = [ReadSetting(min_score=f, budget_tokens=b) for b in BUDGETS for f in BUDGET_FLOORS]
    return top_k + budget


def parse_setting(name: str) -> ReadSetting:
    match = SETTING_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"not a read setting: {name!r} (k<top_k>-f<floor> or b<budget>-f<floor>)")
    mode, size, floor = match.groups()
    if mode == "k":
        return ReadSetting(top_k=int(size), min_score=float(floor))
    return ReadSetting(min_score=float(floor), budget_tokens=int(size))


def _text(row: dict[str, Any]) -> str:
    # A deployed row keeps only its length; a stand-in of that length measures it the same.
    return row["text"] if "text" in row else "x" * row["chars"]


def _as_hit(row: dict[str, Any]) -> search_module.Hit:
    return search_module.Hit(
        source="memory",
        ref=str(row["id"]),
        text=_text(row),
        ts=0.0,
        rerank_score=row["score"],
        meta={"row": row},
    )


def hit_tokens(row: dict[str, Any]) -> int:
    return search_module.estimate_tokens(_as_hit(row))


def apply_setting(setting: ReadSetting, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The hits search() and the API return for the setting, from every candidate in rank order."""
    hits = [_as_hit(row) for row in rows]
    if setting.budget_tokens is None:
        top = hits[: search_module.RERANK_TOP]
        kept = search_module._apply_min_score(top, setting.min_score, True)[: setting.top_k]
    else:
        floored = search_module._apply_min_score(hits, setting.min_score, True)
        kept = search_module._pack_budget(floored, setting.budget_tokens)
    return [hit.meta["row"] for hit in kept]


def _load_integration(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # A dataclass resolves its annotations through sys.modules while the module executes.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@functools.cache
def _prefetch_hook():
    return _load_integration("memory_base_sweep_prefetch_hook", PREFETCH_HOOK_PATH)


@functools.cache
def _hermes_client():
    return _load_integration("memory_base_sweep_hermes_client", HERMES_CLIENT_PATH)


def _client_block(path: str, api_hits: list[dict[str, Any]]) -> str:
    if path == "claude-code":
        return _prefetch_hook().build_context_block(api_hits)
    client = _hermes_client().MemoryBaseClient(url="", api_key="")
    client.search = lambda query: api_hits
    return client.build_prefetch("sweep")


def deliver(path: str, rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The hits a read path puts into the prompt and their estimated tokens (chars / 4)."""
    rows = list(rows)
    if not rows:
        return [], 0
    if path == "search":
        return rows, sum(hit_tokens(row) for row in rows)
    api_hits = [{"date": DATE_PLACEHOLDER, "text": _text(row)} for row in rows]
    block = _client_block(path, api_hits)
    # Both clients cut at a line boundary, so the shortest prefix rendering the same block
    # is the set of hits that reached the prompt.
    kept = next(n for n in range(len(rows) + 1) if _client_block(path, api_hits[:n]) == block)
    return rows[:kept], len(block) // 4


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _is_junk(row: dict[str, Any], answers: set[str]) -> bool:
    return not any(sid in answers for sid, _date in row["sessions"])


def lme_metrics(
    packets: Sequence[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    setting: ReadSetting,
    path: str,
) -> dict[str, Any]:
    """Session recall, tokens, junk share and zero-hit rate of one setting on one read path."""
    derived, tokens = [], []
    for packet in packets:
        hits, used = deliver(path, apply_setting(setting, packet["hits"]))
        derived.append({**packet, "hits": hits})
        tokens.append(used)
    overall = lme.retrieval_metrics(derived, questions)["overall"]
    recall, hits_seen, junk = [], 0, 0
    for packet in derived:
        if "_abs" in packet["question_id"]:
            continue
        answers = set(questions[packet["question_id"]]["answer_session_ids"])
        ranking = lme.session_ranking(packet["hits"])
        recall.append(lme.recall_all_at_k(ranking, answers, k=len(ranking)))
        hits_seen += len(packet["hits"])
        junk += sum(_is_junk(hit, answers) for hit in packet["hits"])
    return {
        "recall_all": _mean(recall),
        "recall_all@10": overall["recall_all@10"],
        "ndcg_any@10": overall["ndcg_any@10"],
        "mean_hits": _mean([len(packet["hits"]) for packet in derived]),
        "mean_tokens": _mean(tokens),
        "junk_share": junk / hits_seen if hits_seen else None,
        "zero_hit_rate": sum(not packet["hits"] for packet in derived) / len(derived),
    }


def probe_metrics(
    rows: Sequence[dict[str, Any]], setting: ReadSetting, path: str
) -> dict[str, dict[str, Any]]:
    """Per intent: how often a prompt injects anything, and how much."""
    report: dict[str, dict[str, Any]] = {}
    for intent in INTENTS:
        group = [row for row in rows if row["intent"] == intent]
        delivered = [deliver(path, apply_setting(setting, row["hits"])) for row in group]
        report[intent] = {
            "prompts": len(group),
            "fire_rate": _mean([float(bool(hits)) for hits, _ in delivered]),
            "mean_hits": _mean([float(len(hits)) for hits, _ in delivered]),
            "mean_tokens": _mean([float(tokens) for _, tokens in delivered]),
        }
    return report


def replay_metrics(
    rows: Sequence[dict[str, Any]],
    corpus_ids: set[str],
    setting: ReadSetting,
    path: str,
) -> dict[str, Any]:
    """The notes replay's recall@5 / MRR@10 and expect-empty checks, with tokens and junk.

    Junk counts hits outside a scored label's relevant ids, so it bounds junk from above:
    an unlabelled hit may still be relevant.
    """
    from memory_base.eval import retrieval

    labels, retrieved, tokens = [], [], []
    junk = hits_seen = 0
    for row in rows:
        label = retrieval.EvalLabel(
            row["query"], row["query_class"], tuple(row["relevant_ids"]), row["expect_empty"]
        )
        hits, used = deliver(path, apply_setting(setting, row["hits"]))
        labels.append(label)
        retrieved.append([hit["id"] for hit in hits])
        tokens.append(used)
        relevant = set(label.relevant_ids) & corpus_ids
        if not label.expect_empty and relevant:
            hits_seen += len(hits)
            junk += sum(hit["id"] not in relevant for hit in hits)
    report = retrieval.score_notes_replay(labels, retrieved, corpus_ids)
    overall = retrieval.aggregate_metrics(report.results)["overall"]
    return {
        "scored": overall.count,
        "recall_at_5": overall.recall_at_5,
        "mrr_at_10": overall.mrr_at_10,
        "expect_empty_passed": report.expect_empty_passed,
        "expect_empty_total": report.expect_empty_total,
        "decayed": report.decayed,
        "mean_tokens": _mean(tokens),
        "junk_share": junk / hits_seen if hits_seen else None,
        "zero_hit_rate": retrieval.count_zero_hit_results(retrieved) / len(rows),
    }


def load_probes(path: Path = PROBES_PATH) -> list[dict[str, Any]]:
    probes = lme.read_jsonl(path)
    for probe in probes:
        if probe.get("intent") not in INTENTS or not str(probe.get("query", "")).strip():
            raise ValueError(f"invalid probe: {probe!r}")
    return probes


def read_only_url(url: str) -> str:
    """asyncpg passes unknown DSN query parameters as server settings."""
    return url + ("&" if "?" in url else "?") + "default_transaction_read_only=on"


async def probe_question(
    question: dict[str, Any],
    notes_by_unit: dict,
    gate: str,
    probes: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load one question's notes and search every probe prompt in its namespace."""
    from memory_base.eval import retrieval
    from memory_base.serve import namespaces

    namespace = lme.NAMESPACE_PREFIX + question["question_id"]
    await namespaces.create_namespace(namespace)
    await lme.load_question_notes(namespace, lme.session_units(question), notes_by_unit, gate=gate)
    rows = []
    for probe in probes:
        hits = await retrieval._search_with_retry(
            probe["query"],
            source="memory",
            namespaces=[namespace],
            budget_tokens=CANDIDATES_BUDGET,
        )
        rows.append(
            {
                "intent": probe["intent"],
                "query": probe["query"],
                "hits": [{"id": h.meta["id"], "score": h.score, "text": h.text} for h in hits],
            }
        )
    return rows


async def _probe_in_database(url: str, *args: Any) -> list[dict[str, Any]]:
    from memory_base.core import db

    await lme._prepare_schema(url)
    try:
        return await probe_question(*args)
    finally:
        await db.close_pool()


def run_lme_probe(args: argparse.Namespace) -> None:
    from memory_base.serve import notes

    subset = lme.select_subset(lme.load_dataset(args.dataset))
    question = (
        subset[0] if args.question is None else lme.filter_questions(subset, [args.question])[0]
    )
    notes_by_unit, _ = lme._notes_by_unit(args.data_dir)
    notes.judge_note_content = lme._gate_pinned_open
    with lme.throwaway_postgres() as database:
        rows = asyncio.run(
            _probe_in_database(
                database["url"], question, notes_by_unit, args.gate, load_probes(args.probes)
            )
        )
    lme.write_jsonl_atomic(args.data_dir / PROBES_FILE, rows)
    print(f"{len(rows)} probes searched in lme-{question['question_id']}")


async def _collect_deployed(probes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import asyncpg

    from memory_base.core import db
    from memory_base.core.config import PG_SCHEMA, db_url
    from memory_base.eval import retrieval

    conn = await asyncpg.connect(db_url(), timeout=5)
    try:
        corpus = await conn.fetch(
            f'SELECT id FROM "{PG_SCHEMA}".memory_chunks WHERE archived_at IS NULL'
        )
    finally:
        await conn.close()

    async def candidates(query: str) -> list[dict[str, Any]]:
        hits = await retrieval._search_with_retry(
            query, source="memory", budget_tokens=CANDIDATES_BUDGET
        )
        return [{"id": h.meta["id"], "score": h.score, "chars": len(h.text)} for h in hits]

    try:
        labels = [
            {
                "query": label.query,
                "query_class": label.query_class,
                "relevant_ids": list(label.relevant_ids),
                "expect_empty": label.expect_empty,
                "hits": await candidates(label.query),
            }
            for label in retrieval.load_labels(retrieval.NOTES_LABELS_PATH)
        ]
        probe_rows = [
            {
                "intent": probe["intent"],
                "query": probe["query"],
                "hits": await candidates(probe["query"]),
            }
            for probe in probes
        ]
    finally:
        await db.close_pool()
    return {
        "corpus_ids": sorted(row["id"] for row in corpus),
        "labels": labels,
        "probes": probe_rows,
    }


def run_deployed(args: argparse.Namespace) -> None:
    from memory_base.core.config import db_url

    os.environ["DB_URL"] = read_only_url(db_url())
    collected = asyncio.run(_collect_deployed(load_probes(args.probes)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(collected, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"corpus={len(collected['corpus_ids'])} labels={len(collected['labels'])} "
        f"probes={len(collected['probes'])} -> {args.out}"
    )


def sweep_report(
    packets: Sequence[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    lme_probes: Sequence[dict[str, Any]],
    deployed: dict[str, Any] | None,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for path in PATHS:
        rows = {}
        for setting in grid():
            row: dict[str, Any] = {"lme": lme_metrics(packets, questions, setting, path)}
            if lme_probes:
                row["lme_probes"] = probe_metrics(lme_probes, setting, path)
            if deployed is not None:
                corpus = set(deployed["corpus_ids"])
                row["deployed"] = replay_metrics(deployed["labels"], corpus, setting, path)
                row["deployed_probes"] = probe_metrics(deployed["probes"], setting, path)
            rows[setting.name] = row
        report[path] = rows
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for path, rows in report.items():
        lines += [
            f"## {path}",
            "",
            "| setting | recall_all | recall_all@10 | ndcg_any@10 | hits | tokens | junk "
            "| zero-hit | LME off-topic fire | deployed R@5 | deployed MRR@10 | deployed "
            "expect-empty | deployed zero-hit | deployed off-topic fire | deployed off-topic "
            "tokens | deployed memory fire |",
            "|---" * 16 + "|",
        ]
        for name, row in rows.items():
            lme_row = row["lme"]
            cells = [
                name,
                lme_row["recall_all"],
                lme_row["recall_all@10"],
                lme_row["ndcg_any@10"],
                lme_row["mean_hits"],
                lme_row["mean_tokens"],
                lme_row["junk_share"],
                lme_row["zero_hit_rate"],
                row.get("lme_probes", {}).get("off_topic", {}).get("fire_rate"),
            ]
            deployed = row.get("deployed")
            probes = row.get("deployed_probes", {})
            cells += (
                [
                    deployed["recall_at_5"],
                    deployed["mrr_at_10"],
                    f"{deployed['expect_empty_passed']}/{deployed['expect_empty_total']}",
                    deployed["zero_hit_rate"],
                    probes["off_topic"]["fire_rate"],
                    probes["off_topic"]["mean_tokens"],
                    probes["memory"]["fire_rate"],
                ]
                if deployed
                else [None] * 7
            )
            lines.append("| " + " | ".join(_fmt(cell) for cell in cells) + " |")
        lines.append("")
    return "\n".join(lines)


def run_report(args: argparse.Namespace) -> None:
    questions = {q["question_id"]: q for q in lme.load_dataset(args.dataset)}
    packets = lme.read_jsonl(lme.packets_path(args.data_dir, CANDIDATES_RUN))
    if not packets:
        raise SystemExit(f"no {CANDIDATES_RUN} packets; run longmemeval retrieve --read candidates")
    lme_probes = lme.read_jsonl(args.data_dir / PROBES_FILE)
    deployed = json.loads(args.deployed.read_text(encoding="utf-8")) if args.deployed else None
    report = sweep_report(packets, questions, lme_probes, deployed)
    (args.data_dir / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    print(render_report(report))


def export_packets(data_dir: Path, setting: ReadSetting, path: str, out_dir: Path) -> Path:
    """Write the path's delivered hits as the packets the answer stage reads."""
    rows = []
    for packet in lme.read_jsonl(lme.packets_path(data_dir, CANDIDATES_RUN)):
        hits, _ = deliver(path, apply_setting(setting, packet["hits"]))
        rows.append(
            {
                **packet,
                "run": EXPORT_RUN,
                "read": {"setting": setting.name, "path": path},
                "hits": hits,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = lme.packets_path(out_dir, EXPORT_RUN)
    lme.write_jsonl_atomic(out_path, rows)
    return out_path


def run_export(args: argparse.Namespace) -> None:
    out_path = export_packets(args.data_dir, parse_setting(args.setting), args.path, args.out)
    print(f"{args.setting} ({args.path}) -> {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m memory_base.eval.read_sweep")
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("lme-probe", help="search the probe prompts in one LME namespace")
    probe.add_argument("--dataset", type=Path, required=True)
    probe.add_argument("--data-dir", type=Path, required=True)
    probe.add_argument("--question", default=None)
    probe.add_argument("--gate", choices=lme.GATES, default="off")
    probe.add_argument("--probes", type=Path, default=PROBES_PATH)
    deployed = commands.add_parser("deployed", help="replay labels and probes, read-only")
    deployed.add_argument("--out", type=Path, required=True)
    deployed.add_argument("--probes", type=Path, default=PROBES_PATH)
    report = commands.add_parser("report", help="metrics for every setting and read path")
    report.add_argument("--dataset", type=Path, required=True)
    report.add_argument("--data-dir", type=Path, required=True)
    report.add_argument("--deployed", type=Path, default=None)
    export = commands.add_parser("export", help="packets of one setting for answer.py")
    export.add_argument("--data-dir", type=Path, required=True)
    export.add_argument("--setting", required=True)
    export.add_argument("--path", choices=PATHS, default="search")
    export.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    {
        "lme-probe": run_lme_probe,
        "deployed": run_deployed,
        "report": run_report,
        "export": run_export,
    }[args.command](args)


if __name__ == "__main__":
    main()
