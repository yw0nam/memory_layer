"""Reproducible document retrieval evaluation with an atom-lane A/B report."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg

from memory_base import common, schema as schema_module
from memory_base.common import DB_URL, EMB_MODEL, RERANK_MODEL
from memory_base.retrieval import search as search_module
from memory_base.retrieval.decompose import deep_search
from memory_base.serve import ingest_api

MULTI_HOP_CLASS = "multi_hop"
SINGLE_HOP_CLASSES = frozenset({"agent_work_history", "companion_fragment", "personal_kb"})

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "eval_docs"
LABELS_PATH = REPO_ROOT / "tests" / "fixtures" / "retrieval_eval.jsonl"
RELEVANT_ID_RE = re.compile(
    r"^doc:[a-z0-9][a-z0-9._-]{0,120}:(?:[0-9]+(?::atom:[0-9]+)?|card:[0-9]+)$"
)
REQUIRED_SERVICE_ENV = ("LLM_URL", "EMB_URL", "RERANK_URL")


@dataclass(frozen=True)
class EvalLabel:
    query: str
    query_class: str
    relevant_ids: tuple[str, ...]


@dataclass(frozen=True)
class QueryMetrics:
    query_class: str
    recall_at_5: float
    mrr_at_10: float
    missing_relevant_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricSummary:
    count: int
    recall_at_5: float
    mrr_at_10: float


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    *,
    k: int,
) -> float:
    """Return the fraction of labeled relevant rows retrieved in the first k results."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved_ids[:k])) / len(relevant)


def reciprocal_rank_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    *,
    k: int,
) -> float:
    """Return reciprocal rank of the first relevant row within the first k results."""
    relevant = set(relevant_ids)
    for rank, row_id in enumerate(retrieved_ids[:k], start=1):
        if row_id in relevant:
            return 1.0 / rank
    return 0.0


def score_query(
    label: EvalLabel,
    retrieved_ids: Sequence[str],
    corpus_ids: Collection[str],
) -> QueryMetrics:
    """Score one labeled query without removing absent relevant rows."""
    corpus = set(corpus_ids)
    missing = tuple(row_id for row_id in label.relevant_ids if row_id not in corpus)
    return QueryMetrics(
        query_class=label.query_class,
        recall_at_5=recall_at_k(retrieved_ids, label.relevant_ids, k=5),
        mrr_at_10=reciprocal_rank_at_k(retrieved_ids, label.relevant_ids, k=10),
        missing_relevant_ids=missing,
    )


def aggregate_metrics(results: Sequence[QueryMetrics]) -> dict[str, MetricSummary]:
    """Average query metrics by class and across the complete evaluation."""
    grouped: defaultdict[str, list[QueryMetrics]] = defaultdict(list)
    for result in results:
        grouped[result.query_class].append(result)
    grouped["overall"] = list(results)
    return {
        query_class: MetricSummary(
            count=len(values),
            recall_at_5=(
                sum(value.recall_at_5 for value in values) / len(values) if values else 0.0
            ),
            mrr_at_10=(sum(value.mrr_at_10 for value in values) / len(values) if values else 0.0),
        )
        for query_class, values in grouped.items()
    }


def load_labels(path: Path = LABELS_PATH) -> list[EvalLabel]:
    """Load labeled retrieval queries from JSON Lines."""
    labels: list[EvalLabel] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                query = payload["query"]
                query_class = payload["query_class"]
                relevant_ids = payload["relevant_ids"]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"invalid eval label on line {line_number}") from exc
            if (
                not isinstance(query, str)
                or not query.strip()
                or not isinstance(query_class, str)
                or not query_class.strip()
                or not isinstance(relevant_ids, list)
                or not relevant_ids
                or any(not isinstance(row_id, str) for row_id in relevant_ids)
            ):
                raise ValueError(f"invalid eval label on line {line_number}")
            labels.append(EvalLabel(query, query_class, tuple(relevant_ids)))
    return labels


def _set_schema(schema_name: str) -> dict[Any, str]:
    modules = (common, schema_module, ingest_api, search_module)
    previous = {module: module.PG_SCHEMA for module in modules}
    for module in modules:
        module.PG_SCHEMA = schema_name
    return previous


def _restore_schema(previous: dict[Any, str]) -> None:
    for module, schema_name in previous.items():
        module.PG_SCHEMA = schema_name


async def _ingest_fixture(path: Path) -> ingest_api.IngestJob:
    descriptor, temp_name = tempfile.mkstemp(
        prefix="memory-base-eval-",
        suffix=path.suffix,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        shutil.copyfile(path, temp_path)
        now = time.time()
        job = ingest_api.IngestJob(
            job_id=uuid.uuid4().hex,
            document_id=path.name.lower(),
            status="running",
            stage="converting",
            created_at=now,
            updated_at=now,
        )
        await ingest_api.run_document_job(
            job,
            temp_path,
            path.name,
            "force",
            f"eval-fixture:{path.name}",
        )
        if job.status != "succeeded":
            raise RuntimeError(f"fixture ingest did not succeed for {path.name}: {job.status}")
        return job
    finally:
        temp_path.unlink(missing_ok=True)


async def _evaluate_mode(
    labels: Sequence[EvalLabel],
    corpus_ids: Collection[str],
    *,
    include_atoms: bool,
) -> list[QueryMetrics]:
    results: list[QueryMetrics] = []
    for index, label in enumerate(labels, start=1):
        hits = await search_module.search(
            label.query,
            source="memory",
            rerank=True,
            include_atoms=include_atoms,
        )
        retrieved_ids = [row_id for hit in hits if isinstance((row_id := hit.meta.get("id")), str)]
        results.append(score_query(label, retrieved_ids, corpus_ids))
        print(
            f"[{'atoms-on' if include_atoms else 'atoms-off'}] "
            f"{index:02d}/{len(labels):02d} {label.query_class}"
        )
    return results


BASELINE_SEARCH_KWARGS: dict[str, Any] = {
    "source": "memory",
    "include_atoms": True,
    "rerank": True,
    "include_archived": False,
}


async def _evaluate_baseline(
    labels: Sequence[EvalLabel],
    corpus_ids: Collection[str],
) -> list[QueryMetrics]:
    results: list[QueryMetrics] = []
    for index, label in enumerate(labels, start=1):
        hits = await search_module.search(label.query, **BASELINE_SEARCH_KWARGS)
        retrieved_ids = [row_id for hit in hits if isinstance((row_id := hit.meta.get("id")), str)]
        results.append(score_query(label, retrieved_ids, corpus_ids))
        print(f"[baseline] {index:02d}/{len(labels):02d} {label.query_class}")
    return results


async def _evaluate_deep(
    labels: Sequence[EvalLabel],
    corpus_ids: Collection[str],
) -> tuple[list[QueryMetrics], list[list[str]]]:
    results: list[QueryMetrics] = []
    evidence_lists: list[list[str]] = []
    for index, label in enumerate(labels, start=1):
        result = await deep_search(label.query)
        evidence_ids = [entry.id for entry in result.evidence]
        evidence_lists.append(evidence_ids)
        results.append(score_query(label, evidence_ids, corpus_ids))
        print(
            f"[deep] {index:02d}/{len(labels):02d} {label.query_class} "
            f"hops={result.hops_used} stopped={result.stopped_reason}"
        )
    return results, evidence_lists


def _print_report(
    off_results: Sequence[QueryMetrics],
    on_results: Sequence[QueryMetrics],
) -> None:
    off = aggregate_metrics(off_results)
    on = aggregate_metrics(on_results)
    classes = sorted(name for name in off if name != "overall")
    for query_class in [*classes, "overall"]:
        off_summary = off[query_class]
        on_summary = on[query_class]
        print(
            f"{query_class} (n={off_summary.count}): "
            f"atoms-off Recall@5={off_summary.recall_at_5:.3f} "
            f"MRR@10={off_summary.mrr_at_10:.3f}; "
            f"atoms-on Recall@5={on_summary.recall_at_5:.3f} "
            f"MRR@10={on_summary.mrr_at_10:.3f}"
        )
    missing_off = sum(len(result.missing_relevant_ids) for result in off_results)
    missing_on = sum(len(result.missing_relevant_ids) for result in on_results)
    print(f"Missing relevant IDs: atoms-off={missing_off} atoms-on={missing_on}")
    verdict = "PASS" if gate_passes(off, on) else "FAIL"
    print(
        f"Gate verdict: {verdict} "
        "(per-class atoms-on Recall@5 >= atoms-off - 0.05; "
        "overall atoms-on Recall@5 >= atoms-off)"
    )


def _print_deep_report(
    baseline_results: Sequence[QueryMetrics],
    deep_results: Sequence[QueryMetrics],
    labels: Sequence[EvalLabel],
    evidence_lists: Sequence[Sequence[str]],
) -> None:
    baseline_agg = aggregate_metrics(baseline_results)
    deep_agg = aggregate_metrics(deep_results)
    baseline_summary = baseline_agg.get("overall", MetricSummary(0, 0.0, 0.0))
    deep_summary = deep_agg.get("overall", MetricSummary(0, 0.0, 0.0))
    completion = path_completion(labels, evidence_lists)
    print(
        f"deep multi_hop (n={deep_summary.count}): "
        f"baseline Recall@5={baseline_summary.recall_at_5:.3f} "
        f"MRR@10={baseline_summary.mrr_at_10:.3f}; "
        f"deep Recall@5={deep_summary.recall_at_5:.3f} "
        f"MRR@10={deep_summary.mrr_at_10:.3f}; "
        f"path_completion={completion:.3f}"
    )
    verdict = "PASS" if deep_gate_passes(baseline_summary, deep_summary, completion) else "FAIL"
    print(
        f"Deep gate verdict: {verdict} "
        "(deep Recall@5 >= baseline Recall@5 on multi_hop; "
        "path_completion >= 0.4)"
    )


def gate_passes(
    atoms_off: dict[str, MetricSummary],
    atoms_on: dict[str, MetricSummary],
) -> bool:
    """Apply the per-class tolerance and zero-drop overall Recall@5 gate."""
    classes = sorted(name for name in atoms_off if name != "overall")
    per_class_pass = all(
        atoms_on[name].recall_at_5 >= atoms_off[name].recall_at_5 - 0.05 for name in classes
    )
    overall_pass = atoms_on["overall"].recall_at_5 >= atoms_off["overall"].recall_at_5
    return per_class_pass and overall_pass


def _document_of(row_id: str) -> str:
    parts = row_id.split(":")
    return parts[1] if len(parts) >= 2 else row_id


def path_completion(
    labels: Sequence[EvalLabel],
    evidence_lists: Sequence[Sequence[str]],
) -> float:
    """Fraction of multi_hop queries whose evidence covers every labeled document."""
    if not labels:
        return 0.0
    completed = 0
    for label, evidence_ids in zip(labels, evidence_lists, strict=True):
        required_docs = {_document_of(rid) for rid in label.relevant_ids}
        covered_docs = {_document_of(eid) for eid in evidence_ids} & required_docs
        if required_docs and covered_docs >= required_docs:
            completed += 1
    return completed / len(labels)


def deep_gate_passes(
    baseline_summary: MetricSummary,
    deep_summary: MetricSummary,
    completion: float,
) -> bool:
    """Deep Recall@5 >= baseline Recall@5 on multi_hop AND path completion >= 0.4."""
    return deep_summary.recall_at_5 >= baseline_summary.recall_at_5 and completion >= 0.4


async def run_evaluation() -> None:
    """Ingest checked-in fixtures into a scratch schema and print the A/B report."""
    print(f"Models: embedding={EMB_MODEL} rerank={RERANK_MODEL}")
    missing_env = [name for name in REQUIRED_SERVICE_ENV if not os.getenv(name)]
    if missing_env:
        raise RuntimeError("missing required service configuration: " + ", ".join(missing_env))

    labels = load_labels()
    single_hop_labels = [label for label in labels if label.query_class != MULTI_HOP_CLASS]
    multi_hop_labels = [label for label in labels if label.query_class == MULTI_HOP_CLASS]
    schema_name = f"memory_eval_{os.getpid()}_{uuid.uuid4().hex[:12]}"
    previous_schema = _set_schema(schema_name)
    previous_atoms_generate = os.environ.get("ATOMS_GENERATE")
    os.environ["ATOMS_GENERATE"] = "true"
    schema_created = False
    try:
        setup_conn = await asyncpg.connect(DB_URL, timeout=5)
        try:
            await setup_conn.execute(f'CREATE SCHEMA "{schema_name}"')
            schema_created = True
            await schema_module.ensure_schema(setup_conn)
        finally:
            await setup_conn.close()
        for path in sorted(FIXTURE_DIR.iterdir()):
            if path.is_file():
                await _ingest_fixture(path)
        report_conn = await asyncpg.connect(DB_URL, timeout=5)
        try:
            rows = await report_conn.fetch(
                f'SELECT id, chunk_kind FROM "{schema_name}".memory_chunks'
            )
        finally:
            await report_conn.close()
        corpus_ids = {row["id"] for row in rows}
        parent_count = sum(row["chunk_kind"] != "atom" for row in rows)
        print(
            f"Corpus: fixtures={len(list(FIXTURE_DIR.iterdir()))} "
            f"parents={parent_count} rows={len(rows)} schema={schema_name}"
        )
        off_results = await _evaluate_mode(single_hop_labels, corpus_ids, include_atoms=False)
        on_results = await _evaluate_mode(single_hop_labels, corpus_ids, include_atoms=True)
        _print_report(off_results, on_results)
        if multi_hop_labels:
            print()
            baseline_results = await _evaluate_baseline(multi_hop_labels, corpus_ids)
            deep_results, evidence_lists = await _evaluate_deep(multi_hop_labels, corpus_ids)
            _print_deep_report(baseline_results, deep_results, multi_hop_labels, evidence_lists)
    finally:
        try:
            if schema_created:
                cleanup_conn = await asyncpg.connect(DB_URL, timeout=5)
                try:
                    await cleanup_conn.execute(f'DROP SCHEMA "{schema_name}" CASCADE')
                finally:
                    await cleanup_conn.close()
        finally:
            _restore_schema(previous_schema)
            if previous_atoms_generate is None:
                os.environ.pop("ATOMS_GENERATE", None)
            else:
                os.environ["ATOMS_GENERATE"] = previous_atoms_generate


def main() -> None:
    """Run the report without making environment availability a process gate."""
    try:
        asyncio.run(run_evaluation())
    except Exception as exc:
        print(f"Evaluation unavailable: {exc}")
        print("Gate verdict: NOT RUN (DB and configured vLLM services are required)")


if __name__ == "__main__":
    main()
