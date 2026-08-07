"""Reproducible document retrieval evaluation."""

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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from memory_base.core import schema as schema_module
from memory_base.core.config import db_url, emb_model, rerank_model
from memory_base.core.logger import setup_logging
from memory_base.retrieval import search as search_module
from memory_base.serve import ingest_api

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "eval_docs"
LABELS_PATH = REPO_ROOT / "tests" / "fixtures" / "retrieval_eval.jsonl"
RELEVANT_ID_RE = re.compile(r"^doc:[a-z0-9][a-z0-9._-]{0,120}:(?:[0-9]+|card:[0-9]+)$")
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


@contextmanager
def _scratch_schema_scope(schema_name: str):
    """Point core.schema.ensure_schema[_once] at the scratch schema for this run.

    search() and ingest_api.run_document_job() take an explicit `schema` argument
    instead (their PG_SCHEMA reads are fully contained in functions the eval calls
    directly). ensure_schema_once has no such argument: it is also called,
    unparameterized, from serve/* modules reachable transitively while ingesting a
    fixture (e.g. namespaces.namespace_exists, invoked from run_document_job) that
    are out of scope for the eval to parameterize — doing so would grow serving-layer
    signatures for the eval's sake. Rebinding the module global for the duration of
    the run is the smallest change that still keeps those calls scoped to the scratch
    schema; core/schema.py's test suite documents that the DDL target and the
    once-guard's cache key move together when PG_SCHEMA is rebound this way.
    """
    previous = schema_module.PG_SCHEMA
    schema_module.PG_SCHEMA = schema_name
    try:
        yield
    finally:
        schema_module.PG_SCHEMA = previous


async def _ingest_fixture(path: Path, schema: str) -> ingest_api.IngestJob:
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
            schema=schema,
        )
        if job.status != "succeeded":
            raise RuntimeError(f"fixture ingest did not succeed for {path.name}: {job.status}")
        return job
    finally:
        temp_path.unlink(missing_ok=True)


async def _evaluate_mode(
    labels: Sequence[EvalLabel],
    corpus_ids: Collection[str],
    schema: str,
) -> list[QueryMetrics]:
    results: list[QueryMetrics] = []
    for index, label in enumerate(labels, start=1):
        hits = await search_module.search(
            label.query,
            source="memory",
            rerank=True,
            schema=schema,
        )
        retrieved_ids = [row_id for hit in hits if isinstance((row_id := hit.meta.get("id")), str)]
        results.append(score_query(label, retrieved_ids, corpus_ids))
        print(f"{index:02d}/{len(labels):02d} {label.query_class}")
    return results


def _print_report(results: Sequence[QueryMetrics]) -> None:
    summary = aggregate_metrics(results)
    classes = sorted(name for name in summary if name != "overall")
    for query_class in [*classes, "overall"]:
        class_summary = summary[query_class]
        print(
            f"{query_class} (n={class_summary.count}): "
            f"Recall@5={class_summary.recall_at_5:.3f} "
            f"MRR@10={class_summary.mrr_at_10:.3f}"
        )
    missing = sum(len(result.missing_relevant_ids) for result in results)
    print(f"Missing relevant IDs: {missing}")


async def run_evaluation() -> None:
    """Ingest checked-in fixtures into a scratch schema and print the report."""
    print(f"Models: embedding={emb_model()} rerank={rerank_model()}")
    missing_env = [name for name in REQUIRED_SERVICE_ENV if not os.getenv(name)]
    if missing_env:
        raise RuntimeError("missing required service configuration: " + ", ".join(missing_env))

    labels = load_labels()
    schema_name = f"memory_eval_{os.getpid()}_{uuid.uuid4().hex[:12]}"
    schema_created = False
    with _scratch_schema_scope(schema_name):
        try:
            setup_conn = await asyncpg.connect(db_url(), timeout=5)
            try:
                await setup_conn.execute(f'CREATE SCHEMA "{schema_name}"')
                schema_created = True
                await schema_module.ensure_schema(setup_conn)
            finally:
                await setup_conn.close()
            for path in sorted(FIXTURE_DIR.iterdir()):
                if path.is_file():
                    await _ingest_fixture(path, schema_name)
            report_conn = await asyncpg.connect(db_url(), timeout=5)
            try:
                rows = await report_conn.fetch(f'SELECT id FROM "{schema_name}".memory_chunks')
            finally:
                await report_conn.close()
            corpus_ids = {row["id"] for row in rows}
            print(
                f"Corpus: fixtures={len(list(FIXTURE_DIR.iterdir()))} "
                f"rows={len(rows)} schema={schema_name}"
            )
            results = await _evaluate_mode(labels, corpus_ids, schema_name)
            _print_report(results)
        finally:
            if schema_created:
                cleanup_conn = await asyncpg.connect(db_url(), timeout=5)
                try:
                    await cleanup_conn.execute(f'DROP SCHEMA "{schema_name}" CASCADE')
                finally:
                    await cleanup_conn.close()


def main() -> None:
    """Run the report without making environment availability a process gate."""
    setup_logging()
    try:
        asyncio.run(run_evaluation())
    except Exception as exc:
        print(f"Evaluation unavailable: {exc}")
        print("Gate verdict: NOT RUN (DB and configured vLLM services are required)")


if __name__ == "__main__":
    main()
