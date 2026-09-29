"""LongMemEval on the agent-distilled write path: retrieval, judge audit, and scoring.

Notes come from scripts/longmemeval/extract.py, an emulated agent outside the server.
`retrieve` loads each selected question's notes into its own namespace of a throwaway
Postgres built from db.Dockerfile through the production save_note path (gate pinned
open, its verdict already recorded at extraction), then runs production search.
scripts/longmemeval/answer.py answers and judges the packets; `audit-sample` draws the
seeded judgments a person grades by hand, and `score` reports QA accuracy, session-level
retrieval metrics, and the judge agreement rate.

Runs: `baseline` (gate on), `gate-off` (`--gate off`: gate-refused notes loaded too), and
`dated` (`--variant dated`: temporal-reasoning questions, date-prefixed embeddings). `--read
prefetch` reads each question the way the prefetch hook does (top 5, score floor 0.6) instead
of search_memory's defaults (top 10, floor 0.25) and prefixes the run name with `prefetch`.

CLI:
  uv run python -m memory_base.eval.longmemeval retrieve --dataset PATH [--gate off]
  uv run python -m memory_base.eval.longmemeval audit-sample --dataset PATH [--gate off]
  uv run python -m memory_base.eval.longmemeval score --dataset PATH
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import secrets
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memory_base.eval import longmemeval_prompts as prompts

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "longmemeval"
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "benchmarks" / "longmemeval-manifest.json"
SUBSET_SEED = 0
SUBSET_SIZE = 100
NOTE_TAGS = ("longmemeval",)
NAMESPACE_PREFIX = "lme-"
DB_LABEL = "memory-base-longmemeval"
RETRIEVE_CONCURRENCY = 4
SAVE_ATTEMPTS = 3
SAVE_BACKOFF_SECONDS = 5.0
METRIC_KS = (5, 10)
VARIANTS = ("baseline", "dated")
GATES = ("on", "off")
RUNS = ("baseline", "gate-off", "dated", "prefetch", "prefetch-gate-off")
# How many hits a reader keeps and the score floor it asks for: search_memory's defaults
# and the values the prefetch hook sends.
READ_SETTINGS = {
    "search": {"top_k": 10, "min_score": None},
    "prefetch": {"top_k": 5, "min_score": 0.6},
}
DATED_VARIANT_TYPES = ("temporal-reasoning",)
STAGES = ("answer", "judge")
JUDGE_AUDIT_SIZE = 20
JUDGE_AUDIT_SEED = 0
NOTES_FILE = "notes.jsonl"
SESSIONS_FILE = "sessions.jsonl"
SESSION_TOTALS = (
    "in_tok",
    "out_tok",
    "gate_in_tok",
    "gate_out_tok",
    "gate_calls",
    "gate_retries",
    "extract_retries",
    "extract_seconds",
    "seconds",
)
REFUSAL_CAUSES = ("validation:", "credential:")
DATASET_DATE_RE = re.compile(r"^(\d{4})/(\d{2})/(\d{2}) \(\w{3}\) (\d{2}):(\d{2})$")

# The session date (ISO) of the note being saved; read by the dated-embedding variant only.
NOTE_DATE: ContextVar[str | None] = ContextVar("lme_note_date", default=None)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def type_quotas(counts: dict[str, int], size: int) -> dict[str, int]:
    """Largest-remainder allocation of `size` across strata; ties go to the larger stratum."""
    total = sum(counts.values())
    quotas = {name: count * size // total for name, count in counts.items()}
    order = sorted(counts, key=lambda name: (-(counts[name] * size % total), -counts[name], name))
    for name in order[: size - sum(quotas.values())]:
        quotas[name] += 1
    return quotas


def select_subset(
    questions: Sequence[dict[str, Any]], size: int = SUBSET_SIZE, seed: int = SUBSET_SEED
) -> list[dict[str, Any]]:
    """Seeded sample proportional per question_type, returned in dataset order."""
    by_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_type[question["question_type"]].append(question)
    quotas = type_quotas({name: len(group) for name, group in by_type.items()}, size)
    rng = random.Random(seed)
    chosen: set[str] = set()
    for name in sorted(by_type):
        chosen.update(q["question_id"] for q in rng.sample(by_type[name], quotas[name]))
    return [q for q in questions if q["question_id"] in chosen]


def filter_questions(
    subset: Sequence[dict[str, Any]], question_ids: Sequence[str] | None
) -> list[dict[str, Any]]:
    """Restrict the subset to the given ids; an id outside the subset is an error."""
    if not question_ids:
        return list(subset)
    known = {q["question_id"] for q in subset}
    unknown = [qid for qid in question_ids if qid not in known]
    if unknown:
        raise ValueError(f"not in the selected subset: {', '.join(unknown)}")
    wanted = set(question_ids)
    return [q for q in subset if q["question_id"] in wanted]


def subset_manifest(subset: Sequence[dict[str, Any]], dataset_sha256: str) -> dict[str, Any]:
    return {
        "dataset_sha256": dataset_sha256,
        "seed": SUBSET_SEED,
        "size": len(subset),
        "question_ids": [q["question_id"] for q in subset],
        "per_type": dict(sorted(Counter(q["question_type"] for q in subset).items())),
        "abstention": sum("_abs" in q["question_id"] for q in subset),
    }


def iso_datetime(date: str) -> str:
    """Dataset form `2023/05/20 (Sat) 02:21` -> ISO 8601 `2023-05-20T02:21:00`."""
    match = DATASET_DATE_RE.fullmatch(date)
    if match is None:
        raise ValueError(f"unexpected dataset date: {date!r}")
    year, month, day, hour, minute = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:00"


def dataset_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y/%m/%d (%a) %H:%M")


def session_units(question: dict[str, Any]) -> list[tuple[str, str]]:
    """The question's distinct (session_id, date) extraction units in date order."""
    units = set(zip(question["haystack_session_ids"], question["haystack_dates"], strict=True))
    return sorted(units, key=lambda unit: (iso_datetime(unit[1]), unit[0]))


def session_turns(questions: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict]]:
    turns: dict[tuple[str, str], list[dict]] = {}
    for question in questions:
        for sid, date, session in zip(
            question["haystack_session_ids"],
            question["haystack_dates"],
            question["haystack_sessions"],
            strict=True,
        ):
            turns.setdefault((sid, date), session)
    return turns


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines, discarding a partial last line left by an interrupted append."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    if not text.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def _jsonl_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False) + "\n"


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append rows and fsync, so a completed unit survives a crash."""
    with Path(path).open("a", encoding="utf-8") as sink:
        sink.writelines(_jsonl_line(row) for row in rows)
        sink.flush()
        os.fsync(sink.fileno())


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as sink:
        sink.writelines(_jsonl_line(row) for row in rows)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temp, path)


def read_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update_manifest(path: Path, section: str, data: Any) -> None:
    """Replace one top-level section of the manifest, keeping the others."""
    path = Path(path)
    manifest = read_manifest(path)
    manifest[section] = data
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def code_revision() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    # The run writes its own manifest, which must not mark the code it ran as modified.
    manifest = f":(exclude){DEFAULT_MANIFEST.relative_to(REPO_ROOT)}"
    dirty = git("status", "--porcelain", "--", ".", manifest)
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty)}


@dataclass
class LoadStats:
    submitted: int = 0
    stored: int = 0
    duplicates: int = 0
    similar_acks: int = 0
    credential_refused: int = 0
    invalid: int = 0
    gate_refused: int = 0


SaveNote = Callable[..., Awaitable[dict[str, Any]]]


async def _save_with_retry(save: SaveNote, content: str, **kwargs: Any) -> dict[str, Any]:
    """Retry transient embedder/database failures; validation errors propagate at once."""
    import asyncpg
    import openai

    for attempt in range(1, SAVE_ATTEMPTS + 1):
        try:
            return await save(content, **kwargs)
        except (TimeoutError, OSError, openai.APIError, asyncpg.PostgresConnectionError):
            if attempt == SAVE_ATTEMPTS:
                raise
            await asyncio.sleep(SAVE_BACKOFF_SECONDS * attempt)
    raise AssertionError("unreachable")


async def load_question_notes(
    namespace: str,
    units: Sequence[tuple[str, str]],
    notes_by_unit: dict[tuple[str, str], list[dict[str, Any]]],
    save: SaveNote | None = None,
    gate: str = "on",
) -> tuple[LoadStats, dict[str, set[tuple[str, str]]]]:
    """Save the units' notes in order and map note ids to their units.

    With the gate on only gate-stored notes load; with it off gate-refused notes load too,
    while notes refused by validation or the credential scan never do (save_note would
    refuse them). A note id is a content hash, so identical notes from two sessions
    collide on one row; the returned provenance maps that row to both.
    """
    from memory_base.serve import notes as notes_module
    from memory_base.serve.namespaces import NamespaceError

    save = save or notes_module.save_note
    stats = LoadStats()
    provenance: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    for unit in units:
        occurred_at = iso_datetime(unit[1])
        for note in notes_by_unit.get(unit, []):
            if not loadable(note, gate):
                continue
            stats.submitted += 1
            stats.gate_refused += note["gate"] == "refused"
            token = NOTE_DATE.set(occurred_at[:10])
            try:
                result = await _save_with_retry(
                    save,
                    note["content"],
                    tags=list(NOTE_TAGS),
                    kind=note["kind"],
                    namespace=namespace,
                    occurred_at=occurred_at,
                    allow_similar=True,
                )
            except notes_module.CredentialNoteError:
                stats.credential_refused += 1
                continue
            except NamespaceError:
                raise
            except ValueError:
                stats.invalid += 1
                continue
            finally:
                NOTE_DATE.reset(token)
            provenance[result["id"]].add(unit)
            if result["stored"]:
                stats.stored += 1
            else:
                stats.duplicates += 1
            if result["similar"]:
                stats.similar_acks += 1
    return stats, dict(provenance)


def loadable(note: dict[str, Any], gate: str) -> bool:
    if note["gate"] == "stored":
        return True
    return gate == "off" and not note["gate_reason"].startswith(REFUSAL_CAUSES)


def dated_embed_text(embed: Callable[[Any, str], Awaitable[str]]):
    """Wrap notes.embed_text so a note embeds as `{date}: {content}`; stored text is unchanged."""

    async def embed_dated(embedder: Any, text: str) -> str:
        date = NOTE_DATE.get()
        return await embed(embedder, f"{date}: {text}" if date else text)

    return embed_dated


def session_ranking(hits: Sequence[dict[str, Any]]) -> list[str]:
    """Distinct benchmark sessions in hit order; a collided note contributes all of its."""
    ranking: list[str] = []
    for hit in hits:
        for sid, _date in hit["sessions"]:
            if sid not in ranking:
                ranking.append(sid)
    return ranking


def recall_all_at_k(ranking: Sequence[str], correct: set[str], *, k: int) -> float:
    if not correct:
        raise ValueError("a scored question needs at least one answer session")
    top = set(ranking[:k])
    return float(all(sid in top for sid in correct))


def _dcg(relevances: Sequence[int]) -> float:
    # Upstream eval_utils.dcg: rank 1 undiscounted, rank r >= 2 divided by log2(r).
    if not relevances:
        return 0.0
    return relevances[0] + sum(rel / math.log2(i + 1) for i, rel in enumerate(relevances) if i)


def ndcg_any_at_k(ranking: Sequence[str], correct: set[str], *, k: int) -> float:
    if not correct:
        raise ValueError("a scored question needs at least one answer session")
    actual = _dcg([1 if sid in correct else 0 for sid in ranking[:k]])
    return actual / _dcg([1] * min(len(correct), k))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def retrieval_metrics(
    packets: Sequence[dict[str, Any]], questions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Official session-level recall_all@k / ndcg_any@k; abstention questions excluded."""
    per_question: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
    zero_hit = zero_hit_scored = 0
    for packet in packets:
        qid = packet["question_id"]
        scored = "_abs" not in qid
        if not packet["hits"]:
            zero_hit += 1
            zero_hit_scored += scored
        if not scored:
            continue
        question = questions[qid]
        correct = set(question["answer_session_ids"])
        ranking = session_ranking(packet["hits"])
        metrics = {}
        for k in METRIC_KS:
            metrics[f"recall_all@{k}"] = recall_all_at_k(ranking, correct, k=k)
            metrics[f"ndcg_any@{k}"] = ndcg_any_at_k(ranking, correct, k=k)
        per_question[question["question_type"]].append(metrics)
        per_question["overall"].append(metrics)
    report: dict[str, Any] = {}
    for group, rows in sorted(per_question.items()):
        report[group] = {"count": len(rows)} | {
            name: _mean([row[name] for row in rows]) for name in rows[0]
        }
    report["zero_hit_packets"] = zero_hit
    report["zero_hit_packets_scored"] = zero_hit_scored
    return report


def answer_prompt(packet: dict[str, Any]) -> str:
    """Upstream facts prompt; hits are date-sorted as upstream sorts retrieved chunks."""
    hits = sorted(packet["hits"], key=lambda hit: iso_datetime(hit["date"]))
    history = "".join(
        "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(
            index, hit["date"], hit["text"]
        )
        for index, hit in enumerate(hits, start=1)
    )
    return prompts.ANSWER_TEMPLATE.format(history, packet["question_date"], packet["question"])


def judge_prompt(question: dict[str, Any], answer_text: str) -> str:
    return prompts.get_anscheck_prompt(
        question["question_type"],
        question["question"],
        question["answer"],
        answer_text,
        abstention="_abs" in question["question_id"],
    )


judge_label = prompts.judge_label


def upstream_manifest() -> dict[str, Any]:
    """The upstream commit and sha256 of every template copied from it."""
    judges = {
        "qa": prompts.QA_TEMPLATE,
        "temporal-reasoning": prompts.TEMPORAL_TEMPLATE,
        "knowledge-update": prompts.KNOWLEDGE_UPDATE_TEMPLATE,
        "single-session-preference": prompts.PREFERENCE_TEMPLATE,
        "abstention": prompts.ABSTENTION_TEMPLATE,
    }
    return {
        "commit": prompts.UPSTREAM_COMMIT,
        "answer_template_sha256": prompt_sha(prompts.ANSWER_TEMPLATE),
        "judge_templates_sha256": {name: prompt_sha(text) for name, text in judges.items()},
    }


def _accuracy(labels: Sequence[bool]) -> dict[str, Any]:
    correct = sum(labels)
    accuracy = correct / len(labels) if labels else None
    return {"correct": correct, "judged": len(labels), "accuracy": accuracy}


def qa_accuracy(
    question_ids: Sequence[str],
    questions: dict[str, dict[str, Any]],
    answers: Sequence[dict[str, Any]],
    judgments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Accuracy overall, per question_type, and over abstention questions.

    A question's answer is its latest answer row; a judgment counts only when it graded
    that answer (its prompt sha matches), and the latest such judgment wins.
    """
    labels = {
        qid: judge_label(row["text"]) for qid, row in _judged(questions, answers, judgments).items()
    }
    groups: defaultdict[str, list[bool]] = defaultdict(list, overall=[])
    for qid in question_ids:
        if qid not in labels:
            continue
        groups["overall"].append(labels[qid])
        groups[questions[qid]["question_type"]].append(labels[qid])
        if "_abs" in qid:
            groups["abstention"].append(labels[qid])
    report: dict[str, Any] = {name: _accuracy(values) for name, values in sorted(groups.items())}
    report["unjudged"] = [qid for qid in question_ids if qid not in labels]
    return report


def _judged(
    questions: dict[str, dict[str, Any]],
    answers: Sequence[dict[str, Any]],
    judgments: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Each question's latest judgment of its latest answer."""
    answer_by_id = {row["question_id"]: row for row in answers}
    expected = {
        qid: judge_prompt(questions[qid], row["text"])
        for qid, row in answer_by_id.items()
        if qid in questions
    }
    return current_rows(judgments, expected)


def judge_audit_sample(
    question_ids: Sequence[str],
    questions: dict[str, dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    judgments: Sequence[dict[str, Any]],
    *,
    size: int = JUDGE_AUDIT_SIZE,
    seed: int = JUDGE_AUDIT_SEED,
) -> list[dict[str, Any]]:
    """A seeded sample of judgments for a person to grade by setting `human_label`."""
    judged = _judged(questions, list(answers.values()), judgments)
    candidates = [qid for qid in question_ids if qid in judged]
    picked = set(random.Random(seed).sample(candidates, min(size, len(candidates))))
    return [
        {
            "question_id": qid,
            "question_type": questions[qid]["question_type"],
            "question": questions[qid]["question"],
            "reference": questions[qid]["answer"],
            "response": answers[qid]["text"],
            "judge_reply": judged[qid]["text"],
            "judge_label": judge_label(judged[qid]["text"]),
            "human_label": None,
        }
        for qid in candidates
        if qid in picked
    ]


def judge_agreement(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Agreement between the judge and the hand labels; unlabeled rows are left out."""
    labeled = [row for row in rows if isinstance(row.get("human_label"), bool)]
    agreed = sum(row["human_label"] == row["judge_label"] for row in labeled)
    return {
        "sample": len(rows),
        "labeled": len(labeled),
        "agreed": agreed,
        "agreement_rate": agreed / len(labeled) if labeled else None,
    }


def gate_rates(
    question_ids: Sequence[str],
    questions: dict[str, dict[str, Any]],
    notes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Refused-save rate over the selected questions' notes, overall and per question_type."""
    total: Counter[tuple[str, str]] = Counter()
    refused: Counter[tuple[str, str]] = Counter()
    for note in notes:
        unit = (note["session_id"], note["date"])
        total[unit] += 1
        refused[unit] += note["gate"] == "refused"

    def rate(units: Iterable[tuple[str, str]]) -> dict[str, Any]:
        units = list(units)
        count = sum(total[u] for u in units)
        bad = sum(refused[u] for u in units)
        return {"notes": count, "refused": bad, "refused_rate": bad / count if count else None}

    by_type: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    every: set[tuple[str, str]] = set()
    for qid in question_ids:
        units = session_units(questions[qid])
        by_type[questions[qid]["question_type"]].extend(units)
        every.update(units)
    return {"overall": rate(every)} | {name: rate(units) for name, units in sorted(by_type.items())}


def throwaway_db_url(password: str, port: str) -> str:
    if not port.isdigit():
        raise ValueError(f"unexpected docker port: {port!r}")
    return f"postgresql://memory:{password}@127.0.0.1:{port}/memory_base"


def _docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@contextmanager
def throwaway_postgres():
    """A fresh Postgres from db.Dockerfile on tmpfs; DB_URL points at it until removal."""
    image = _docker("build", "-q", "-f", str(REPO_ROOT / "db.Dockerfile"), str(REPO_ROOT))
    password = secrets.token_hex(16)
    container = _docker(
        "run", "-d", "--rm", "--label", DB_LABEL,
        "--tmpfs", "/var/lib/postgresql/data",
        "-e", "POSTGRES_USER=memory", "-e", f"POSTGRES_PASSWORD={password}",
        "-e", "POSTGRES_DB=memory_base",
        "-p", "127.0.0.1::5432",
        image, "postgres", "-c", "shared_preload_libraries=pg_textsearch",
    )  # fmt: skip
    try:
        port = _docker("port", container, "5432/tcp").splitlines()[0].rsplit(":", 1)[1]
        url = throwaway_db_url(password, port)
        os.environ["DB_URL"] = url
        os.environ["TABLES_QUERY_PASSWORD"] = secrets.token_hex(16)
        yield {"image": image, "url": url}
    finally:
        _docker("rm", "-f", container)


async def _prepare_schema(url: str, deadline_seconds: float = 90) -> list[str]:
    """Wait for the fresh server, create the schema, and return its extensions."""
    import asyncpg

    from memory_base.core.schema import ensure_schema

    deadline = time.monotonic() + deadline_seconds
    while True:
        try:
            conn = await asyncpg.connect(url, timeout=3)
            break
        except (OSError, asyncpg.CannotConnectNowError, asyncpg.ConnectionDoesNotExistError):
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.5)
    try:
        await ensure_schema(conn)
        rows = await conn.fetch("SELECT extname, extversion FROM pg_extension ORDER BY extname")
    finally:
        await conn.close()
    return [f"{row['extname']} {row['extversion']}" for row in rows]


def run_name(variant: str, gate: str, read: str = "search") -> str:
    """The run a variant, gate and read setting select; gate-off runs only the baseline variant."""
    if gate == "on":
        run = variant
    elif variant != "baseline":
        raise ValueError("the gate-off run covers the baseline variant only")
    else:
        run = "gate-off"
    if read == "search":
        return run
    return read if run == "baseline" else f"{read}-{run}"


def read_setting(run: str) -> dict[str, Any]:
    return READ_SETTINGS["prefetch" if run.startswith("prefetch") else "search"]


def run_suffix(run: str) -> str:
    return "" if run == "baseline" else f"-{run}"


def packets_path(data_dir: Path, run: str) -> Path:
    return data_dir / f"packets{run_suffix(run)}.jsonl"


def stage_output_path(data_dir: Path, stage: str, run: str) -> Path:
    name = "answers" if stage == "answer" else "judgments"
    return data_dir / f"{name}{run_suffix(run)}.jsonl"


def judge_audit_path(data_dir: Path, run: str) -> Path:
    return data_dir / f"judge-audit{run_suffix(run)}.jsonl"


def _hit_record(hit: Any, provenance: dict[str, set[tuple[str, str]]]) -> dict[str, Any]:
    note_id = hit.meta["id"]
    if note_id not in provenance:
        raise RuntimeError(f"hit {note_id} was not loaded for this question")
    return {
        "id": note_id,
        "date": dataset_date(hit.ts),
        "score": hit.score,
        "text": hit.text,
        "sessions": [list(unit) for unit in sorted(provenance[note_id])],
    }


async def _gate_pinned_open(content: str, kind: str):
    from memory_base.serve.notes import ContentVerdict

    return ContentVerdict(accepted=True, reason="verdict recorded at extraction")


async def retrieve_question(
    question: dict[str, Any],
    notes_by_unit: dict[tuple[str, str], list[dict[str, Any]]],
    run: str,
) -> dict[str, Any]:
    from memory_base.eval.retrieval import _search_with_retry
    from memory_base.serve import namespaces

    namespace = NAMESPACE_PREFIX + question["question_id"]
    await namespaces.create_namespace(namespace)
    started = time.monotonic()
    stats, provenance = await load_question_notes(
        namespace,
        session_units(question),
        notes_by_unit,
        gate="off" if run.endswith("gate-off") else "on",
    )
    loaded = time.monotonic()
    setting = read_setting(run)
    hits = await _search_with_retry(
        question["question"],
        source="memory",
        namespaces=[namespace],
        min_score=setting["min_score"],
    )
    hits = hits[: setting["top_k"]]
    return {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "question": question["question"],
        "question_date": question["question_date"],
        "run": run,
        "load": asdict(stats),
        "seconds": {"load": loaded - started, "search": time.monotonic() - loaded},
        "hits": [_hit_record(hit, provenance) for hit in hits],
    }


def _notes_by_unit(data_dir: Path) -> tuple[dict, set[tuple[str, str]]]:
    completed = {(row["session_id"], row["date"]) for row in read_jsonl(data_dir / SESSIONS_FILE)}
    by_unit: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for note in read_jsonl(data_dir / NOTES_FILE):
        unit = (note["session_id"], note["date"])
        if unit in completed:
            by_unit[unit].append(note)
    return dict(by_unit), completed


def _retrieval_constants() -> dict[str, Any]:
    from memory_base.retrieval import search
    from memory_base.serve import notes

    return {
        "NOTE_SIMILAR_THRESHOLD": notes.NOTE_SIMILAR_THRESHOLD,
        "MIN_SCORE": search.MIN_SCORE,
        "RERANK_TOP": search.RERANK_TOP,
        "FUSED_TOP": search.FUSED_TOP,
        "CANDIDATES_PER_SIGNAL": search.CANDIDATES_PER_SIGNAL,
        "PER_FILE_CAP": search.PER_FILE_CAP,
        "TIME_DECAY_HALF_LIFE_DAYS": search.TIME_DECAY_HALF_LIFE_DAYS,
    }


async def _retrieve_all(
    pending: Sequence[dict[str, Any]],
    notes_by_unit: dict,
    run: str,
    out_path: Path,
    db_url: str,
) -> list[str]:
    from memory_base.core import db

    extensions = await _prepare_schema(db_url)
    semaphore = asyncio.Semaphore(RETRIEVE_CONCURRENCY)
    done = 0

    async def one(question: dict[str, Any]) -> None:
        nonlocal done
        async with semaphore:
            packet = await retrieve_question(question, notes_by_unit, run)
            append_jsonl(out_path, [packet])
            done += 1
            print(
                f"{done:03d}/{len(pending):03d} {packet['question_id']} "
                f"notes={packet['load']['submitted']} hits={len(packet['hits'])}",
                flush=True,
            )

    try:
        await asyncio.gather(*(one(q) for q in pending))
    finally:
        await db.close_pool()
    return extensions


def run_retrieve(args: argparse.Namespace) -> None:
    from memory_base.core.config import emb_model, rerank_model
    from memory_base.serve import notes

    code = code_revision()
    run = run_name(args.variant, args.gate, args.read)
    dataset = load_dataset(args.dataset)
    dataset_sha = sha256_file(args.dataset)
    subset = select_subset(dataset)
    selected = filter_questions(subset, args.questions)
    if run == "dated":
        selected = [q for q in selected if q["question_type"] in DATED_VARIANT_TYPES]
    notes_by_unit, completed = _notes_by_unit(args.data_dir)
    missing = {u for q in selected for u in session_units(q)} - completed
    if missing:
        raise SystemExit(f"{len(missing)} extraction units are not extracted yet; run extract")
    out_path = packets_path(args.data_dir, run)
    done_ids = {row["question_id"] for row in read_jsonl(out_path)}
    pending = [q for q in selected if q["question_id"] not in done_ids]
    print(f"questions: {len(selected)} selected, {len(pending)} pending ({run})")

    notes.judge_note_content = _gate_pinned_open
    if run == "dated":
        notes.embed_text = dated_embed_text(notes.embed_text)
    image = extensions = None
    if pending:
        with throwaway_postgres() as database:
            image = database["image"]
            extensions = asyncio.run(
                _retrieve_all(pending, notes_by_unit, run, out_path, database["url"])
            )

    packets = read_jsonl(out_path)
    questions = {q["question_id"]: q for q in dataset}
    load = Counter()
    for packet in packets:
        load.update(packet["load"])
    section = f"retrieve{run_suffix(run)}"
    previous = read_manifest(args.manifest).get(section, {})
    update_manifest(args.manifest, "subset", subset_manifest(subset, dataset_sha))
    update_manifest(
        args.manifest,
        section,
        {
            "code": code,
            "gate": args.gate,
            "read": read_setting(run),
            "db_image": image or previous.get("db_image"),
            "db_extensions": extensions or previous.get("db_extensions"),
            "embedder": emb_model(),
            "reranker": rerank_model(),
            "embedding_input": "{date}: {content}" if run == "dated" else "{content}",
            "constants": _retrieval_constants(),
            "questions": len(packets),
            "load": dict(load),
            "zero_hit_packets": sum(not p["hits"] for p in packets),
            "packets_sha256": sha256_file(out_path) if out_path.exists() else None,
            "metrics": retrieval_metrics(packets, questions),
        },
    )
    print(json.dumps(retrieval_metrics(packets, questions)["overall"] if packets else {}))


def _packets_by_id(data_dir: Path, run: str) -> dict[str, dict[str, Any]]:
    packets: dict[str, dict[str, Any]] = {}
    for packet in read_jsonl(packets_path(data_dir, run)):
        packets.setdefault(packet["question_id"], packet)
    return packets


def current_rows(
    rows: Sequence[dict[str, Any]], expected: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Each question's latest row answering its current prompt (prompt sha matches).

    `expected` maps a question id to the prompt text it is currently given.
    """
    shas = {qid: prompt_sha(text) for qid, text in expected.items()}
    current: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = row.get("question_id")
        if qid in shas and row.get("prompt_sha256") == shas[qid]:
            current[qid] = row
    return current


def current_answers(data_dir: Path, run: str) -> dict[str, dict[str, Any]]:
    """Answers whose prompt matches the current packet."""
    packets = _packets_by_id(data_dir, run)
    expected = {qid: answer_prompt(packet) for qid, packet in packets.items()}
    return current_rows(read_jsonl(stage_output_path(data_dir, "answer", run)), expected)


def stage_prompts(
    stage: str, data_dir: Path, run: str, questions: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Each question's current prompt for the stage, keyed by question id."""
    if stage == "answer":
        packets = _packets_by_id(data_dir, run)
        return {qid: answer_prompt(packet) for qid, packet in packets.items()}
    return {
        qid: judge_prompt(questions[qid], row["text"])
        for qid, row in current_answers(data_dir, run).items()
    }


def _usage(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "questions": len(rows),
        "in_tok": sum(row["in_tok"] for row in rows),
        "out_tok": sum(row["out_tok"] for row in rows),
        "seconds_per_question": _mean([row["seconds"] for row in rows]),
    }


def run_audit_sample(args: argparse.Namespace) -> None:
    run = run_name(args.variant, args.gate, args.read)
    out_path = judge_audit_path(args.data_dir, run)
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; it may hold hand labels, so it is not replaced")
    questions = {q["question_id"]: q for q in load_dataset(args.dataset)}
    answers = current_answers(args.data_dir, run)
    judgments = read_jsonl(stage_output_path(args.data_dir, "judge", run))
    rows = judge_audit_sample(
        list(_packets_by_id(args.data_dir, run)), questions, answers, judgments
    )
    write_jsonl_atomic(out_path, rows)
    print(f"{len(rows)} judgments in {out_path}; set each human_label to true or false")


def _extraction_summary(
    data_dir: Path, question_ids: Sequence[str], questions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    units = {u for qid in question_ids for u in session_units(questions[qid])}
    sessions = [
        row
        for row in read_jsonl(data_dir / SESSIONS_FILE)
        if (row["session_id"], row["date"]) in units
    ]
    notes = [
        row
        for row in read_jsonl(data_dir / NOTES_FILE)
        if (row["session_id"], row["date"]) in units
    ]
    reasons = Counter(
        note["gate_reason"].split(":", 1)[0]
        if note["gate_reason"].startswith(REFUSAL_CAUSES)
        else "gate"
        for note in notes
        if note["gate"] == "refused"
    )
    totals = Counter()
    for row in sessions:
        totals.update({name: row[name] for name in SESSION_TOTALS})

    def spread(name: str) -> dict[str, float | None]:
        values = sorted(row[name] for row in sessions)
        return {
            "mean": _mean(values),
            "median": values[len(values) // 2] if values else None,
            "max": values[-1] if values else None,
        }

    return {
        "units": len(units),
        "units_extracted": len(sessions),
        "notes": len(notes),
        "notes_per_unit": len(notes) / len(sessions) if sessions else None,
        "zero_note_units": sum(row["notes"] == 0 for row in sessions),
        "provider_refused_units": sum(bool(row.get("provider_refused")) for row in sessions),
        "refused_by": dict(sorted(reasons.items())),
        "tokens": dict(totals),
        "seconds_per_unit": spread("seconds"),
        "extract_seconds_per_unit": spread("extract_seconds"),
        "gate": gate_rates(question_ids, questions, notes),
    }


def _run_report(
    data_dir: Path, run: str, questions: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    packets = _packets_by_id(data_dir, run)
    if not packets:
        return None
    question_ids = list(packets)
    answers = current_answers(data_dir, run)
    judgments = read_jsonl(stage_output_path(data_dir, "judge", run))
    judged = _judged(questions, list(answers.values()), judgments)
    load = Counter()
    for packet in packets.values():
        load.update(packet["load"])
    return {
        "questions": len(packets),
        "qa": qa_accuracy(question_ids, questions, list(answers.values()), judgments),
        "retrieval": retrieval_metrics(list(packets.values()), questions),
        "load": dict(load),
        "similar_acks_per_question": load["similar_acks"] / len(packets),
        "answer_usage": _usage(list(answers.values())),
        "judge_usage": _usage(list(judged.values())),
        "judge_audit": judge_agreement(read_jsonl(judge_audit_path(data_dir, run))),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_report(report: dict[str, Any]) -> str:
    lines = [f"# LongMemEval_S subset (n={report['subset']['size']})", ""]
    for run, body in report["runs"].items():
        lines += [f"## {run} (questions with packets: {body['questions']})", ""]
        lines += ["| group | judged | accuracy |", "|---|---|---|"]
        for group, row in body["qa"].items():
            if group != "unjudged":
                lines.append(f"| {group} | {row['judged']} | {_fmt(row['accuracy'])} |")
        lines += ["", f"Unjudged: {len(body['qa']['unjudged'])}", ""]
        names = [f"{m}@{k}" for k in METRIC_KS for m in ("recall_all", "ndcg_any")]
        lines += ["| group | n | " + " | ".join(names) + " |", "|---" * (len(names) + 2) + "|"]
        for group, row in body["retrieval"].items():
            if isinstance(row, dict):
                cells = " | ".join(_fmt(row[name]) for name in names)
                lines.append(f"| {group} | {row['count']} | {cells} |")
        lines += [
            "",
            f"Zero-hit packets: {body['retrieval']['zero_hit_packets']} "
            f"(scored: {body['retrieval']['zero_hit_packets_scored']})",
            f"Load: {json.dumps(body['load'])}",
            f"Answer usage: {json.dumps(body['answer_usage'])}",
            f"Judge usage: {json.dumps(body['judge_usage'])}",
            f"Judge audit: {json.dumps(body['judge_audit'])}",
            "",
        ]
    extraction = report["extraction"]
    lines += [
        "## Extraction",
        "",
        f"Units: {extraction['units_extracted']}/{extraction['units']}, notes: "
        f"{extraction['notes']} ({_fmt(extraction['notes_per_unit'])} per unit), zero-note "
        f"units: {extraction['zero_note_units']}, provider-refused units: "
        f"{extraction['provider_refused_units']}",
        f"Tokens and seconds: {json.dumps(extraction['tokens'])}",
        f"Seconds per unit: {json.dumps(extraction['seconds_per_unit'])}, extraction only: "
        f"{json.dumps(extraction['extract_seconds_per_unit'])}",
        f"Refused by: {json.dumps(extraction['refused_by'])}",
        "",
        "| group | notes | refused | refused rate |",
        "|---|---|---|---|",
    ]
    for group, row in extraction["gate"].items():
        lines.append(
            f"| {group} | {row['notes']} | {row['refused']} | {_fmt(row['refused_rate'])} |"
        )
    return "\n".join(lines) + "\n"


def run_score(args: argparse.Namespace) -> None:
    code = code_revision()
    dataset = load_dataset(args.dataset)
    questions = {q["question_id"]: q for q in dataset}
    subset = select_subset(dataset)
    runs = {}
    for run in RUNS:
        body = _run_report(args.data_dir, run, questions)
        if body is not None:
            runs[run] = body
    if not runs:
        raise SystemExit("no packets yet; run retrieve")
    scope = list(_packets_by_id(args.data_dir, next(iter(runs))))
    report = {
        "subset": subset_manifest(subset, sha256_file(args.dataset)),
        "runs": runs,
        "extraction": _extraction_summary(args.data_dir, scope, questions),
    }
    (args.data_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = render_report(report)
    (args.data_dir / "report.md").write_text(markdown)
    artefacts = {
        path.name: sha256_file(path)
        for path in sorted(args.data_dir.glob("*.jsonl"))
        if path.is_file()
    }
    update_manifest(args.manifest, "upstream", upstream_manifest())
    update_manifest(
        args.manifest,
        "score",
        {
            "code": code,
            "questions_scored": len(scope),
            "extraction": report["extraction"],
            "runs": runs,
            "artefacts_sha256": artefacts,
        },
    )
    print(markdown)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m memory_base.eval.longmemeval")
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--dataset", type=Path, required=True)
        sub.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        sub.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    def run_flags(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--variant", choices=VARIANTS, default="baseline")
        sub.add_argument("--gate", choices=GATES, default="on")
        sub.add_argument("--read", choices=tuple(READ_SETTINGS), default="search")

    retrieve = commands.add_parser("retrieve", help="load notes and search per question")
    common(retrieve)
    run_flags(retrieve)
    retrieve.add_argument("--questions", type=lambda s: s.split(","), default=None)
    audit = commands.add_parser("audit-sample", help="draw the seeded judge-audit sample")
    common(audit)
    run_flags(audit)
    common(commands.add_parser("score", help="report accuracy and retrieval metrics"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    {
        "retrieve": run_retrieve,
        "audit-sample": run_audit_sample,
        "score": run_score,
    }[args.command](args)


if __name__ == "__main__":
    main()
