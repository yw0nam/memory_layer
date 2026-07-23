"""Incrementally distill source history into ``memory_chunks``.

The JSONL parsing and burst construction functions in this module are pure so
they can be tested without Postgres, an LLM, or an embedding service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import asyncpg

from memory_base.adapters import ADAPTERS
from memory_base.adapters.base import Burst, Message, Session, SourceAdapter, SourceFile
from memory_base.adapters.claude_code import parse_jsonl as parse_jsonl
from memory_base.common import (
    DB_URL,
    LLM_MODEL,
    PG_SCHEMA,
    SERVICE_TIMEOUT_SECONDS,
    VllmEmbedder,
    llm_client,
)
from memory_base.common import embed_text as _embed
from memory_base.schema import ensure_schema as _ensure_schema

LOGGER = logging.getLogger("history_index")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[\uac00-\ud7a3]{2,}")
TRUNCATION_MARKER = "\n...[truncated]...\n"
ACTIVE_WINDOW_SECONDS = 10 * 60


@dataclass
class Distillation:
    one_line_question: str
    summary: str
    resolution: str
    references: list[str]
    decisions: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(
            (
                self.one_line_question,
                self.summary,
                self.resolution,
                ", ".join(self.references),
            )
        )


def build_transcript(messages: Sequence[Message], max_chars: int = 100_000) -> str:
    transcript = "\n".join(f"{m.role.upper()}: {m.text}" for m in messages)
    if len(transcript) <= max_chars:
        return transcript
    return (
        transcript[: int(max_chars * 0.6)] + TRUNCATION_MARKER + transcript[-int(max_chars * 0.4) :]
    )


def group_sessions(messages: Iterable[Message], fallback_timestamp: float = 0.0) -> list[Session]:
    grouped: dict[str, list[Message]] = defaultdict(list)
    for message in messages:
        grouped[message.session_id].append(message)
    sessions: list[Session] = []
    for session_id, session_messages in grouped.items():
        timestamps = [m.timestamp for m in session_messages if m.timestamp is not None]
        sessions.append(
            Session(
                session_id=session_id,
                messages=session_messages,
                transcript=build_transcript(session_messages),
                ts_last_active=max(timestamps, default=fallback_timestamp),
                tool_names=[name for m in session_messages for name in m.tool_names],
                tool_error_count=sum(m.tool_error for m in session_messages),
            )
        )
    return sessions


def group_bursts(messages: Sequence[Message], min_chars: int = 200) -> list[Burst]:
    """Group consecutive same-role text messages and apply the length filter."""
    grouped: list[list[Message]] = []
    for message in messages:
        if grouped and grouped[-1][0].role == message.role:
            grouped[-1].append(message)
        else:
            grouped.append([message])

    bursts: list[Burst] = []
    for index, members in enumerate(grouped):
        text = "\n".join(m.text for m in members)
        if len(text) < min_chars:
            continue
        has_error = any(m.tool_error for m in members)
        followed_by_quick_user = False
        if index + 1 < len(grouped) and grouped[index + 1][0].role == "user":
            end_ts = members[-1].timestamp
            next_ts = grouped[index + 1][0].timestamp
            followed_by_quick_user = (
                end_ts is not None and next_ts is not None and 0 <= next_ts - end_ts <= 180
            )
        bursts.append(
            Burst(
                role=members[0].role,
                text=text,
                messages=list(members),
                social_weight=1.5 if has_error or followed_by_quick_user else 1.0,
            )
        )
    return bursts


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def mean_idf(text: str, document_count: int, dfs: dict[str, int]) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return sum(
        math.log((document_count + 1) / (dfs.get(token, 0) + 1)) + 1 for token in tokens
    ) / len(tokens)


def build_corpus_df(transcripts: Iterable[str]) -> tuple[dict[str, int], int]:
    """Return document frequencies and the number of supplied transcripts."""
    dfs: Counter[str] = Counter()
    document_count = 0
    for transcript in transcripts:
        document_count += 1
        dfs.update(tokenize(transcript))
    return dict(dfs), document_count


def triage_heuristic(session: Session) -> str:
    """Cheaply classify a session before any LLM or embedding work."""
    text_messages = [message for message in session.messages if message.text]
    assistant_messages = [message for message in text_messages if message.role == "assistant"]
    if not assistant_messages:
        return "skip"

    user_messages = [message for message in text_messages if message.role == "user"]
    user_chars = sum(len(message.text) for message in user_messages)
    if len(user_messages) <= 2 and user_chars < 200:
        return "skip"

    total_chars = sum(len(message.text) for message in text_messages)
    if total_chars >= 5_000 or len(text_messages) >= 20:
        return "keep"
    return "borderline"


def burst_score(mean_idf_value: float, has_social: bool) -> float:
    return mean_idf_value + (1.0 if has_social else 0.0)


def passes_burst_gate(
    burst: Burst,
    mean_idf_value: float,
    document_count: int,
    has_social: bool | None = None,
) -> bool:
    if document_count < 20:
        return True
    if has_social is None:
        has_social = burst.social_weight > 1.0
    return burst_score(mean_idf_value, has_social) >= 4.0


def _valid_session(session: Session) -> bool:
    return len(session.messages) >= 5 and sum(len(m.text) for m in session.messages) >= 500


async def _triage_llm(session: Session) -> bool:
    prompt = (
        "Determine whether the following Claude Code session is worth retrieving later "
        "to answer a how-did-we-solve-this question. Return only a JSON object using this "
        'schema: {"keep":true|false,"reason":"reason for the judgment in English"}\n\n'
        + session.transcript[:3_000]
    )
    try:
        response = await asyncio.wait_for(
            llm_client().chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            ),
            timeout=SERVICE_TIMEOUT_SECONDS,
        )
        content = response.choices[0].message.content or ""
        parsed = json.loads(content)
        if not isinstance(parsed.get("keep"), bool):
            raise ValueError("LLM triage omitted boolean keep")
        return parsed["keep"]
    except Exception as exc:
        LOGGER.warning("triage failed for %s; keeping session: %s", session.session_id, exc)
        return True


def parse_distillation(parsed: dict[str, Any]) -> Distillation:
    question = str(parsed.get("one_line_question") or "").strip()
    summary = str(parsed.get("summary") or "").strip()
    resolution = str(parsed.get("resolution") or "unresolved").strip()

    refs = parsed.get("references")
    references = [str(ref).strip() for ref in refs] if isinstance(refs, list) else []

    raw_decisions = parsed.get("decisions")
    decisions = (
        [str(decision).strip() for decision in raw_decisions]
        if isinstance(raw_decisions, list)
        else []
    )
    if not question or not summary:
        raise ValueError("LLM distillation omitted required fields")
    return Distillation(
        question,
        summary,
        resolution,
        [ref for ref in references if ref],
        [decision for decision in decisions if decision][:10],
    )


async def _distill(session: Session, semaphore: asyncio.Semaphore) -> Distillation:
    prompt = (
        "Distill the following Claude Code session in English, even when the transcript "
        "is in Korean or Japanese. Keep code, error strings, identifiers, and file paths "
        "verbatim. Return only a JSON object using this schema: "
        '{"one_line_question":"one-line question for later retrieval in English",'
        '"summary":"3-5 sentence summary in English",'
        '"resolution":"final solution or conclusion in English, or unresolved",'
        '"references":["mentioned files, systems, or commands"],'
        '"decisions":["one sentence in English for each key decision, stating what was '
        'decided and why; use an empty array if there were none"]}\n\n' + session.transcript
    )
    async with semaphore:
        response = await asyncio.wait_for(
            llm_client().chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            ),
            timeout=SERVICE_TIMEOUT_SECONDS,
        )
    content = response.choices[0].message.content or ""
    parsed = json.loads(content)
    raw_decisions = parsed.get("decisions")
    if isinstance(raw_decisions, list) and len(raw_decisions) > 10:
        LOGGER.warning(
            "distillation returned %d decisions for %s; keeping first 10",
            len(raw_decisions),
            session.session_id,
        )
    return parse_distillation(parsed)


async def _old_session_ids(conn: asyncpg.Connection, file_path: str) -> list[str]:
    return [
        row["session_id"]
        for row in await conn.fetch(
            f'SELECT session_id FROM "{PG_SCHEMA}".history_file_sessions WHERE file_path=$1',
            file_path,
        )
    ]


async def _write_file(
    conn: asyncpg.Connection,
    file: SourceFile,
    sessions: list[Session],
    rows: list[dict[str, Any]],
    old_ids: list[str],
) -> None:
    path = file.path
    schema = f'"{PG_SCHEMA}"'
    async with conn.transaction():
        if old_ids:
            await conn.execute(
                f"DELETE FROM {schema}.memory_chunks WHERE session_id=ANY($1::text[])", old_ids
            )
        await conn.execute(
            f"DELETE FROM {schema}.history_file_sessions WHERE file_path=$1", str(path)
        )

        for session in sessions:
            await conn.execute(
                f"INSERT INTO {schema}.history_file_sessions(file_path, session_id) VALUES($1,$2)",
                str(path),
                session.session_id,
            )
        insert_sql = f"""
            INSERT INTO {schema}.memory_chunks
              (id, source_type, source_ref, chunk_kind, session_id, content_raw,
               distilled, embedding, ts_last_active, idf_score, metadata)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8::halfvec,$9,$10,$11::jsonb)
        """
        await conn.executemany(
            insert_sql,
            [
                (
                    row["id"],
                    row["source_type"],
                    row["source_ref"],
                    row["kind"],
                    row["session_id"],
                    row["raw"],
                    row["distilled"],
                    row["embedding"],
                    row["timestamp"],
                    row["idf"],
                    json.dumps(row["metadata"], ensure_ascii=False),
                )
                for row in rows
            ],
        )
        await conn.execute(
            f"INSERT INTO {schema}.ingest_state(file_path,mtime,size,ingested_at) VALUES($1,$2,$3,$4) "
            "ON CONFLICT(file_path) DO UPDATE SET mtime=EXCLUDED.mtime,size=EXCLUDED.size,ingested_at=EXCLUDED.ingested_at",
            str(path),
            file.mtime,
            file.size,
            time.time(),
        )


def build_rows(
    session: Session,
    distillation: Distillation,
    adapter: SourceAdapter,
    dfs: dict[str, int],
    n: int,
) -> list[dict[str, Any]]:
    file = getattr(session, "_source_file", None)
    path = file.path if file is not None else None
    source_ref = (
        f"{path.parent.name}/{session.session_id}" if path is not None else session.session_id
    )
    metadata = {
        "file_path": str(path) if path is not None else "",
        "cwd": session.messages[-1].cwd,
        "git_branch": session.messages[-1].git_branch,
        "tool_names": sorted(set(session.tool_names)),
        "tool_error_count": session.tool_error_count,
    }
    rows = [
        {
            "id": f"{session.session_id}:session",
            "source_ref": source_ref,
            "kind": "session",
            "session_id": session.session_id,
            "raw": session.transcript,
            "distilled": distillation.text,
            "timestamp": session.ts_last_active,
            "idf": None,
            "metadata": metadata,
            "source_type": adapter.source_type,
        }
    ]
    for index, decision in enumerate(distillation.decisions):
        decision_distilled = f"[{distillation.one_line_question}] Decision: {decision}"
        rows.append(
            {
                "id": f"{session.session_id}:decision:{index}",
                "source_ref": source_ref,
                "kind": "decision",
                "session_id": session.session_id,
                "raw": decision,
                "distilled": decision_distilled,
                "timestamp": session.ts_last_active,
                "idf": None,
                "metadata": {**metadata, "index": index},
                "source_type": adapter.source_type,
            }
        )
    if not adapter.emit_bursts:
        return rows
    accepted_bursts: list[Burst] = []
    for burst in group_bursts(session.messages):
        burst.mean_idf = mean_idf(burst.text, n, dfs)
        if passes_burst_gate(burst, burst.mean_idf, n, adapter.has_social(burst)):
            accepted_bursts.append(burst)
    for index, burst in enumerate(accepted_bursts):
        burst_distilled = f"[{distillation.one_line_question}] {burst.text}"
        rows.append(
            {
                "id": f"{session.session_id}:burst:{index}",
                "source_ref": source_ref,
                "kind": "burst",
                "session_id": session.session_id,
                "raw": burst.text,
                "distilled": burst_distilled,
                "timestamp": session.ts_last_active,
                "idf": burst.mean_idf,
                "metadata": {
                    **metadata,
                    "role": burst.role,
                    "social_weight": burst.social_weight,
                },
                "source_type": adapter.source_type,
            }
        )
    return rows


async def _process_file(
    conn: asyncpg.Connection,
    file: SourceFile,
    adapter: SourceAdapter,
    sessions: list[Session],
    dfs: dict[str, int],
    document_count: int,
    stats: Counter[str],
) -> None:
    path = file.path
    valid = [session for session in sessions if _valid_session(session)]
    stats["noise_session"] += len(sessions) - len(valid)

    kept: list[Session] = []
    borderline: list[Session] = []
    for session in valid:
        decision = triage_heuristic(session)
        if decision == "keep":
            stats["triage_keep"] += 1
            kept.append(session)
        elif decision == "skip":
            stats["triage_skip_heuristic"] += 1
        else:
            stats["triage_borderline"] += 1
            borderline.append(session)

    if borderline:
        triage_results = await asyncio.gather(*(_triage_llm(session) for session in borderline))
        for session, keep in zip(borderline, triage_results, strict=True):
            if keep:
                stats["triage_llm_keep"] += 1
                kept.append(session)
            else:
                stats["triage_llm_skip"] += 1

    old_ids = await _old_session_ids(conn, str(path))
    if not kept:
        # All-noise and all-skipped files still advance ingest_state.
        await _write_file(conn, file, [], [], old_ids)
        stats["files_processed"] += 1
        return

    semaphore = asyncio.Semaphore(4)
    results = await asyncio.gather(
        *(_distill(session, semaphore) for session in kept), return_exceptions=True
    )
    if any(isinstance(result, BaseException) for result in results):
        for session, result in zip(kept, results, strict=True):
            if isinstance(result, BaseException):
                LOGGER.warning("distillation failed for %s: %s", session.session_id, result)
                stats["llm_failure"] += 1
        return
    distillations = {
        session.session_id: result for session, result in zip(kept, results, strict=True)
    }

    embedder = VllmEmbedder()
    output_rows: list[dict[str, Any]] = []
    try:
        for session in kept:
            distilled = distillations[session.session_id]
            rows = build_rows(session, distilled, adapter, dfs, document_count)
            non_burst_rows = [row for row in rows if row["kind"] != "burst"]
            burst_rows = [row for row in rows if row["kind"] == "burst"]
            for row in non_burst_rows:
                row["embedding"] = await _embed(embedder, row["distilled"])
                output_rows.append(row)
            if adapter.emit_bursts:
                stats["burst_below_threshold"] += len(group_bursts(session.messages)) - len(
                    burst_rows
                )
            for row in burst_rows:
                row["embedding"] = await _embed(embedder, row["distilled"])
                output_rows.append(row)
    except Exception as exc:
        LOGGER.warning("embedding failed for %s: %s", path, exc)
        stats["embedding_failure"] += 1
        return

    await _write_file(conn, file, kept, output_rows, old_ids)
    stats["files_processed"] += 1
    stats["session_rows"] += len(kept)
    stats["burst_rows"] += sum(row["kind"] == "burst" for row in output_rows)


def _select_files(args: argparse.Namespace, adapter: SourceAdapter) -> list[SourceFile]:
    files = adapter.discover()
    if args.project:
        files = [file for file in files if args.project in file.path.parent.name]
    return files[: args.limit] if args.limit is not None else files


def _scan_files(
    files: Sequence[SourceFile], adapter: SourceAdapter, stats: Counter[str]
) -> dict[SourceFile, list[Session]]:
    """Parse every selected file once for the corpus pass and ingest pass."""
    scanned: dict[SourceFile, list[Session]] = {}
    for file in files:
        try:
            data = file.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            LOGGER.warning("cannot read %s: %s", file.path, exc)
            stats["read_error"] += 1
            continue
        sessions = adapter.parse(data, file)
        scanned[file] = sessions
    return scanned


def _count_dry_run_session(
    session: Session,
    adapter: SourceAdapter,
    dfs: dict[str, int],
    document_count: int,
    stats: Counter[str],
) -> None:
    decision = triage_heuristic(session)
    if decision == "skip":
        stats["triage_skip_heuristic"] += 1
        return
    if decision == "borderline":
        stats["triage_borderline"] += 1
        return

    stats["triage_keep"] += 1
    stats["session_rows"] += 1
    if not adapter.emit_bursts:
        return
    bursts = group_bursts(session.messages)
    stats["burst_candidates"] += len(bursts)
    for burst in bursts:
        burst_idf = mean_idf(burst.text, document_count, dfs)
        if passes_burst_gate(burst, burst_idf, document_count, adapter.has_social(burst)):
            stats["burst_rows"] += 1
        else:
            stats["burst_below_threshold"] += 1


async def run(args: argparse.Namespace) -> Counter[str]:
    stats: Counter[str] = Counter()
    selected_adapters = (
        {args.adapter: ADAPTERS[args.adapter]} if getattr(args, "adapter", None) else ADAPTERS
    )
    scanned: dict[SourceFile, tuple[SourceAdapter, list[Session]]] = {}
    for adapter in selected_adapters.values():
        files = _select_files(args, adapter)
        stats["files_selected"] += len(files)
        scanned.update(
            (file, (adapter, sessions))
            for file, sessions in _scan_files(files, adapter, stats).items()
        )
    valid_sessions = [
        session
        for _, sessions in scanned.values()
        for session in sessions
        if _valid_session(session)
    ]
    dfs, document_count = build_corpus_df(session.transcript for session in valid_sessions)

    if args.dry_run:
        for file, (adapter, sessions) in scanned.items():
            if time.time() - file.mtime < ACTIVE_WINDOW_SECONDS:
                stats["active"] += 1
                continue
            valid = [session for session in sessions if _valid_session(session)]
            stats["files_processed"] += 1
            stats["noise_session"] += len(sessions) - len(valid)
            for session in valid:
                _count_dry_run_session(session, adapter, dfs, document_count, stats)
        return stats

    conn = await asyncpg.connect(DB_URL, timeout=15)
    try:
        await _ensure_schema(conn)
        state_rows = await conn.fetch(
            f'SELECT file_path,mtime,size FROM "{PG_SCHEMA}".ingest_state'
        )
        state = {row["file_path"]: (float(row["mtime"]), int(row["size"])) for row in state_rows}
        for file, (adapter, sessions) in scanned.items():
            if time.time() - file.mtime < ACTIVE_WINDOW_SECONDS:
                stats["active"] += 1
                continue
            if not args.full and state.get(str(file.path)) == (file.mtime, file.size):
                stats["unchanged"] += 1
                continue
            await _process_file(conn, file, adapter, sessions, dfs, document_count, stats)
    finally:
        await conn.close()
    return stats


def _print_summary(stats: Counter[str]) -> None:
    print(
        "summary: "
        f"selected={stats['files_selected']} processed_files={stats['files_processed']} "
        f"session_rows={stats['session_rows']} burst_rows={stats['burst_rows']}"
    )
    triage_keys = (
        "triage_keep",
        "triage_skip_heuristic",
        "triage_borderline",
        "triage_llm_keep",
        "triage_llm_skip",
    )
    print("triage: " + " ".join(f"{key}={stats[key]}" for key in triage_keys))
    skip_keys = (
        "active",
        "unchanged",
        "noise_session",
        "burst_below_threshold",
        "llm_failure",
        "embedding_failure",
        "read_error",
    )
    print("skips: " + " ".join(f"{key}={stats[key]}" for key in skip_keys))
    if stats["burst_candidates"]:
        print(f"dry_run_burst_candidates={stats['burst_candidates']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--project")
    parser.add_argument("--adapter", choices=ADAPTERS)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _print_summary(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
