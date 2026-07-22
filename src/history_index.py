"""Incrementally distill Claude Code JSONL history into ``memory_chunks``.

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
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import asyncpg

from common import DB_URL, LLM_MODEL, PG_SCHEMA, VllmEmbedder, llm_client

LOGGER = logging.getLogger("history_index")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[가-힣]{2,}")
TRUNCATION_MARKER = "\n...[truncated]...\n"
ACTIVE_WINDOW_SECONDS = 10 * 60
SERVICE_TIMEOUT_SECONDS = 120


@dataclass
class Message:
    role: str
    text: str
    timestamp: float | None
    session_id: str
    cwd: str | None = None
    git_branch: str | None = None
    uuid: str | None = None
    parent_uuid: str | None = None
    tool_names: tuple[str, ...] = ()
    tool_error: bool = False


@dataclass
class Burst:
    role: str
    text: str
    messages: list[Message]
    social_weight: float
    mean_idf: float | None = None


@dataclass
class Session:
    session_id: str
    messages: list[Message]
    transcript: str
    ts_last_active: float
    tool_names: list[str] = field(default_factory=list)
    tool_error_count: int = 0


@dataclass
class Distillation:
    one_line_question: str
    summary: str
    resolution: str
    references: list[str]

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


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_jsonl(data: str, default_session_id: str = "") -> list[Message]:
    """Parse supported Claude Code messages, skipping malformed/noise records."""
    messages: list[Message] = []
    latest_by_session: dict[str, Message] = {}
    for line in data.splitlines():
        try:
            item = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        role = item.get("type")
        if role not in ("user", "assistant") or item.get("isSidechain") is True:
            continue
        session_id = str(item.get("sessionId") or default_session_id)
        content = (item.get("message") or {}).get("content")
        texts: list[str] = []
        tool_names: list[str] = []
        tool_error = False
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text" and isinstance(block.get("text"), str):
                    texts.append(block["text"])
                elif role == "assistant" and block_type == "tool_use":
                    if isinstance(block.get("name"), str):
                        tool_names.append(block["name"])
                elif role == "user" and block_type == "tool_result":
                    tool_error = tool_error or block.get("is_error") is True
        text = "\n".join(part.strip() for part in texts if part.strip()).strip()
        if not text:
            # A tool_result-only user record is not a text message, but its
            # failure belongs to the tool-using message immediately before it.
            if tool_error and session_id in latest_by_session:
                latest_by_session[session_id].tool_error = True
            continue
        message = Message(
            role=role,
            text=text,
            timestamp=_timestamp(item.get("timestamp")),
            session_id=session_id,
            cwd=item.get("cwd"),
            git_branch=item.get("gitBranch"),
            uuid=item.get("uuid"),
            parent_uuid=item.get("parentUuid"),
            tool_names=tuple(tool_names),
            tool_error=tool_error,
        )
        messages.append(message)
        latest_by_session[session_id] = message
    return messages


def build_transcript(messages: Sequence[Message], max_chars: int = 100_000) -> str:
    transcript = "\n".join(f"{m.role.upper()}: {m.text}" for m in messages)
    if len(transcript) <= max_chars:
        return transcript
    return transcript[: int(max_chars * 0.6)] + TRUNCATION_MARKER + transcript[-int(max_chars * 0.4) :]


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
    return sum(math.log((document_count + 1) / (dfs.get(token, 0) + 1)) + 1 for token in tokens) / len(tokens)


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


def passes_burst_gate(burst: Burst, mean_idf_value: float, document_count: int) -> bool:
    # ponytail: strict AND gate, relax to scoring if recall suffers
    return (document_count < 20 or mean_idf_value >= 4.0) and burst.social_weight > 1.0


def _valid_session(session: Session) -> bool:
    return len(session.messages) >= 5 and sum(len(m.text) for m in session.messages) >= 500


async def _triage_llm(session: Session) -> bool:
    prompt = (
        "다음 Claude Code 세션이 나중에 '이 문제 어떻게 풀었지?'로 검색할 가치가 있는지 판정하세요. "
        "반드시 JSON 객체만 반환하세요. 스키마: "
        '{"keep":true|false,"reason":"판정 이유"}\n\n'
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


async def _distill(session: Session, semaphore: asyncio.Semaphore) -> Distillation:
    prompt = (
        "다음 Claude Code 세션을 한국어로 증류하세요. 코드, 에러 문자열, 식별자는 원문을 유지하세요. "
        "반드시 JSON 객체만 반환하세요. 스키마: "
        '{"one_line_question":"나중에 검색할 질문 한 줄","summary":"3~5문장 요약",'
        '"resolution":"최종 해결책/결론, 없으면 미해결","references":["파일/시스템/명령 언급"]}\n\n'
        + session.transcript
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
    question = str(parsed.get("one_line_question") or "").strip()
    summary = str(parsed.get("summary") or "").strip()
    resolution = str(parsed.get("resolution") or "미해결").strip()
    refs = parsed.get("references")
    references = [str(ref).strip() for ref in refs] if isinstance(refs, list) else []
    if not question or not summary:
        raise ValueError("LLM distillation omitted required fields")
    return Distillation(question, summary, resolution, [ref for ref in references if ref])


def _vector_literal(vector: Any) -> str:
    return "[" + ",".join(f"{value:.6f}" for value in vector.astype(float)) + "]"


async def _embed(embedder: VllmEmbedder, text: str) -> str:
    vector = await asyncio.wait_for(
        embedder.embed(text), timeout=SERVICE_TIMEOUT_SECONDS
    )
    return _vector_literal(vector)


async def _ensure_schema(conn: asyncpg.Connection) -> None:
    schema = f'"{PG_SCHEMA}"'
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.memory_chunks (
          id text PRIMARY KEY, source_type text NOT NULL, source_ref text NOT NULL,
          chunk_kind text NOT NULL, session_id text NOT NULL, content_raw text NOT NULL,
          distilled text, embedding halfvec(2048) NOT NULL,
          ts_last_active double precision NOT NULL, idf_score double precision,
          metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb
        );
        CREATE INDEX IF NOT EXISTS memory_chunks__fts ON {schema}.memory_chunks
          USING GIN (to_tsvector('simple', content_raw));
        CREATE INDEX IF NOT EXISTS memory_chunks__vec ON {schema}.memory_chunks
          USING hnsw (embedding halfvec_cosine_ops);
        CREATE INDEX IF NOT EXISTS memory_chunks__session ON {schema}.memory_chunks (session_id);
        CREATE TABLE IF NOT EXISTS {schema}.ingest_state (
          file_path text PRIMARY KEY, mtime double precision NOT NULL,
          size bigint NOT NULL, ingested_at double precision NOT NULL
        );
        DROP TABLE IF EXISTS {schema}.df_stats;
        DROP TABLE IF EXISTS {schema}.history_session_tokens;
        CREATE TABLE IF NOT EXISTS {schema}.history_file_sessions (
          file_path text NOT NULL, session_id text NOT NULL,
          PRIMARY KEY (file_path, session_id)
        );
        """
    )


async def _old_session_ids(conn: asyncpg.Connection, file_path: str) -> list[str]:
    return [
        row["session_id"]
        for row in await conn.fetch(
            f'SELECT session_id FROM "{PG_SCHEMA}".history_file_sessions WHERE file_path=$1', file_path
        )
    ]


async def _write_file(
    conn: asyncpg.Connection,
    path: Path,
    stat: Any,
    sessions: list[Session],
    rows: list[dict[str, Any]],
    old_ids: list[str],
) -> None:
    schema = f'"{PG_SCHEMA}"'
    async with conn.transaction():
        if old_ids:
            await conn.execute(f"DELETE FROM {schema}.memory_chunks WHERE session_id=ANY($1::text[])", old_ids)
        await conn.execute(f"DELETE FROM {schema}.history_file_sessions WHERE file_path=$1", str(path))

        for session in sessions:
            await conn.execute(
                f"INSERT INTO {schema}.history_file_sessions(file_path, session_id) VALUES($1,$2)",
                str(path), session.session_id,
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
                    row["id"], "claude_code", row["source_ref"], row["kind"], row["session_id"],
                    row["raw"], row["distilled"], row["embedding"], row["timestamp"],
                    row["idf"], json.dumps(row["metadata"], ensure_ascii=False),
                )
                for row in rows
            ],
        )
        await conn.execute(
            f"INSERT INTO {schema}.ingest_state(file_path,mtime,size,ingested_at) VALUES($1,$2,$3,$4) "
            "ON CONFLICT(file_path) DO UPDATE SET mtime=EXCLUDED.mtime,size=EXCLUDED.size,ingested_at=EXCLUDED.ingested_at",
            str(path), stat.st_mtime, stat.st_size, time.time(),
        )


async def _process_file(
    conn: asyncpg.Connection,
    path: Path,
    stat: Any,
    sessions: list[Session],
    dfs: dict[str, int],
    document_count: int,
    stats: Counter[str],
) -> None:
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
        await _write_file(conn, path, stat, [], [], old_ids)
        stats["files_processed"] += 1
        return

    semaphore = asyncio.Semaphore(4)
    results = await asyncio.gather(*(_distill(session, semaphore) for session in kept), return_exceptions=True)
    if any(isinstance(result, BaseException) for result in results):
        for session, result in zip(kept, results, strict=True):
            if isinstance(result, BaseException):
                LOGGER.warning("distillation failed for %s: %s", session.session_id, result)
                stats["llm_failure"] += 1
        return
    distillations = {session.session_id: result for session, result in zip(kept, results, strict=True)}

    embedder = VllmEmbedder()
    output_rows: list[dict[str, Any]] = []
    try:
        for session in kept:
            distilled = distillations[session.session_id]
            source_ref = f"{path.parent.name}/{session.session_id}"
            metadata = {
                "file_path": str(path), "cwd": session.messages[-1].cwd,
                "git_branch": session.messages[-1].git_branch,
                "tool_names": sorted(set(session.tool_names)), "tool_error_count": session.tool_error_count,
            }
            output_rows.append(
                {
                    "id": f"{session.session_id}:session", "source_ref": source_ref, "kind": "session",
                    "session_id": session.session_id, "raw": session.transcript, "distilled": distilled.text,
                    "embedding": await _embed(embedder, distilled.text),
                    "timestamp": session.ts_last_active, "idf": None, "metadata": metadata,
                }
            )
            accepted_bursts: list[Burst] = []
            for burst in group_bursts(session.messages):
                burst.mean_idf = mean_idf(burst.text, document_count, dfs)
                idf_passes = document_count < 20 or burst.mean_idf >= 4.0
                if passes_burst_gate(burst, burst.mean_idf, document_count):
                    accepted_bursts.append(burst)
                elif idf_passes:
                    stats["burst_no_social"] += 1
                else:
                    stats["low_idf_burst"] += 1
            for index, burst in enumerate(accepted_bursts):
                burst_distilled = f"[{distilled.one_line_question}] {burst.text}"
                output_rows.append(
                    {
                        "id": f"{session.session_id}:burst:{index}", "source_ref": source_ref, "kind": "burst",
                        "session_id": session.session_id, "raw": burst.text, "distilled": burst_distilled,
                        "embedding": await _embed(embedder, burst_distilled),
                        "timestamp": session.ts_last_active, "idf": burst.mean_idf,
                        "metadata": {**metadata, "role": burst.role, "social_weight": burst.social_weight},
                    }
                )
    except Exception as exc:
        LOGGER.warning("embedding failed for %s: %s", path, exc)
        stats["embedding_failure"] += 1
        return

    await _write_file(conn, path, stat, kept, output_rows, old_ids)
    stats["files_processed"] += 1
    stats["session_rows"] += len(kept)
    stats["burst_rows"] += len(output_rows) - len(kept)


def _select_files(args: argparse.Namespace) -> list[Path]:
    files = list((Path.home() / ".claude" / "projects").glob("*/*.jsonl"))
    if args.project:
        files = [path for path in files if args.project in path.parent.name]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[: args.limit] if args.limit is not None else files


def _scan_files(
    files: Sequence[Path], stats: Counter[str]
) -> dict[Path, tuple[Any, list[Session]]]:
    """Parse every selected file once for the corpus pass and ingest pass."""
    scanned: dict[Path, tuple[Any, list[Session]]] = {}
    for path in files:
        try:
            stat = path.stat()
            data = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            LOGGER.warning("cannot read %s: %s", path, exc)
            stats["read_error"] += 1
            continue
        scanned[path] = (
            stat,
            group_sessions(parse_jsonl(data, path.stem), stat.st_mtime),
        )
    return scanned


def _count_dry_run_session(
    session: Session,
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
    bursts = group_bursts(session.messages)
    stats["burst_candidates"] += len(bursts)
    for burst in bursts:
        burst_idf = mean_idf(burst.text, document_count, dfs)
        idf_passes = document_count < 20 or burst_idf >= 4.0
        if passes_burst_gate(burst, burst_idf, document_count):
            stats["burst_rows"] += 1
        elif idf_passes:
            stats["burst_no_social"] += 1
        else:
            stats["low_idf_burst"] += 1


async def run(args: argparse.Namespace) -> Counter[str]:
    stats: Counter[str] = Counter()
    files = _select_files(args)
    stats["files_selected"] = len(files)
    scanned = _scan_files(files, stats)
    valid_sessions = [
        session
        for _, sessions in scanned.values()
        for session in sessions
        if _valid_session(session)
    ]
    dfs, document_count = build_corpus_df(session.transcript for session in valid_sessions)

    if args.dry_run:
        for stat, sessions in scanned.values():
            if time.time() - stat.st_mtime < ACTIVE_WINDOW_SECONDS:
                stats["active"] += 1
                continue
            valid = [session for session in sessions if _valid_session(session)]
            stats["files_processed"] += 1
            stats["noise_session"] += len(sessions) - len(valid)
            for session in valid:
                _count_dry_run_session(session, dfs, document_count, stats)
        return stats

    conn = await asyncpg.connect(DB_URL, timeout=15)
    try:
        await _ensure_schema(conn)
        state_rows = await conn.fetch(f'SELECT file_path,mtime,size FROM "{PG_SCHEMA}".ingest_state')
        state = {row["file_path"]: (float(row["mtime"]), int(row["size"])) for row in state_rows}
        for path, (stat, sessions) in scanned.items():
            if time.time() - stat.st_mtime < ACTIVE_WINDOW_SECONDS:
                stats["active"] += 1
                continue
            if not args.full and state.get(str(path)) == (stat.st_mtime, stat.st_size):
                stats["unchanged"] += 1
                continue
            await _process_file(conn, path, stat, sessions, dfs, document_count, stats)
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
        "triage_keep", "triage_skip_heuristic", "triage_borderline",
        "triage_llm_keep", "triage_llm_skip",
    )
    print("triage: " + " ".join(f"{key}={stats[key]}" for key in triage_keys))
    skip_keys = (
        "active", "unchanged", "noise_session", "low_idf_burst", "burst_no_social",
        "llm_failure", "embedding_failure", "read_error",
    )
    print("skips: " + " ".join(f"{key}={stats[key]}" for key in skip_keys))
    if stats["burst_candidates"]:
        print(f"dry_run_burst_candidates={stats['burst_candidates']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--project")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _print_summary(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
