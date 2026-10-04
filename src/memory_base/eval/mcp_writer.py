"""LongMemEval writer that saves through memory_base's real MCP tools.

Each benchmark session goes to a fresh headless Claude Code process whose only MCP server
is a stdio memory_base server talking to a throwaway REST API on a throwaway Postgres.
The agent sees the published instruction, the session date, and the transcript, and
decides by itself what to save. Each question has its own namespace and key; its sessions
run in date order, and earlier memory is reachable only through search; questions run
concurrently up to --concurrency. The run records every session's tool calls, saves,
refusals, usage, and the notes it created, and exports each question's end-state notes
with provenance.

    uv run python -m memory_base.eval.mcp_writer --dataset PATH --questions ID[,ID...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from memory_base.core.config import PG_SCHEMA
from memory_base.eval.longmemeval import (
    DEFAULT_DATA_DIR,
    NAMESPACE_PREFIX,
    REPO_ROOT,
    _prepare_schema,
    append_jsonl,
    code_revision,
    filter_questions,
    iso_datetime,
    load_dataset,
    read_jsonl,
    select_subset,
    session_turns,
    session_units,
    sha256_file,
    throwaway_postgres,
    write_jsonl_atomic,
)

AUTHOR = "lme-writer"
INSTRUCTION = (
    "The conversation below has just ended. You are the assistant in it. If anything in it "
    "is worth remembering for future conversations with this user, save it with your memory "
    "tools; otherwise do nothing."
)
AUTHOR_LINE = f"Your author name for the memory tools is {AUTHOR}."
SERVER_NAME = "memory-base"
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"
SAVE_TOOLS = ("save_memory",)
# The production REST port; the eval backend must never be reached through it.
PRODUCTION_PORTS = {8010}
SESSION_TIMEOUT_SECONDS = 600.0
API_BOOT_SECONDS = 60.0

SESSIONS_FILE = "writer-sessions.jsonl"
NOTES_FILE = "writer-notes.jsonl"
QUESTIONS_FILE = "writer-questions.jsonl"


def writer_prompt(date: str, turns: Sequence[dict[str, Any]]) -> str:
    """The published instruction, the author line, the session date, and the transcript."""
    transcript = "\n\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in turns)
    return f"{INSTRUCTION}\n{AUTHOR_LINE}\n\nSession date: {iso_datetime(date)}\n\n{transcript}"


def _result_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def parse_stream(lines: Iterable[str]) -> dict[str, Any]:
    """A stream-json session reduced to its tool calls, saves, refusals, and usage."""
    record: dict[str, Any] = {
        "tools": [],
        "mcp_servers": [],
        "tool_calls": [],
        "saves": [],
        "refusals": [],
        "usage": {},
        "num_turns": None,
        "models": [],
        "is_error": True,
        "subtype": "no_result",
        "permission_denials": 0,
    }
    pending: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            record["tools"] = event.get("tools", [])
            record["mcp_servers"] = [server["name"] for server in event.get("mcp_servers", [])]
        elif kind == "assistant":
            for block in event["message"]["content"]:
                if block.get("type") == "tool_use":
                    name = block["name"].removeprefix(TOOL_PREFIX)
                    pending[block["id"]] = name
                    record["tool_calls"].append({"name": name, "input": block["input"]})
        elif kind == "user":
            for block in event["message"].get("content", []):
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                name = pending.get(block.get("tool_use_id"), "")
                if name not in SAVE_TOOLS:
                    continue
                text = _result_text(block.get("content"))
                if block.get("is_error"):
                    record["refusals"].append({"tool": name, "error": text})
                    continue
                result = json.loads(text)
                record["saves"].append(
                    {
                        "tool": name,
                        "id": result["id"],
                        "stored": result["stored"],
                        "superseded": result.get("superseded"),
                    }
                )
        elif kind == "result":
            record["usage"] = {
                key: event.get("usage", {}).get(key, 0)
                for key in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                    "output_tokens",
                )
            }
            record["num_turns"] = event.get("num_turns")
            record["models"] = sorted(event.get("modelUsage", {}))
            record["is_error"] = bool(event.get("is_error"))
            record["subtype"] = event.get("subtype")
            record["permission_denials"] = len(event.get("permission_denials") or [])
    return record


def diff_rows(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Note ids a session created, and ids it archived that were active before it."""
    created = sorted(set(after) - set(before))
    archived = sorted(
        note_id
        for note_id, row in after.items()
        if row["archived"] and note_id in before and not before[note_id]["archived"]
    )
    return created, archived


def record_provenance(
    provenance: dict[str, set[str]],
    session_id: str,
    saves: Sequence[dict[str, Any]],
    *,
    created: Sequence[str],
) -> None:
    """Credit the session with every note it saved, created, or re-saved as a duplicate.

    A supersede rewrite also carries the sessions of the note it replaced, because the
    rewrite states the replaced value with its date."""
    for note_id in created:
        provenance.setdefault(note_id, set()).add(session_id)
    for save in saves:
        sessions = provenance.setdefault(save["id"], set())
        sessions.add(session_id)
        if save["superseded"]:
            sessions.update(provenance.get(save["superseded"], set()))


def mcp_config(repo_root: Path, api_key: str, rest_url: str) -> dict[str, Any]:
    """The writer's only MCP server: a stdio memory_base server on a loopback eval backend."""
    url = urlparse(rest_url)
    if url.hostname != "127.0.0.1" or url.port is None or url.port in PRODUCTION_PORTS:
        raise ValueError(f"the writer must reach a loopback eval backend, not {rest_url}")
    command = ["run", "--no-sync", "--directory", str(repo_root), "python", "-m"]
    return {
        "mcpServers": {
            SERVER_NAME: {
                "type": "stdio",
                "command": "uv",
                "args": [*command, "memory_base.serve.mcp_server"],
                "env": {
                    "MCP_TRANSPORT": "stdio",
                    "MEMORY_API_KEY": api_key,
                    "REST_URL": rest_url,
                },
            }
        }
    }


@dataclass
class ClaudeCodeWriter:
    """One headless Claude Code session per benchmark session, with only memory-base tools.

    Claude Code keeps its default system prompt, so the agent meets the tools the way a
    production client does; a fresh empty working directory per session keeps CLAUDE.md
    and any auto-memory out."""

    model: str
    timeout: float = SESSION_TIMEOUT_SECONDS

    async def run(self, prompt: str, config: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as cwd:
            config_path = Path(cwd) / "mcp.json"
            config_path.write_text(json.dumps(config))
            argv = (
                "claude", "-p", "--model", self.model,
                "--tools", "", "--setting-sources", "", "--strict-mcp-config",
                "--mcp-config", str(config_path), "--allowedTools", f"mcp__{SERVER_NAME}",
                "--settings", json.dumps({"autoMemoryEnabled": False}),
                "--no-session-persistence", "--output-format", "stream-json", "--verbose",
            )  # fmt: skip
            env = {
                **os.environ,
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
            }
            started = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            out, err = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=self.timeout
            )
        record = parse_stream(out.decode().splitlines())
        record["seconds"] = round(time.monotonic() - started, 2)
        record["returncode"] = proc.returncode
        if proc.returncode:
            record["is_error"] = True
            record["stderr"] = err.decode()[:500]
        return record


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class EvalApi:
    """The REST API as a subprocess on a free loopback port, against the throwaway DB_URL."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self.state = tempfile.TemporaryDirectory()

    def __enter__(self) -> EvalApi:
        argv = [
            "uv", "run", "--no-sync", "uvicorn", "memory_base.serve.api:app",
            "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "warning",
        ]  # fmt: skip
        state = Path(self.state.name)
        env = {
            **os.environ,
            "INGEST_SPOOL": str(state / "ingest-spool"),
            "REPO_CACHE": str(state / "repos-cache"),
            "LOG_DIR": str(state / "logs"),
            "COCOINDEX_DB": str(state / "cocoindex"),
        }
        self.proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=env)
        deadline = time.monotonic() + API_BOOT_SECONDS
        while True:
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.__exit__(None, None, None)
                raise RuntimeError(f"eval REST API did not come up on {self.url}")
            time.sleep(0.5)

    def __exit__(self, *exc: Any) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.state.cleanup()


async def _rows(namespace: str) -> dict[str, dict[str, Any]]:
    from memory_base.core import db

    async with db.acquire() as conn:
        rows = await conn.fetch(
            f"SELECT id, chunk_kind, content_raw, archived_at, occurred_at, ts_last_active, metadata "
            f'FROM "{PG_SCHEMA}".memory_chunks WHERE namespace = $1 '
            "AND chunk_kind IN ('personal', 'work')",
            namespace,
        )
    out = {}
    for row in rows:
        metadata = row["metadata"]
        metadata = json.loads(metadata) if isinstance(metadata, str) else dict(metadata)
        out[row["id"]] = {
            "kind": row["chunk_kind"],
            "content": row["content_raw"],
            "archived": row["archived_at"] is not None,
            "occurred_at": row["occurred_at"],
            "saved_at": row["ts_last_active"],
            "author": metadata.get("author"),
            "supersedes": metadata.get("supersedes"),
            "archived_by": metadata.get("archived_by"),
            "tags": metadata.get("tags", []),
        }
    return out


async def _prepare_question(question_id: str) -> tuple[str, str]:
    from memory_base.serve.access import keys, namespaces

    namespace = NAMESPACE_PREFIX + question_id
    await namespaces.create_namespace(namespace)
    label = f"{AUTHOR}-{question_id}"
    api_key = await keys.new_key(label, home=namespace)
    await keys.set_authors(label, [AUTHOR])
    return namespace, api_key


async def write_question(
    question: dict[str, Any],
    turns: dict[tuple[str, str], list[dict]],
    writer: ClaudeCodeWriter,
    api_url: str,
    data_dir: Path,
) -> None:
    """Replay one question's sessions through the writer and export its end-state notes."""
    qid = question["question_id"]
    namespace, api_key = await _prepare_question(qid)
    config = mcp_config(REPO_ROOT, api_key, api_url)
    provenance: dict[str, set[str]] = {}
    evidence = set(question.get("answer_session_ids", []))
    for session_id, date in session_units(question):
        before = await _rows(namespace)
        record = await writer.run(writer_prompt(date, turns[(session_id, date)]), config)
        after = await _rows(namespace)
        created, archived = diff_rows(before, after)
        record_provenance(provenance, session_id, record["saves"], created=created)
        record.update(
            question_id=qid,
            session_id=session_id,
            date=date,
            evidence=session_id in evidence,
            created=created,
            archived=archived,
            occurred_at=[after[i]["occurred_at"] for i in created],
        )
        append_jsonl(data_dir / SESSIONS_FILE, [record])
    end = await _rows(namespace)
    notes = [
        {"question_id": qid, "id": note_id, **row, "sessions": sorted(provenance.get(note_id, ()))}
        for note_id, row in sorted(end.items(), key=lambda item: item[1]["saved_at"])
    ]
    append_jsonl(data_dir / NOTES_FILE, notes)
    append_jsonl(data_dir / QUESTIONS_FILE, [{"question_id": qid, "notes": len(notes)}])


def drop_partial(data_dir: Path) -> set[str]:
    """Completed question ids; session rows of a question that never completed are dropped,
    since its throwaway database died with the run and it restarts from its first session."""
    done = {row["question_id"] for row in read_jsonl(data_dir / QUESTIONS_FILE)}
    sessions = data_dir / SESSIONS_FILE
    if sessions.exists():
        write_jsonl_atomic(
            sessions, [row for row in read_jsonl(sessions) if row["question_id"] in done]
        )
    return done


async def _write_all(
    questions: Sequence[dict[str, Any]], args: argparse.Namespace, api_url: str
) -> None:
    turns = session_turns(questions)
    writer = ClaudeCodeWriter(model=args.model)
    slots = asyncio.Semaphore(args.concurrency)
    finished = 0

    async def one(question: dict[str, Any]) -> None:
        nonlocal finished
        async with slots:
            started = time.monotonic()
            await write_question(question, turns, writer, api_url, args.data_dir)
            finished += 1
            print(
                f"[{finished}/{len(questions)}] {question['question_id']} "
                f"({len(session_units(question))} sessions, {time.monotonic() - started:.0f}s)",
                flush=True,
            )

    await asyncio.gather(*(one(question) for question in questions))


def run(args: argparse.Namespace) -> None:
    args.data_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    selected = filter_questions(select_subset(dataset), args.questions)
    done = drop_partial(args.data_dir)
    pending = [q for q in selected if q["question_id"] not in done]
    print(f"questions: {len(selected)} selected, {len(pending)} pending")
    if not pending:
        return
    manifest = {
        "dataset_sha256": sha256_file(args.dataset),
        "code": code_revision(),
        "writer": {"harness": "claude-code", "model": args.model, "author": AUTHOR},
        "instruction": INSTRUCTION,
        "author_line": AUTHOR_LINE,
        "concurrency": args.concurrency,
        "run_id": secrets.token_hex(4),
    }
    append_jsonl(args.data_dir / "writer-runs.jsonl", [manifest])
    with throwaway_postgres() as database:
        asyncio.run(_prepare_schema(database["url"]))
        with EvalApi() as api:
            asyncio.run(_write_all(pending, args, api.url))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memory_base.eval.mcp_writer")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=lambda value: [part for part in value.split(",") if part],
        default=None,
        help="comma-separated question ids from the seeded subset",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR / "mcp-writer")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--concurrency", type=int, default=1, help="questions written at once")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
