#!/usr/bin/env python3
"""Emulated agent for the LongMemEval harness: distill each benchmark session into notes.

Every (session_id, date) unit of the selected questions goes once to the extractor model
with one committed prompt: "agent" (a personal assistant's memory writer), "digest" (the
session-digest rules with the memory save policy), or "personal" (the session-digest
rules with the personal memory policy); each returned note is then judged by the production content
gate (after the same length, kind, and credential checks save_note applies first) and its
verdict recorded, or recorded as "unjudged" with `--gate off`, which never calls the gate.
<data-dir>/notes.jsonl holds one line per note, <data-dir>/sessions.jsonl one line per
completed unit, zero-note units included. Rerunning resumes where it stopped.

Usage:
  uv run python scripts/longmemeval/extract.py --dataset PATH [--data-dir DIR] [--gate off]
  ... --prompt digest --backend claude-code --model claude-sonnet-5-5 --effort high
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI

from memory_base.core import llm
from memory_base.core.secrets import find_secret
from memory_base.eval import longmemeval as lme
from memory_base.serve import notes as notes_module

# Run as a script, this file sees its own directory on sys.path, not the package's parent.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longmemeval.answer import ClaudeCodeModel  # noqa: E402

NOTES_FILE = lme.NOTES_FILE
SESSIONS_FILE = lme.SESSIONS_FILE
PROMPT_FILES = {
    "agent": "extract_prompt.txt",
    "digest": "extract_prompt_digest.txt",
    "personal": "extract_prompt_personal.txt",
}
PROMPTS = {
    name: Path(__file__).with_name(file).read_text(encoding="utf-8")
    for name, file in PROMPT_FILES.items()
}
SYSTEM_PROMPT = (
    'Return only JSON: {"notes": [{"content": string, "kind": "note"|"decision"|"episode"}]}'
)
DEFAULT_MODEL = {"zai": "glm-5.3-flash", "claude-code": "claude-sonnet-5-5"}
DEFAULT_CONCURRENCY = 5
EXTRACT_ATTEMPTS = 3
GATE_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5.0
EXTRACT_TIMEOUT_SECONDS = 180.0
# A thinking session reads the whole transcript before it answers.
CLAUDE_EXTRACT_TIMEOUT_SECONDS = 600.0
# z.ai's content filter: a 400 with this code refuses the input itself, so a retry cannot pass.
CONTENT_FILTER_CODE = "1301"

Gate = Callable[[str, str], Awaitable[notes_module.ContentVerdict]]

# Token usage of the gate calls made for the unit running in the current task.
_gate_usage: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "lme_gate_usage", default=None
)


class UnitFailed(RuntimeError):
    """A unit could not be extracted or judged; nothing is written, so a rerun retries it."""


class ProviderRefused(RuntimeError):
    """The provider's content filter refused the session; the unit completes with no notes."""


def is_content_filter_refusal(exc: BaseException) -> bool:
    return (
        isinstance(exc, openai.BadRequestError)
        and exc.status_code == 400
        and exc.code == CONTENT_FILTER_CODE
    )


def prompt_sha256(prompt: str = "agent") -> str:
    return lme.prompt_sha(SYSTEM_PROMPT + "\n" + PROMPTS[prompt])


def render_session(turns: Sequence[dict[str, Any]]) -> str:
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)


def build_messages(
    date: str, turns: Sequence[dict[str, Any]], prompt: str = "agent"
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PROMPTS[prompt].format(date=date, session=render_session(turns)),
        },
    ]


def parse_extraction(text: str) -> list[dict[str, str]]:
    """The reply's notes as {content, kind}; a malformed reply raises ValueError."""
    payload = json.loads(text)
    notes = payload.get("notes") if isinstance(payload, dict) else None
    if not isinstance(notes, list):
        raise ValueError("reply has no notes list")
    parsed = []
    for note in notes:
        if not isinstance(note, dict):
            raise ValueError("a note is not an object")
        content, kind = note.get("content"), note.get("kind", "note")
        if not isinstance(content, str) or not isinstance(kind, str):
            raise ValueError("a note's content or kind is not a string")
        parsed.append({"content": content, "kind": kind})
    return parsed


@dataclass
class OpenAIExtractor:
    """The extractor model on the z.ai OpenAI-compatible endpoint chosen from the env."""

    client: AsyncOpenAI
    model: str
    provider: str = "zai"
    temperature: float | None = 0
    thinking: str = "disabled"

    @classmethod
    def from_env(cls, env: Mapping[str, str], *, model: str) -> OpenAIExtractor:
        provider = llm.resolve_llm_provider(env)
        if provider.name != "zai":
            raise RuntimeError(f"the extractor needs the zai provider, not {provider.name}")
        return cls(AsyncOpenAI(base_url=provider.base_url, api_key=provider.api_key), model)

    async def complete(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            ),
            timeout=EXTRACT_TIMEOUT_SECONDS,
        )
        usage = response.usage
        return (
            response.choices[0].message.content or "",
            usage.prompt_tokens,
            usage.completion_tokens,
        )


@dataclass
class ClaudeCodeExtractor:
    """The extractor as one tool-less headless Claude Code session per unit."""

    model: str
    effort: str
    provider: str = "claude-code"
    temperature: float | None = None

    @property
    def thinking(self) -> str:
        return f"effort {self.effort}"

    async def complete(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        system, user = messages[0]["content"], messages[1]["content"]
        session = ClaudeCodeModel(self.model, self.effort, system, CLAUDE_EXTRACT_TIMEOUT_SECONDS)
        text, in_tok, out_tok = await session.complete(user, max_tokens=0)
        return _strip_fence(text), in_tok, out_tok


def _strip_fence(text: str) -> str:
    """The reply without a surrounding ```json fence, which Claude adds without a JSON mode."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def record_gate_usage() -> None:
    """Wrap the server's chat client factory so gate token usage reaches the unit's counter."""
    factory = llm._openai_client

    def recording_client(provider: llm.LlmProvider) -> AsyncOpenAI:
        client = factory(provider)
        create = client.chat.completions.create

        async def create_and_record(**kwargs: Any) -> Any:
            response = await create(**kwargs)
            usage = _gate_usage.get()
            if usage is not None and response.usage is not None:
                usage["in"] += response.usage.prompt_tokens
                usage["out"] += response.usage.completion_tokens
            return response

        client.chat.completions.create = create_and_record
        return client

    llm._openai_client = recording_client


def read_notes(data_dir: Path) -> list[dict[str, Any]]:
    return lme.read_jsonl(Path(data_dir) / NOTES_FILE)


def read_sessions(data_dir: Path) -> list[dict[str, Any]]:
    return lme.read_jsonl(Path(data_dir) / SESSIONS_FILE)


def prepare_resume(data_dir: Path) -> set[tuple[str, str]]:
    """Drop partial last lines and notes of units that never completed; return completed units."""
    data_dir = Path(data_dir)
    sessions = read_sessions(data_dir)
    completed = {(row["session_id"], row["date"]) for row in sessions}
    notes = [n for n in read_notes(data_dir) if (n["session_id"], n["date"]) in completed]
    if (data_dir / SESSIONS_FILE).exists():
        lme.write_jsonl_atomic(data_dir / SESSIONS_FILE, sessions)
    if (data_dir / NOTES_FILE).exists():
        lme.write_jsonl_atomic(data_dir / NOTES_FILE, notes)
    return completed


async def _judge_with_retry(gate: Gate, content: str, kind: str) -> tuple[Any, int]:
    from memory_base.serve.notes import ContentVerdict

    for attempt in range(GATE_ATTEMPTS):
        try:
            return await gate(content, kind), attempt
        except Exception as exc:
            if is_content_filter_refusal(exc):
                # Production fails open when the gate cannot judge; the note is saved unjudged.
                reason = "content gate unavailable: content_filter"
                return ContentVerdict(accepted=True, reason=reason), attempt
            if attempt == GATE_ATTEMPTS - 1:
                raise UnitFailed(f"content gate unavailable: {exc!r}") from exc
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * 2**attempt)
    raise AssertionError("unreachable")


async def _complete_with_retry(client: Any, messages: list[dict[str, str]]):
    in_tok = out_tok = 0
    for attempt in range(EXTRACT_ATTEMPTS):
        try:
            text, used_in, used_out = await client.complete(messages)
            in_tok += used_in
            out_tok += used_out
            return parse_extraction(text), in_tok, out_tok, attempt
        except Exception as exc:
            if is_content_filter_refusal(exc):
                raise ProviderRefused("content_filter") from exc
            if attempt == EXTRACT_ATTEMPTS - 1:
                raise UnitFailed(f"extraction failed: {exc!r}") from exc
            if not isinstance(exc, ValueError):
                await asyncio.sleep(RETRY_BACKOFF_SECONDS * 2**attempt)
    raise AssertionError("unreachable")


def _save_path_refusal(content: str, kind: str) -> str | None:
    """The refusal save_note would raise before its gate call, if any."""
    if kind not in notes_module.NOTE_KINDS:
        return f"validation: kind must be one of {notes_module.NOTE_KINDS}"
    if len(content) > notes_module.NOTE_MAX_CHARS:
        return f"validation: content exceeds {notes_module.NOTE_MAX_CHARS} chars"
    secret_type = find_secret(content)
    if secret_type is not None:
        return f"credential: {secret_type}"
    return None


async def extract_unit(
    session_id: str,
    date: str,
    turns: Sequence[dict[str, Any]],
    *,
    client: Any,
    gate: Gate | None,
    prompt: str = "agent",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    usage = {"in": 0, "out": 0}
    _gate_usage.set(usage)
    started = time.monotonic()
    try:
        raw_notes, in_tok, out_tok, extract_retries = await _complete_with_retry(
            client, build_messages(date, turns, prompt)
        )
    except ProviderRefused as refusal:
        return [], _refused_session(session_id, date, client, started, str(refusal))
    extracted = time.monotonic()
    rows: list[dict[str, Any]] = []
    gate_calls = gate_retries = 0
    for note in raw_notes:
        content = note["content"].strip()
        if not content:
            continue
        kind = note["kind"]
        reason = _save_path_refusal(content, kind)
        if reason is not None:
            outcome = "refused"
        elif gate is None:
            outcome, reason = "unjudged", "gate off"
        else:
            verdict, retries = await _judge_with_retry(gate, content, kind)
            gate_calls += 1 + retries
            gate_retries += retries
            outcome, reason = ("stored" if verdict.accepted else "refused"), verdict.reason
        rows.append(
            {
                "session_id": session_id,
                "date": date,
                "content": content,
                "kind": kind,
                "gate": outcome,
                "gate_reason": reason,
            }
        )
    session = {
        "session_id": session_id,
        "date": date,
        "notes": len(rows),
        "stored": sum(row["gate"] == "stored" for row in rows),
        "in_tok": in_tok,
        "out_tok": out_tok,
        "gate_in_tok": usage["in"],
        "gate_out_tok": usage["out"],
        "gate_calls": gate_calls,
        "gate_retries": gate_retries,
        "extract_retries": extract_retries,
        "extract_seconds": round(extracted - started, 3),
        "seconds": round(time.monotonic() - started, 3),
        "model": getattr(client, "model", None),
    }
    return rows, session


def _refused_session(
    session_id: str, date: str, client: Any, started: float, refused: str
) -> dict[str, Any]:
    """The completion record of a unit the provider refused: no notes and no usage."""
    seconds = round(time.monotonic() - started, 3)
    return {
        "session_id": session_id,
        "date": date,
        "notes": 0,
        "stored": 0,
        **{name: 0 for name in lme.SESSION_TOTALS},
        "extract_seconds": seconds,
        "seconds": seconds,
        "model": getattr(client, "model", None),
        "provider_refused": refused,
    }


async def extract_units(
    units: Sequence[tuple[str, str, Sequence[dict[str, Any]]]],
    data_dir: Path,
    *,
    client: Any,
    gate: Gate | None,
    concurrency: int = DEFAULT_CONCURRENCY,
    prompt: str = "agent",
) -> dict[str, Any]:
    """Extract every unit not completed yet; each unit's lines are appended once it is done."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    completed = prepare_resume(data_dir)
    pending: dict[tuple[str, str], Sequence[dict[str, Any]]] = {}
    for session_id, date, turns in units:
        if (session_id, date) not in completed:
            pending.setdefault((session_id, date), turns)
    semaphore = asyncio.Semaphore(concurrency)
    summary: dict[str, Any] = {"skipped": len(units) - len(pending), "completed": 0, "failed": 0}
    failures: list[str] = []

    async def one(unit: tuple[str, str], turns: Sequence[dict[str, Any]]) -> None:
        async with semaphore:
            try:
                rows, session = await extract_unit(
                    *unit, turns, client=client, gate=gate, prompt=prompt
                )
            except UnitFailed as exc:
                summary["failed"] += 1
                failures.append(f"{unit[0]} {unit[1]}: {exc}")
                return
            lme.append_jsonl(data_dir / NOTES_FILE, rows)
            lme.append_jsonl(data_dir / SESSIONS_FILE, [session])
            summary["completed"] += 1
            done = summary["completed"] + summary["failed"]
            if done % 25 == 0 or done == len(pending):
                print(f"{done}/{len(pending)} units", flush=True)

    await asyncio.gather(*(one(unit, turns) for unit, turns in pending.items()))
    summary["failures"] = failures
    return summary


def _units_for(questions: Sequence[dict[str, Any]]):
    turns = lme.session_turns(questions)
    units: dict[tuple[str, str], None] = {}
    for question in questions:
        units.update(dict.fromkeys(lme.session_units(question)))
    return [(sid, date, turns[(sid, date)]) for sid, date in units]


def _extract_manifest(
    args: argparse.Namespace,
    selected: Sequence[dict[str, Any]],
    client: OpenAIExtractor | ClaudeCodeExtractor,
    summary: dict[str, Any],
    code: dict[str, Any],
) -> dict[str, Any]:
    units = {(sid, date) for sid, date, _ in _units_for(selected)}
    sessions = [s for s in read_sessions(args.data_dir) if (s["session_id"], s["date"]) in units]
    notes = [n for n in read_notes(args.data_dir) if (n["session_id"], n["date"]) in units]
    gate_provider = None if args.gate == "off" else llm.resolve_llm_provider(os.environ)
    totals = {name: sum(s[name] for s in sessions) for name in lme.SESSION_TOTALS}
    return {
        "code": code,
        "questions": None if args.questions is None else [q["question_id"] for q in selected],
        "extractor": {
            "provider": client.provider,
            "model": client.model,
            "temperature": client.temperature,
            "thinking": client.thinking,
            "response_format": "json_object" if client.provider == "zai" else None,
            "prompt": args.prompt,
            "prompt_sha256": prompt_sha256(args.prompt),
            "concurrency": args.concurrency,
        },
        "gate": gate_provider
        and {
            "provider": gate_provider.name,
            "model": gate_provider.model,
            "judge_prompt_sha256": lme.prompt_sha(notes_module.JUDGE_PROMPT),
        },
        "units": {
            "selected": len(units),
            "completed": len(sessions),
            "provider_refused": sum(bool(s.get("provider_refused")) for s in sessions),
            "failed_this_run": summary["failed"],
        },
        "notes": {
            "total": len(notes),
            "stored": sum(n["gate"] == "stored" for n in notes),
            "refused": sum(n["gate"] == "refused" for n in notes),
        },
        "totals": totals,
        "notes_sha256": lme.sha256_file(args.data_dir / NOTES_FILE),
        "sessions_sha256": lme.sha256_file(args.data_dir / SESSIONS_FILE),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=lme.DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=lme.DEFAULT_MANIFEST)
    parser.add_argument("--prompt", choices=tuple(PROMPTS), default="agent")
    parser.add_argument("--backend", choices=tuple(DEFAULT_MODEL), default="zai")
    parser.add_argument("--model")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--gate", choices=lme.GATES, default="on")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--questions", type=lambda s: s.split(","), default=None)
    args = parser.parse_args(argv)

    code = lme.code_revision()
    load_dotenv()
    dataset = lme.load_dataset(args.dataset)
    subset = lme.select_subset(dataset)
    selected = lme.filter_questions(subset, args.questions)
    model = args.model or DEFAULT_MODEL[args.backend]
    client = (
        OpenAIExtractor.from_env(os.environ, model=model)
        if args.backend == "zai"
        else ClaudeCodeExtractor(model, args.effort)
    )
    if args.gate == "on":
        record_gate_usage()
    units = _units_for(selected)
    print(f"questions: {len(selected)}, units: {len(units)}", flush=True)
    summary = asyncio.run(
        extract_units(
            units,
            args.data_dir,
            client=client,
            gate=notes_module.judge_note_content if args.gate == "on" else None,
            concurrency=args.concurrency,
            prompt=args.prompt,
        )
    )
    for failure in summary["failures"]:
        print(f"failed: {failure}")
    lme.update_manifest(
        args.manifest,
        "subset",
        lme.subset_manifest(subset, lme.sha256_file(args.dataset)),
    )
    lme.update_manifest(
        args.manifest, "extract", _extract_manifest(args, selected, client, summary, code)
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}))


if __name__ == "__main__":
    main()
