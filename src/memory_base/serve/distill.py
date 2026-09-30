"""Distill a stored conversation source's undistilled turns into linked notes.

A conversation job reads the source's `distilled_through` cursor, sends the turns
after it to the chat model in batches with the extraction prompt chosen by the
source's origin, saves each returned unit through `save_note` linked to its turns,
and advances the cursor by compare-and-swap after every batch.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from typing import Any, ClassVar

from loguru import logger

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA
from memory_base.core.llm import chat_json
from memory_base.core.schema import ensure_schema_once
from memory_base.retrieval.search import parse_time_bound
from memory_base.serve import notes
from memory_base.serve.job_store import JobBase

DISTILL_BATCH_CHARS = 60_000
DISTILL_TIMEOUT_SECONDS = 180.0
EXTRACT_ATTEMPTS = 2
PROMPT_FILES = {
    "digest": "extract_prompt_digest.txt",
    "personal": "extract_prompt_personal.txt",
}
ORIGIN_PROMPTS = {"claude_code": "digest", "hermes": "personal"}
EXTRACTION_SYSTEM_PROMPT = (
    'Return only JSON: {"notes": [{"content": string, "kind": "note"|"decision"|"episode", '
    '"turn_start": integer, "turn_end": integer, "tags": [string, ...], '
    '"date": "YYYY-MM-DD" (episodes)}]}'
)
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "kind": {"type": "string", "enum": list(notes.NOTE_KINDS)},
                    "turn_start": {"type": "integer"},
                    "turn_end": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "date": {"type": "string"},
                },
                "required": ["content", "kind", "turn_start", "turn_end", "tags"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["notes"],
    "additionalProperties": False,
}


class ExtractionFailed(RuntimeError):
    """The chat model returned unparseable output on every attempt."""


class CursorMoved(RuntimeError):
    """Another job advanced the source's distilled_through while this one ran."""


@dataclass
class ConversationJob(JobBase):
    conversation_id: str
    namespace: str = "default"
    key_id: str = ""
    key_label: str = ""
    result: dict[str, int] | None = None

    RESPONSE_EXCLUDE: ClassVar[frozenset[str]] = frozenset({"key_id", "key_label"})

    @classmethod
    def from_row(cls, row: Any):
        job = super().from_row(row)
        if isinstance(job.result, str):
            job.result = json.loads(job.result)
        return job

    @property
    def kind(self) -> str:
        return "conversation"


def load_prompt(name: str) -> str:
    """The committed extraction prompt `digest` or `personal`, with {date} and {session}."""
    path = resources.files("memory_base.serve").joinpath("prompts", PROMPT_FILES[name])
    return path.read_text(encoding="utf-8")


def _optional_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_units(payload: Any) -> list[dict[str, Any]]:
    """The reply's units as {content, kind} plus whichever optional fields are well-typed.

    A reply without a notes list, or with a unit whose content or kind is not a
    string, raises ValueError.
    """
    units = payload.get("notes") if isinstance(payload, dict) else None
    if not isinstance(units, list):
        raise ValueError("reply has no notes list")
    parsed = []
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("a note is not an object")
        content, kind = unit.get("content"), unit.get("kind", "note")
        if not isinstance(content, str) or not isinstance(kind, str):
            raise ValueError("a note's content or kind is not a string")
        clean: dict[str, Any] = {"content": content, "kind": kind}
        if _optional_int(unit.get("turn_start")) and _optional_int(unit.get("turn_end")):
            clean["turn_start"], clean["turn_end"] = unit["turn_start"], unit["turn_end"]
        tags = unit.get("tags")
        if isinstance(tags, list) and all(isinstance(tag, str) for tag in tags):
            clean["tags"] = tags
        if isinstance(unit.get("date"), str):
            clean["date"] = unit["date"]
        parsed.append(clean)
    return parsed


def parse_extraction(text: str) -> list[dict[str, Any]]:
    """parse_units over a JSON reply text; malformed JSON raises ValueError."""
    return parse_units(json.loads(text))


def render_turns(turns: Sequence[Mapping[str, str]], first: int = 0) -> str:
    """One `[index] role: text` line per turn, numbered from `first`, each cut to the batch cap."""
    return "\n".join(
        f"[{first + offset}] {turn['role']}: {turn['text'][:DISTILL_BATCH_CHARS]}"
        for offset, turn in enumerate(turns)
    )


def batch_ranges(turns: Sequence[Mapping[str, str]], start: int) -> list[tuple[int, int]]:
    """Inclusive turn ranges from `start` of at most DISTILL_BATCH_CHARS each, never splitting a turn."""
    ranges: list[tuple[int, int]] = []
    first, size = start, 0
    for index in range(start, len(turns)):
        length = min(len(turns[index]["text"]), DISTILL_BATCH_CHARS)
        if index > first and size + length > DISTILL_BATCH_CHARS:
            ranges.append((first, index - 1))
            first, size = index, 0
        size += length
    if first < len(turns):
        ranges.append((first, len(turns) - 1))
    return ranges


def build_messages(
    prompt: str, date: str, turns: Sequence[Mapping[str, str]], first: int = 0
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": load_prompt(prompt).format(date=date, session=render_turns(turns, first)),
        },
    ]


async def _extract(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    for attempt in range(EXTRACT_ATTEMPTS):
        try:
            payload = await chat_json(messages, EXTRACTION_SCHEMA, timeout=DISTILL_TIMEOUT_SECONDS)
            return parse_units(payload)
        except ValueError as exc:
            logger.warning("distill: unparseable extraction (attempt {}): {}", attempt + 1, exc)
    raise ExtractionFailed(f"extraction output unparseable after {EXTRACT_ATTEMPTS} attempts")


async def _load_source(conversation_id: str) -> dict[str, Any] | None:
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        row = await conn.fetchrow(
            f"""
            SELECT namespace, origin, started_at, turns, distilled_through, metadata
            FROM "{PG_SCHEMA}".conversation_sources
            WHERE id = $1
            """,
            conversation_id,
        )
    if row is None:
        return None
    source = dict(row)
    for field in ("turns", "metadata"):
        if isinstance(source[field], str):
            source[field] = json.loads(source[field])
    return source


async def _advance_cursor(conversation_id: str, read: int, new: int) -> bool:
    async with db.acquire() as conn:
        await ensure_schema_once(conn)
        status = await conn.execute(
            f"""
            UPDATE "{PG_SCHEMA}".conversation_sources
            SET distilled_through = $3
            WHERE id = $1 AND distilled_through = $2
            """,
            conversation_id,
            read,
            new,
        )
    return status == "UPDATE 1"


def _link(unit: Mapping[str, Any], first: int, last: int) -> tuple[int, int]:
    start, end = unit.get("turn_start"), unit.get("turn_end")
    if start is not None and first <= start <= end <= last:
        return start, end
    return first, last


def _occurred_at(unit: Mapping[str, Any]) -> str | None:
    date = unit.get("date")
    if date is None:
        return None
    try:
        return date if parse_time_bound(date) <= time.time() else None
    except ValueError:
        return None


async def _save_unit(
    unit: Mapping[str, Any],
    first: int,
    last: int,
    *,
    job: ConversationJob,
    source: Mapping[str, Any],
    extra_tags: list[str],
) -> str:
    """Save one unit; the outcome is stored, refused, similar, or duplicate."""
    turn_start, turn_end = _link(unit, first, last)
    tags = [*unit.get("tags", []), *extra_tags] or [source["origin"]]
    try:
        result = await notes.save_note(
            unit["content"],
            tags=tags,
            kind=unit["kind"],
            namespace=source["namespace"],
            occurred_at=_occurred_at(unit),
            author=source["origin"],
            conversation_id=job.conversation_id,
            turn_start=turn_start,
            turn_end=turn_end,
            source_ref="distill",
        )
    except notes.SimilarNotesError:
        return "similar"
    except ValueError as exc:
        reason = getattr(exc, "reason", exc)
        logger.info(
            "distill {} [{}-{}] refused: {}", job.conversation_id, turn_start, turn_end, reason
        )
        return "refused"
    return "stored" if result["stored"] else "duplicate"


async def run_conversation_job(job: ConversationJob) -> None:
    """Distill the turns after the source's cursor; sets job.status and job.result."""
    counts = {"units": 0, "stored": 0, "refused": 0, "similar": 0}
    job.result = counts
    source = await _load_source(job.conversation_id)
    if source is None:
        raise RuntimeError(f"conversation source {job.conversation_id} no longer exists")
    turns, cursor = source["turns"], source["distilled_through"]
    if len(turns) - cursor < 2:
        job.status = "no_op"
        return
    prompt = ORIGIN_PROMPTS.get(source["origin"])
    if prompt is None:
        raise RuntimeError(f"no extraction prompt for origin {source['origin']!r}")
    date = datetime.fromtimestamp(source["started_at"], tz=timezone.utc).strftime("%Y-%m-%d")
    repo = source["metadata"].get("repo")
    extra_tags = [f"repo:{repo}"] if isinstance(repo, str) and repo.strip() else []
    for first, last in batch_ranges(turns, cursor):
        units = await _extract(build_messages(prompt, date, turns[first : last + 1], first))
        for unit in units:
            counts["units"] += 1
            outcome = await _save_unit(
                unit, first, last, job=job, source=source, extra_tags=extra_tags
            )
            if outcome in counts:
                counts[outcome] += 1
        if not await _advance_cursor(job.conversation_id, cursor, last + 1):
            raise CursorMoved(
                f"another job moved conversation source {job.conversation_id}'s "
                f"distilled_through past {cursor}; this job stores nothing further"
            )
        cursor = last + 1
    job.status = "succeeded"
