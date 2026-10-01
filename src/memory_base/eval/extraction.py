"""The committed extraction prompts, their JSON reply parser, and the numbered-turn renderer."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib import resources
from typing import Any

BATCH_CHARS = 60_000
PROMPT_FILES = {
    "digest": "extract_prompt_digest.txt",
    "personal": "extract_prompt_personal.txt",
}
EXTRACTION_SYSTEM_PROMPT = (
    'Return only JSON: {"notes": [{"content": string, "kind": "note"|"decision"|"episode", '
    '"turn_start": integer, "turn_end": integer, "tags": [string, ...], '
    '"date": "YYYY-MM-DD" (episodes)}]}'
)
PERSONAL_SYSTEM_PROMPT = 'Return only JSON: {"notes": [{"content": string}]}'


def load_prompt(name: str) -> str:
    """The committed extraction prompt `digest` or `personal`, with {date} and {session}."""
    path = resources.files("memory_base.eval").joinpath("prompts", PROMPT_FILES[name])
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
    """One `[index] role: text` line per turn, numbered from `first`, each cut to BATCH_CHARS."""
    return "\n".join(
        f"[{first + offset}] {turn['role']}: {turn['text'][:BATCH_CHARS]}"
        for offset, turn in enumerate(turns)
    )
