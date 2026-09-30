"""An emulated client agent that decides at write time whether a LongMemEval note supersedes one.

Eval-only: before each save it searches the question namespace for the note, asks the
writer model to choose `new` or `supersede` over the top candidates, and saves a
supersede as a rewrite that names the replaced note. Every failure falls back to the
plain save and is counted in LoadStats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memory_base.eval import longmemeval as lme
from memory_base.eval.claude_code import strip_fence

WRITER_MODEL = "claude-sonnet-5-5"
WRITER_EFFORT = "medium"
WRITER_TOP_K = 5
WRITER_ATTEMPTS = 3
WRITER_SYSTEM_PROMPT = (
    "You maintain one user's long-term memory. You receive a new note with its date and the "
    "existing notes a search found for it. Decide whether the new note states a newer value of "
    "the same fact that one existing note states. Reply with JSON only: "
    '{"action": "new"} or {"action": "supersede", "index": i, "content": "<rewritten note>"}. '
    "Choose supersede only when the new note updates the same fact of the same entity that "
    "candidate i states; a related note, the same topic but a different fact, or a duplicate "
    'is "new". The rewritten note states the current value and the previous value with their '
    'dates, e.g. "20 dozen eggs as of 2023-05 (30 dozen as of 2023-01)", and keeps everything '
    "else the new note says."
)
WRITER_PROMPT = "New note ({date}):\n{note}\n\nExisting notes:\n{candidates}"


def writer_prompt(content: str, date: str, candidates: list[dict]) -> str:
    lines = "\n".join(
        f"[{index}] ({candidate['date']}) {candidate['text']}"
        for index, candidate in enumerate(candidates)
    )
    return WRITER_PROMPT.format(date=date, note=content, candidates=lines)


def parse_decision(text: str, count: int) -> tuple[str, int | None, str | None]:
    """("new", None, None) or ("supersede", index, content); ValueError when malformed."""
    decision = json.loads(strip_fence(text))
    if not isinstance(decision, dict):
        raise ValueError("the decision is not a JSON object")
    action = decision.get("action")
    if action == "new":
        return "new", None, None
    if action != "supersede":
        raise ValueError(f"unknown action: {action!r}")
    index, content = decision.get("index"), decision.get("content")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
        raise ValueError(f"index out of range: {index!r}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("a supersede needs non-empty content")
    return "supersede", index, content


def _candidate(hit: Any) -> dict[str, Any]:
    date = datetime.fromtimestamp(hit.meta["occurred_at"], tz=timezone.utc).strftime("%Y-%m-%d")
    return {"id": hit.meta["id"], "date": date, "text": hit.text}


@dataclass
class AgentWriter:
    model: Any
    search: Any
    model_name: str
    effort: str

    def config(self) -> dict[str, Any]:
        return {
            "kind": "agent",
            "model": self.model_name,
            "effort": self.effort,
            "prompt_sha": lme.prompt_sha(WRITER_SYSTEM_PROMPT + "\n" + WRITER_PROMPT),
        }

    async def _decide(
        self, content: str, date: str, candidates: list[dict], stats: lme.LoadStats
    ) -> tuple[str, int | None, str | None] | None:
        prompt = writer_prompt(content, date, candidates)
        for _ in range(WRITER_ATTEMPTS):
            stats.agent_calls += 1
            try:
                text, _in_tok, _out_tok = await self.model.complete(prompt, max_tokens=0)
                return parse_decision(text, len(candidates))
            except Exception:
                continue
        return None

    async def save(
        self, save: lme.SaveNote, content: str, *, stats: lme.LoadStats, date: str, **kwargs: Any
    ) -> dict[str, Any]:
        from memory_base.retrieval.search import MIN_SCORE

        try:
            hits = await self.search(
                content, source="memory", namespaces=[kwargs["namespace"]], min_score=MIN_SCORE
            )
        except Exception:
            stats.agent_errors += 1
            return await lme._save_with_retry(save, content, **kwargs)
        candidates = [_candidate(hit) for hit in hits[:WRITER_TOP_K]]
        if not candidates:
            return await lme._save_with_retry(save, content, **kwargs)
        decision = await self._decide(content, date, candidates, stats)
        if decision is None:
            stats.agent_errors += 1
            return await lme._save_with_retry(save, content, **kwargs)
        action, index, rewritten = decision
        if action == "new":
            return await lme._save_with_retry(save, content, **kwargs)
        try:
            result = await lme._save_with_retry(
                save, rewritten, supersedes=candidates[index]["id"], **kwargs
            )
        except ValueError:
            stats.agent_errors += 1
            return await lme._save_with_retry(save, content, **kwargs)
        stats.superseded += 1
        return result
