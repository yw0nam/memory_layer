"""What one prompt shows: the recalled items this session has not seen, under the per-prompt caps."""

from __future__ import annotations

from recall import dedup

MAX_DECISIONS = 3
MAX_NOTES = 3


def choose(
    decisions: list[dict], notes: list[dict], scope: str | None, seen: set[str]
) -> tuple[list[dict], list[dict]]:
    """The unseen decisions and notes, each list cut to its cap."""
    fresh_decisions = [
        d for d in decisions if dedup.decision_key(scope, d.get("number")) not in seen
    ]
    fresh_notes = [n for n in notes if str(n.get("id")) not in seen]
    return fresh_decisions[:MAX_DECISIONS], fresh_notes[:MAX_NOTES]
