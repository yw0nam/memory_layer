"""The recall query for one prompt and its one note search, bounded by a cutoff."""

from __future__ import annotations

import re
import threading
import time

MIN_SCORE = 0.5
# TOP_K is also the cap on notes per prompt.
TOP_K = 3
QUERY_CHARS = 2000
MIN_QUERY_CHARS = 20

_FENCED = re.compile(
    r"<\s*(memory-context|memory-recall)\s*>.*?<\s*/\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)


def query(prompt: str) -> str:
    """The prompt without the hooks' complete memory fences, cut to 2000 characters.

    A prompt shorter than 20 characters or a slash command gives "": nothing to recall.
    """
    text = _FENCED.sub(" ", prompt).strip()
    if len(text) < MIN_QUERY_CHARS or text.startswith("/"):
        return ""
    return text[:QUERY_CHARS]


def search(post, text: str, cutoff: float) -> list[dict]:
    """The note hits scoring at least MIN_SCORE that arrived by `cutoff` (monotonic), else []."""
    hits: list[dict] = []

    def ask() -> None:
        try:
            reply = post(
                "/search",
                {"query": text, "top_k": TOP_K, "min_score": MIN_SCORE},
                timeout=max(cutoff - time.monotonic(), 0.1),
            )
            if isinstance(reply, list):
                hits.extend(
                    hit
                    for hit in reply
                    if isinstance(hit, dict) and (hit.get("score") or 0) >= MIN_SCORE
                )
        except Exception:
            pass

    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    thread.join(max(cutoff - time.monotonic(), 0))
    return [] if thread.is_alive() else hits
