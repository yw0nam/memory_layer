"""The recall query for one prompt and its one /recall call, bounded by a cutoff."""

from __future__ import annotations

import re
import threading
import time

NOTES_FLOOR = 0.5
DECISIONS_FLOOR = 0.5
# Asked of the server; the shown ones are dropped before the caps in prompt_recall.
ASK = 5
QUERY_CHARS = 2000
MIN_QUERY_CHARS = 20

_FENCED = re.compile(
    r"<\s*(memory-context|memory-recall)\s*>.*?<\s*/\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)


def query(prompt: str, previous: str) -> str:
    """The previous assistant text, a blank line and the prompt without the hooks' complete
    memory fences, cut to 2000 characters.

    A slash command, an empty prompt, or a prompt under 20 characters with no previous
    assistant text gives "": nothing to recall.
    """
    text = _FENCED.sub(" ", prompt).strip()
    if not text or text.startswith("/") or (not previous and len(text) < MIN_QUERY_CHARS):
        return ""
    text = text[:QUERY_CHARS]
    return f"{previous}\n\n{text}" if previous else text


def search(post, text: str, scope: str | None, cutoff: float) -> tuple[list, list, list]:
    """The (notes, decisions, errors) of one /recall that arrived by `cutoff` (monotonic).

    A failed or late call gives ([], [], ["unavailable"]).
    """
    body = {
        "query": text,
        "notes": {"top_k": ASK, "min_score": NOTES_FLOOR},
        "decisions": {"top_k": ASK, "min_score": DECISIONS_FLOOR},
    }
    if scope:
        body["scope"] = scope
    result: list[tuple[list, list, list]] = []

    def ask() -> None:
        try:
            reply = post("/recall", body, timeout=max(cutoff - time.monotonic(), 0.1))
            result.append(
                (
                    [hit for hit in reply["notes"] if isinstance(hit, dict)],
                    [hit for hit in reply["decisions"] if isinstance(hit, dict)],
                    [str(name) for name in reply["errors"]],
                )
            )
        except Exception:
            pass

    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    thread.join(max(cutoff - time.monotonic(), 0))
    return result[0] if result and not thread.is_alive() else ([], [], ["unavailable"])
