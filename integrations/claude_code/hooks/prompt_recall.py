"""Claude Code UserPromptSubmit hook: recall memory-base notes for the prompt.

Prints `hookSpecificOutput.additionalContext` holding one <memory-recall> fence of the notes
this session has not seen, or nothing. One deadline from process start bounds HTTP and
rendering; a stalled socket cannot outlive it. Every failure mode is fail-open: exit 0.
Stdlib only.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from datetime import timezone

from hook_common import Config
from hook_common import data_dir
from hook_common import post
from recall import dedup
from recall import render
from recall import search

START = time.monotonic()
CUTOFF = 2.5
DEADLINE = 3.0
LOG = "recall.log"


def run(payload: dict, post, started: float) -> str:
    """The context to inject for this prompt via `post(path, body, timeout)`, or ""."""
    text = search.query(payload.get("prompt") or "")
    if not text:
        return ""
    session = payload.get("session_id") or ""
    seen = dedup.load(session)
    hits = search.search(post, text, started + CUTOFF)
    context, ids = render.render([hit for hit in hits if str(hit.get("id")) not in seen])
    dedup.add(session, ids)
    _log(started, ids)
    return context


def _log(started: float, ids: list[str]) -> None:
    """Append one line for this call; it never holds the prompt, and a failed write is dropped."""
    elapsed = int((time.monotonic() - started) * 1000)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with open(data_dir() / LOG, "a") as handle:
            handle.write(f"{stamp} ms={elapsed} emitted={','.join(ids)}\n")
    except OSError:
        pass


def main() -> None:
    # The deadline exits unconditionally: no lock, no log, nothing that could block it.
    timer = threading.Timer(max(START + DEADLINE - time.monotonic(), 0), os._exit, (0,))
    timer.daemon = True
    timer.start()
    context = ""
    try:
        payload = json.load(sys.stdin)
        cfg = Config.from_env()
        if cfg.configured:
            context = run(
                payload, lambda path, body, timeout: post(cfg, path, body, timeout), START
            )
    except Exception:
        pass
    try:
        if context:
            output = {"hookEventName": "UserPromptSubmit", "additionalContext": context}
            sys.stdout.write(json.dumps({"hookSpecificOutput": output}))
            sys.stdout.flush()
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
