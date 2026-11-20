"""Claude Code UserPromptSubmit hook: recall memory-base decisions and notes for the prompt.

Prints `hookSpecificOutput.additionalContext` holding one <memory-recall> fence of the
decisions and notes this session has not seen, or nothing. The query is the previous
assistant text and the prompt. One deadline from process start bounds HTTP and
rendering; a stalled socket cannot outlive it. Every failure mode is fail-open: exit 0.
Stdlib only.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

from hook_common import Config
from hook_common import git_origin
from hook_common import post
from hook_common import repo_scope
from recall import dedup
from recall import log
from recall import render
from recall import search
from recall import tail
from recall.choose import choose

START = time.monotonic()
CUTOFF = 2.5
DEADLINE = 3.0


def run(payload: dict, send, started: float, remote_of=git_origin) -> tuple[str, list[str]]:
    """The context to inject for this prompt and the keys it shows, via `send(path, body, timeout)`.

    The caller records the keys once the context is printed.
    """
    session = payload.get("session_id") or ""
    previous = tail.previous(session, payload.get("transcript_path") or "")
    text = search.query(payload.get("prompt") or "", previous)
    if not text:
        return "", []
    cwd = payload.get("cwd") or ""
    scope = repo_scope(remote_of(cwd)) if cwd else None
    notes, decisions, errors = search.search(send, text, scope, started + CUTOFF)
    decisions, notes = choose(decisions, notes, scope, dedup.load(session))
    context, keys = render.render(decisions, notes, scope)
    log.write(started, keys, errors)
    return context, keys


def main() -> None:
    # The deadline exits unconditionally: no lock, no log, nothing that could block it.
    timer = threading.Timer(max(START + DEADLINE - time.monotonic(), 0), os._exit, (0,))
    timer.daemon = True
    timer.start()
    session, context, keys = "", "", []
    try:
        payload = json.load(sys.stdin)
        session = payload.get("session_id") or ""
        cfg = Config.from_env()
        if cfg.configured:
            context, keys = run(
                payload, lambda path, body, timeout: post(cfg, path, body, timeout), START
            )
    except Exception:
        pass
    try:
        if context:
            output = {"hookEventName": "UserPromptSubmit", "additionalContext": context}
            sys.stdout.write(json.dumps({"hookSpecificOutput": output}))
            sys.stdout.flush()
            dedup.add(session, keys)
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
