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
from datetime import datetime
from datetime import timezone

from hook_common import Config
from hook_common import data_dir
from hook_common import git_origin
from hook_common import post
from hook_common import repo_scope
from recall import dedup
from recall import render
from recall import search
from recall import tail

START = time.monotonic()
CUTOFF = 2.5
DEADLINE = 3.0
LOG = "recall.log"
MAX_DECISIONS = 2
MAX_NOTES = 3


def run(payload: dict, post, started: float, remote_of=git_origin) -> str:
    """The context to inject for this prompt via `post(path, body, timeout)`, or ""."""
    session = payload.get("session_id") or ""
    previous = tail.previous(session, payload.get("transcript_path") or "")
    text = search.query(payload.get("prompt") or "", previous)
    if not text:
        return ""
    cwd = payload.get("cwd") or ""
    scope = repo_scope(remote_of(cwd)) if cwd else None
    notes, decisions, errors = search.search(post, text, scope, started + CUTOFF)
    seen = dedup.load(session)
    decisions = [d for d in decisions if dedup.decision_key(scope, d.get("number")) not in seen]
    notes = [n for n in notes if str(n.get("id")) not in seen]
    context, keys = render.render(decisions[:MAX_DECISIONS], notes[:MAX_NOTES], scope)
    dedup.add(session, keys)
    _log(started, keys, errors)
    return context


def _log(started: float, keys: list[str], errors: list[str]) -> None:
    """Append one line for this call; it never holds the prompt, and a failed write is dropped."""
    elapsed = int((time.monotonic() - started) * 1000)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with open(data_dir() / LOG, "a") as handle:
            handle.write(
                f"{stamp} ms={elapsed} emitted={','.join(keys)} errors={','.join(errors)}\n"
            )
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
