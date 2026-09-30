"""Claude Code SessionEnd hook: upload the session's user and assistant turns to memory-base.

Reads the hook payload on stdin, parses the session transcript, and posts its
text turns to `POST /conversations`, where the server stores them unembedded as
the evidence a note can link to.
Every failure mode is fail-open: exit 0, one log row, never a blocked session
exit. Stdlib only — the script runs under whatever python3 Claude Code invokes,
outside any venv.

Install next to the prefetch hook (identical on every machine):
    cp integrations/claude_code/capture_hook.py ~/.claude/hooks/memory_base_capture.py
    # ~/.claude/settings.json, beside the UserPromptSubmit entry:
    {"hooks": {"SessionEnd": [{"hooks": [{"type": "command",
        "command": "python3 \"$HOME/.claude/hooks/memory_base_capture.py\"",
        "timeout": 30}]}]}}

A turn is the text of one user entry, or of consecutive assistant entries that
carry the same `message.id` (one model message split across entries). Tool use,
tool results, images, thinking, meta entries, compaction summaries, and harness
noise (slash-command echoes, local command output, system reminders, task
notifications, interruption markers) never become turns. Because turn boundaries
depend only on the entries themselves, a resumed session's upload starts with the
turns of the earlier one.

Config via environment, every var optional: MEMORY_BASE_URL (default
http://127.0.0.1:8010), MEMORY_BASE_API_KEY, MEMORY_BASE_ENV (env file holding
the key; default ~/.config/memory-base/env), MEMORY_BASE_CAPTURE_NAMESPACE
(default dev). Each invocation appends one row to MEMORY_CAPTURE_LOG (default
~/.claude/memory_capture_log.jsonl).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

HTTP_TIMEOUT_SECONDS = 20.0
MIN_TURNS = 2
NOISE_PREFIXES = (
    "<command-name>",
    "<command-message>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<local-command-caveat>",
    "<system-reminder>",
    "<task-notification>",
    "[Request interrupted",
)


def _epoch(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _entry_text(entry: dict) -> str:
    """The entry's text blocks joined; user blocks that are harness noise are dropped."""
    content = (entry.get("message") or {}).get("content")
    blocks = [content] if isinstance(content, str) else []
    if isinstance(content, list):
        blocks = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
    if entry.get("type") == "user":
        blocks = [block for block in blocks if not block.lstrip().startswith(NOISE_PREFIXES)]
    return "\n\n".join(block.strip() for block in blocks if block.strip())


def transcript_turns(lines: Iterable[str]) -> tuple[list[dict], float | None, float | None]:
    """The transcript's text turns with the first and last kept entries' timestamps."""
    turns: list[dict] = []
    last_message_id = None
    started = ended = None
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("type") not in ("user", "assistant"):
            continue
        if entry.get("isMeta") or entry.get("isCompactSummary"):
            continue
        text = _entry_text(entry)
        if not text:
            continue
        role = entry["type"]
        message_id = (entry.get("message") or {}).get("id")
        if (
            role == "assistant"
            and turns
            and turns[-1]["role"] == "assistant"
            and (message_id is not None and message_id == last_message_id)
        ):
            turns[-1]["text"] += "\n\n" + text
        else:
            turns.append({"role": role, "text": text})
        last_message_id = message_id if role == "assistant" else None
        stamp = _epoch(entry.get("timestamp"))
        if stamp is not None:
            started = stamp if started is None else started
            ended = stamp
    return turns, started, ended


def repo_name(cwd: str) -> str:
    """The basename of the git toplevel containing cwd, else of cwd itself."""
    try:
        top = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        if top:
            return Path(top).name
    except Exception:
        pass
    return Path(cwd).name


def build_body(payload: dict, turns: list[dict], started, ended, namespace: str) -> dict:
    now = time.time()
    cwd = payload.get("cwd") or ""
    return {
        "origin": "claude_code",
        "external_session_id": payload["session_id"],
        "namespace": namespace,
        "started_at": started if started is not None else now,
        "ended_at": ended if ended is not None else now,
        "turns": turns,
        "metadata": {"repo": repo_name(cwd), "cwd": cwd} if cwd else {},
    }


def _log(log_path: Path, row: dict) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run_hook(payload: dict, post: Callable[[dict], dict], namespace: str) -> dict:
    """Parse the transcript and upload it via `post`; returns the log row."""
    row = {
        "ts": time.time(),
        "session_id": payload.get("session_id"),
        "reason": payload.get("reason"),
    }
    with open(payload["transcript_path"], encoding="utf-8") as f:
        turns, started, ended = transcript_turns(f)
    row["turns"] = len(turns)
    if len(turns) < MIN_TURNS:
        row["decision"] = "skipped"
        return row
    reply = post(build_body(payload, turns, started, ended, namespace))
    row.update(decision="uploaded", conversation_id=reply.get("id"))
    return row


def _resolve_api_key() -> str:
    key = os.environ.get("MEMORY_BASE_API_KEY", "")
    if key:
        return key
    env_file = os.environ.get("MEMORY_BASE_ENV", "") or str(
        Path.home() / ".config" / "memory-base" / "env"
    )
    try:
        for line in Path(env_file).read_text().splitlines():
            if line.startswith("MEMORY_BASE_API_KEY="):
                return line.split("=", 1)[1].strip()
    except Exception:
        pass
    return ""


def _post_to_server(url: str, api_key: str) -> Callable[[dict], dict]:
    def post(body: dict) -> dict:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/conversations",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": api_key},
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read())

    return post


def main() -> int:
    log_path = Path(
        os.environ.get("MEMORY_CAPTURE_LOG", Path.home() / ".claude" / "memory_capture_log.jsonl")
    )
    row: dict = {"ts": time.time()}
    try:
        payload = json.load(sys.stdin)
        row["session_id"] = payload.get("session_id")
        api_key = _resolve_api_key()
        if not api_key:
            row["decision"] = "no_api_key"
        else:
            url = os.environ.get("MEMORY_BASE_URL", "http://127.0.0.1:8010")
            namespace = os.environ.get("MEMORY_BASE_CAPTURE_NAMESPACE") or "dev"
            row = run_hook(payload, _post_to_server(url, api_key), namespace)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        row.update(decision="error", error=f"HTTP {exc.code}: {detail}"[:300])
    except Exception as exc:
        row.update(decision="error", error=f"{type(exc).__name__}: {exc}"[:300])
    _log(log_path, row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
