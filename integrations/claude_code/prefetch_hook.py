"""Claude Code UserPromptSubmit hook: inject relevant memory-base notes.

Reads the hook payload on stdin, searches memory-base with the prompt, and
prints a <memory-context> block for fresh, relevant hits. Every failure mode
is fail-open: no output, exit 0, never a blocked prompt. Stdlib only — the
script runs under whatever python3 Claude Code invokes, outside any venv.

Install (identical on every machine):
    cp integrations/claude_code/prefetch_hook.py ~/.claude/hooks/memory_base_prefetch.py
    printf 'MEMORY_BASE_API_KEY=...\n' > ~/.config/memory-base/env   # chmod 600
    # ~/.claude/settings.json:
    {"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command",
        "command": "python3 \"$HOME/.claude/hooks/memory_base_prefetch.py\"",
        "timeout": 5}]}]}}

Config via environment, every var optional: MEMORY_BASE_URL (default
http://127.0.0.1:8010), MEMORY_BASE_API_KEY, MEMORY_BASE_ENV (env file
holding the key; default ~/.config/memory-base/env). Each invocation
appends one metrics row to MEMORY_PREFETCH_LOG (default
~/.claude/memory_prefetch_log.jsonl) for later evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

MIN_PROMPT_CHARS = 20
QUERY_LIMIT = 500
BLOCK_LIMIT = 1500
TOP_K = 3
MIN_SCORE = 0.6
HTTP_TIMEOUT_SECONDS = 1.5

HARNESS_PREFIXES = (
    "/",
    "!",
    "#",
    "<task-notification",
    "<command-",
    "<local-command",
    "<bash-",
    "<teammate-message",
    "<system-reminder",
    "Caveat:",
)


def is_trivial_prompt(prompt: str) -> bool:
    """True for prompts too low-signal to search with: short, commands, harness noise."""
    text = prompt.strip()
    if len(text) < MIN_PROMPT_CHARS:
        return True
    return text.startswith(HARNESS_PREFIXES)


def build_context_block(hits: list[dict]) -> str:
    """Format hits as a memory-context block, truncated at a line boundary."""
    header = (
        '<memory-context source="memory-base">\n'
        "[Auto-retrieved from long-term memory; may be irrelevant — ignore freely.]"
    )
    footer = "</memory-context>"
    lines: list[str] = []
    used = len(header) + len(footer) + 2
    for hit in hits:
        line = f"- [{hit.get('date', '?')}] {hit.get('text', '').strip()}"
        if used + len(line) + 1 > BLOCK_LIMIT:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return "\n".join([header, *lines, footer])


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()


def _state_file(state_dir: Path, session_id: str) -> Path:
    return state_dir / f"{_hash(session_id)}.json"


def _load_seen(state_dir: Path, session_id: str) -> set[str]:
    try:
        return set(json.loads(_state_file(state_dir, session_id).read_text()))
    except Exception:
        return set()


def _save_seen(state_dir: Path, session_id: str, seen: set[str]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    _state_file(state_dir, session_id).write_text(json.dumps(sorted(seen)))


def _log(log_path: Path, row: dict) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run_hook(payload: dict, fetch, state_dir: Path, log_path: Path) -> str:
    """Gate, search via `fetch`, dedup against the session, and format the block."""
    prompt = payload.get("prompt") or ""
    session_id = payload.get("session_id") or "unknown"
    started = time.monotonic()
    row = {
        "ts": time.time(),
        "session_id": session_id,
        "cwd": payload.get("cwd", ""),
        "query": prompt.strip()[:200],
    }

    if is_trivial_prompt(prompt):
        row["decision"] = "trivial"
        _log(log_path, row)
        return ""

    try:
        hits = fetch(prompt.strip()[:QUERY_LIMIT])
    except Exception as exc:
        row.update(decision="error", error=str(exc)[:200])
        row["latency_ms"] = int((time.monotonic() - started) * 1000)
        _log(log_path, row)
        return ""

    seen = _load_seen(state_dir, session_id)
    fresh = [h for h in hits if _hash(h.get("text", "")) not in seen]
    block = build_context_block(fresh)
    row["latency_ms"] = int((time.monotonic() - started) * 1000)
    row["hits"] = [
        {"score": round(h.get("score", 0), 3), "text": h.get("text", "")[:80]} for h in hits
    ]
    if block:
        _save_seen(state_dir, session_id, seen | {_hash(h.get("text", "")) for h in fresh})
        row["decision"] = "injected"
    else:
        row["decision"] = "empty"
    _log(log_path, row)
    return block


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


def _fetch_from_server(url: str, api_key: str):
    def fetch(query: str) -> list[dict]:
        body = json.dumps(
            {
                "query": query,
                "source": "memory",
                "top_k": TOP_K,
                "min_score": MIN_SCORE,
                "namespaces": ["default"],
            }
        ).encode()
        req = urllib.request.Request(
            f"{url.rstrip('/')}/search",
            data=body,
            headers={"Content-Type": "application/json", "X-API-Key": api_key},
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read())

    return fetch


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        api_key = _resolve_api_key()
        if not api_key:
            return 0
        url = os.environ.get("MEMORY_BASE_URL", "http://127.0.0.1:8010")
        state_dir = Path(tempfile.gettempdir()) / "cc_memory_prefetch"
        log_path = Path(
            os.environ.get(
                "MEMORY_PREFETCH_LOG",
                Path.home() / ".claude" / "memory_prefetch_log.jsonl",
            )
        )
        block = run_hook(payload, _fetch_from_server(url, api_key), state_dir, log_path)
        if block:
            print(block)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
