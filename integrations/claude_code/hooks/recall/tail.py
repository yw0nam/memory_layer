"""The assistant text before a prompt: the Stop hook's tail file, else the transcript's tail."""

from __future__ import annotations

import json
import os

from hook_common import session_file

TAIL_CHARS = 1500
SCAN_BYTES = 256 * 1024


def save(session_id: str, text: str) -> None:
    """Record the last TAIL_CHARS of the assistant's message by atomic replace; a failure is dropped."""
    try:
        path = session_file(session_id, ".tail")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text[-TAIL_CHARS:])
        os.replace(tmp, path)
    except (OSError, TypeError):
        pass


def previous(session_id: str, transcript_path: str) -> str:
    """The previous assistant text (at most TAIL_CHARS), or "" when neither source has it."""
    try:
        text = session_file(session_id, ".tail").read_text()
    except (OSError, ValueError):
        text = ""
    return text or _from_transcript(transcript_path)


def _from_transcript(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(size - SCAN_BYTES, 0))
            lines = handle.read().split(b"\n")
    except OSError:
        return ""
    if size > SCAN_BYTES:
        del lines[0]  # starts mid-line
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            if entry.get("isMeta") or entry.get("isSidechain") or entry.get("type") != "assistant":
                continue
            blocks = entry["message"]["content"]
            text = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        if text:
            return text[-TAIL_CHARS:]
    return ""
