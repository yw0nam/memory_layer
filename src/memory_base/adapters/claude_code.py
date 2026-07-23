"""Claude Code history adapter."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from memory_base.adapters.base import Burst, Message, Session, SourceAdapter, SourceFile


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_jsonl(data: str, default_session_id: str = "") -> list[Message]:
    """Parse supported Claude Code messages, skipping malformed/noise records."""
    messages: list[Message] = []
    latest_by_session: dict[str, Message] = {}
    for line in data.splitlines():
        try:
            item = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        role = item.get("type")
        if role not in ("user", "assistant") or item.get("isSidechain") is True:
            continue
        session_id = str(item.get("sessionId") or default_session_id)
        content = (item.get("message") or {}).get("content")
        texts: list[str] = []
        tool_names: list[str] = []
        tool_error = False
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text" and isinstance(block.get("text"), str):
                    texts.append(block["text"])
                elif role == "assistant" and block_type == "tool_use":
                    if isinstance(block.get("name"), str):
                        tool_names.append(block["name"])
                elif role == "user" and block_type == "tool_result":
                    tool_error = tool_error or block.get("is_error") is True
        text = "\n".join(part.strip() for part in texts if part.strip()).strip()
        if not text:
            if tool_error and session_id in latest_by_session:
                latest_by_session[session_id].tool_error = True
            continue
        message = Message(
            role=role,
            text=text,
            timestamp=_timestamp(item.get("timestamp")),
            session_id=session_id,
            cwd=item.get("cwd"),
            git_branch=item.get("gitBranch"),
            uuid=item.get("uuid"),
            parent_uuid=item.get("parentUuid"),
            tool_names=tuple(tool_names),
            tool_error=tool_error,
        )
        messages.append(message)
        latest_by_session[session_id] = message
    return messages


class ClaudeCodeAdapter(SourceAdapter):
    source_type = "claude_code"
    emit_bursts = False

    def discover(self) -> list[SourceFile]:
        files = []
        for path in (Path.home() / ".claude" / "projects").glob("*/*.jsonl"):
            stat = path.stat()
            files.append(SourceFile(path, stat.st_mtime, stat.st_size))
        files.sort(key=lambda file: file.mtime, reverse=True)
        return files

    def parse(self, text: str, file: SourceFile) -> list[Session]:
        from memory_base.ingest.history import group_sessions

        sessions = group_sessions(parse_jsonl(text, file.path.stem), file.mtime)
        for session in sessions:
            session._source_file = file
        return sessions

    def has_social(self, burst: Burst) -> bool:
        return burst.social_weight > 1.0
