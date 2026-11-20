"""One transcript line -> the session events of its blocks, in block order."""

from __future__ import annotations

import json
import uuid

TOOL_CUT = 2000
TEXT_CUT = 64000
NAME_CUT = 200
SKIP_FLAGS = ("isMeta", "isSidechain", "isCompactSummary")


def cut(text: str, limit: int) -> str:
    """The text, or its head plus a ` [cut, N chars]` marker within `limit` chars."""
    if len(text) <= limit:
        return text
    marker = f" [cut, {len(text)} chars]"
    return text[: limit - len(marker)] + marker


def _str(value) -> str:
    """A string as is, None as "", anything else as JSON."""
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _name(value) -> str | None:
    return None if value is None else cut(_str(value), NAME_CUT)


def _flat(content) -> str:
    if not isinstance(content, list):
        return _str(content)
    return "\n".join(
        (_str(b.get("text")) if b.get("type") == "text" else f"[{b.get('type')}]")
        if isinstance(b, dict)
        else _str(b)
        for b in content
    )


def _block(kind: str, block: dict):
    """(role, content, tool_name, tool_use_id) of a block, or None for a skipped block."""
    btype = block.get("type")
    if btype == "text":
        return (kind, cut(_str(block.get("text")), TEXT_CUT), None, None)
    if btype == "tool_use":
        given = block.get("input")
        text = _str(given) if given else ""
        return ("tool_call", cut(text, TOOL_CUT), _name(block.get("name")), _name(block.get("id")))
    if btype == "tool_result":
        content = cut(_flat(block.get("content")), TOOL_CUT)
        return ("tool_result", content, None, _name(block.get("tool_use_id")))
    return None


def events(line: bytes, session_id: str, where: dict) -> list[dict]:
    """The contract events of one main-transcript line; `where` holds repo, host and cwd.

    An entry's own `cwd` replaces the given one.
    """
    try:
        entry = json.loads(line)
        kind, uid, at = entry["type"], entry["uuid"], entry["timestamp"]
        content = entry["message"]["content"]
    except (ValueError, KeyError, TypeError, RecursionError):
        return []
    if kind not in ("user", "assistant") or any(entry.get(flag) for flag in SKIP_FLAGS):
        return []
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(blocks, list):
        return []
    cwd = entry.get("cwd")
    where = {**where, "cwd": cwd} if isinstance(cwd, str) and cwd else where
    out = []
    for index, block in enumerate(blocks):
        mapped = _block(kind, block) if isinstance(block, dict) else None
        if mapped is None:
            continue
        role, text, tool_name, tool_use_id = mapped
        out.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{uid}:{index}")),
                "session_id": session_id,
                "role": role,
                "content": text,
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "event_at": at,
                **where,
            }
        )
    return out
