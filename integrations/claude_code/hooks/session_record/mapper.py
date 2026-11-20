"""One transcript line -> the session events of its blocks, in block order."""

from __future__ import annotations

import json
import uuid

TOOL_CUT = 2000
TEXT_CUT = 64000
SKIP_FLAGS = ("isMeta", "isSidechain", "isCompactSummary")


def cut(text: str, limit: int) -> str:
    """The text, or its head plus a ` [cut, N chars]` marker within `limit` chars."""
    if len(text) <= limit:
        return text
    marker = f" [cut, {len(text)} chars]"
    return text[: limit - len(marker)] + marker


def _flat(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if b.get("type") == "text" else f"[{b.get('type')}]" for b in content
        )
    return content or ""


def _block(kind: str, block: dict):
    """(role, content, tool_name, tool_use_id) of a block, or None for a skipped block."""
    btype = block.get("type")
    if btype == "text":
        return (kind, cut(block.get("text") or "", TEXT_CUT), None, None)
    if btype == "tool_use":
        given = block.get("input")
        text = json.dumps(given, ensure_ascii=False) if given else ""
        return ("tool_call", cut(text, TOOL_CUT), block.get("name"), block.get("id"))
    if btype == "tool_result":
        return (
            "tool_result",
            cut(_flat(block.get("content")), TOOL_CUT),
            None,
            block.get("tool_use_id"),
        )
    return None


def events(line: bytes, session_id: str, where: dict) -> list[dict]:
    """The contract events of one main-transcript line; `where` holds repo, host and cwd."""
    try:
        entry = json.loads(line)
        kind, uid, at = entry["type"], entry["uuid"], entry["timestamp"]
        content = entry["message"]["content"]
    except (ValueError, KeyError, TypeError):
        return []
    if kind not in ("user", "assistant") or any(entry.get(flag) for flag in SKIP_FLAGS):
        return []
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(blocks, list):
        return []
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
