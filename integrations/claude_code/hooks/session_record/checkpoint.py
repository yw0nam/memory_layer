"""The acknowledged offset of one session's transcript, in `data_dir()/records/<session>.json`.

A checkpoint holds the transcript path, session id and cwd, the file's dev and inode, the
acknowledged offset, and the start and sha256 of the last acknowledged line.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from hook_common import data_dir
from hook_common import session_file


def records_dir() -> Path:
    path = data_dir() / "records"
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_of(session_id: str, suffix: str) -> Path:
    """The session's file with `suffix` in the records directory."""
    return records_dir() / session_file(session_id, suffix).name


def load(session_id: str) -> dict | None:
    try:
        return json.loads(file_of(session_id, ".json").read_text())
    except (OSError, ValueError):
        return None


def save(point: dict) -> None:
    """Write the checkpoint by atomic replace."""
    path = file_of(point["session_id"], ".json")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(point))
    os.replace(tmp, path)


def pending(path: str, session_id: str, cwd: str) -> dict:
    """The session's checkpoint; a new one is written at offset 0 first."""
    point = load(session_id)
    if point is None:
        point = {"path": path, "session_id": session_id, "cwd": cwd, "offset": 0}
        save(point)
    return point


def digest(line: bytes) -> str:
    return hashlib.sha256(line).hexdigest()


def start(point: dict, handle) -> int:
    """Where to read the open transcript from: the offset, or 0 when the file is not the one acknowledged."""
    offset = point.get("offset", 0)
    stat = os.fstat(handle.fileno())
    if not offset or (stat.st_dev, stat.st_ino) != (point.get("dev"), point.get("inode")):
        return 0
    if stat.st_size < offset:
        return 0
    handle.seek(point["line_start"])
    if digest(handle.read(offset - point["line_start"])) != point.get("sha256"):
        return 0
    return offset


def advance(point: dict, handle, line_start: int, line: bytes) -> None:
    """Record `line`, read from `line_start` of the open transcript, as the last acknowledged one."""
    stat = os.fstat(handle.fileno())
    point.update(
        dev=stat.st_dev,
        inode=stat.st_ino,
        offset=line_start + len(line),
        line_start=line_start,
        sha256=digest(line),
    )
    save(point)
