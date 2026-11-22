"""Post a session's new transcript lines to /session_events, in order, from its checkpoint.

`python3 -m session_record.uploader <transcript_path> <session_id> <cwd>` uploads one session;
`--sweep` uploads every session whose checkpoint is behind its file. One uploader runs per
session: a second one exits at once. Only complete lines are read, one line's events go in
one chunk, and the checkpoint advances after each acknowledged chunk. A chunk the server
rejects with 400 is resent line by line, and a line that still gets 400 is skipped; any
other failure stops this session. `records.log` gets one line per post and per error with
counts, event ids and error classes, never content. Stdlib only.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import sys
import urllib.error
from datetime import datetime
from datetime import timezone

from hook_common import Config
from hook_common import data_dir
from hook_common import git_origin
from hook_common import post
from hook_common import repo_scope
from session_record import checkpoint
from session_record import mapper

MAX_EVENTS = 500
MAX_BYTES = 1_900_000
POST_TIMEOUT = 30.0
LOG = "records.log"


def _log(session_id: str, text: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with open(data_dir() / LOG, "a") as handle:
            handle.write(f"{stamp} session={session_id} {text}\n")
    except OSError:
        pass


def _chunks(handle, offset: int, map_line):
    """Per chunk, the (line_start, line, events) of its complete lines from `offset`."""
    lines, count, size = [], 0, 0
    while True:
        line = handle.readline()
        if not line.endswith(b"\n"):
            break
        new = map_line(line)
        grown = sum(len(json.dumps(event)) + 2 for event in new)
        if count and (count + len(new) > MAX_EVENTS or size + grown > MAX_BYTES):
            yield lines
            lines, count, size = [], 0, 0
        lines.append((offset, line, new))
        count, size = count + len(new), size + grown
        offset += len(line)
    if lines:
        yield lines


def _post(session_id: str, events: list[dict], send) -> bool:
    """Post the events and log the reply; False when the server rejects them with 400.

    Any other failure raises.
    """
    try:
        reply = send({"events": events})
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            return False
        raise
    conflicts = ",".join(reply.get("conflicts") or [])
    _log(
        session_id,
        f"sent={len(events)} stored={reply.get('stored')}"
        f" known={reply.get('known')} conflicts={conflicts}",
    )
    return True


def _drain(point: dict, send) -> int:
    """Upload the complete lines after the checkpoint; the bytes read. A failure raises.

    A chunk the server rejects with 400 is resent line by line, and a line that still gets
    400 is logged and skipped.
    """
    session_id = point["session_id"]
    where = {
        "repo": repo_scope(git_origin(point["cwd"])),
        "host": socket.gethostname(),
        "cwd": point["cwd"],
    }
    with open(point["path"], "rb") as handle:
        offset = checkpoint.start(point, handle)
        handle.seek(offset)
        for lines in _chunks(handle, offset, lambda line: mapper.events(line, session_id, where)):
            events = [event for _, _, found in lines for event in found]
            if not events or _post(session_id, events, send):
                checkpoint.advance(point, handle, *lines[-1][:2])
                continue
            for line_start, line, found in lines:
                if found and not _post(session_id, found, send):
                    first = found[0]["id"]
                    _log(session_id, f"rejected={len(found)} first_id={first} status=400")
                checkpoint.advance(point, handle, line_start, line)
        return handle.tell()


def upload(path: str, session_id: str, cwd: str, send) -> None:
    """Upload the session's new lines through `send(body) -> reply` unless another uploader
    holds it; any error stops this session only."""
    try:
        while True:
            with open(checkpoint.file_of(session_id, ".lock"), "a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    return
                point = checkpoint.pending(path, session_id, cwd)
                point.update(path=path, cwd=cwd)
                read = _drain(point, send)
            # A writer refused while the lock was held left its lines to this run.
            if os.path.getsize(path) <= read:
                return
    except Exception as exc:
        status = f" status={exc.code}" if isinstance(exc, urllib.error.HTTPError) else ""
        _log(session_id, f"error={type(exc).__name__}{status}")


def _forget(item) -> None:
    """Remove a checkpoint and its lock file unless an uploader holds the lock."""
    lock_path = item.with_suffix(".lock")
    with open(lock_path, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        item.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def sweep(send) -> None:
    """Upload every session whose checkpoint offset is behind its file; forget those whose
    file is gone."""
    for item in sorted(checkpoint.records_dir().glob("*.json")):
        try:
            point = json.loads(item.read_text())
            if not os.path.exists(point["path"]):
                _forget(item)
                continue
            behind = os.path.getsize(point["path"]) != point.get("offset", 0)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if behind:
            upload(point["path"], point["session_id"], point["cwd"], send)


def main(argv: list[str]) -> int:
    cfg = Config.from_env()
    if not cfg.configured:
        return 0

    def send(body: dict):
        return post(cfg, "/session_events", body, POST_TIMEOUT)

    if argv == ["--sweep"]:
        sweep(send)
    elif len(argv) == 3:
        upload(*argv, send)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
