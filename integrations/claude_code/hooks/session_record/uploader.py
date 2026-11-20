"""Post a session's new transcript lines to /session_events, in order, from its checkpoint.

`python3 -m session_record.uploader <transcript_path> <session_id> <cwd>` uploads one session;
`--sweep` uploads every session whose checkpoint is behind its file. One uploader runs per
session: a second one exits at once. Only complete lines are read, one line's events go in
one chunk, and the checkpoint advances after each acknowledged chunk; the first failure
stops the run. `records.log` gets one line per chunk, never its content. Stdlib only.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import sys
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
    """(events, line_start, last line) per chunk of the complete lines from `offset`."""
    events, size, last = [], 0, None
    while True:
        line = handle.readline()
        if not line.endswith(b"\n"):
            break
        new = map_line(line)
        grown = sum(len(json.dumps(event)) + 2 for event in new)
        if events and (len(events) + len(new) > MAX_EVENTS or size + grown > MAX_BYTES):
            yield events, *last
            events, size = [], 0
        events, size, last = events + new, size + grown, (offset, line)
        offset += len(line)
    if last is not None:
        yield events, *last


def _drain(point: dict, send) -> int | None:
    """Upload the complete lines after the checkpoint; the bytes read, or None on a failure."""
    session_id = point["session_id"]
    where = {
        "repo": repo_scope(git_origin(point["cwd"])),
        "host": socket.gethostname(),
        "cwd": point["cwd"],
    }
    with open(point["path"], "rb") as handle:
        offset = checkpoint.start(point, handle)
        handle.seek(offset)
        for events, line_start, line in _chunks(
            handle, offset, lambda line: mapper.events(line, session_id, where)
        ):
            if events:
                try:
                    reply = send({"events": events})
                except Exception as exc:
                    _log(session_id, f"sent={len(events)} error={type(exc).__name__}")
                    return None
                conflicts = ",".join(reply.get("conflicts") or [])
                _log(
                    session_id,
                    f"sent={len(events)} stored={reply.get('stored')} known={reply.get('known')}"
                    f" conflicts={conflicts}",
                )
            checkpoint.advance(point, handle, line_start, line)
        return handle.tell()


def upload(path: str, session_id: str, cwd: str, send) -> None:
    """Upload the session's new lines through `send(body) -> reply` unless another uploader holds it."""
    point = checkpoint.pending(path, session_id, cwd)
    point.update(path=path, cwd=cwd)
    while True:
        with open(checkpoint.file_of(session_id, ".lock"), "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return
            try:
                read = _drain(point, send)
            except OSError:
                return
        # A writer refused while the lock was held left its lines to this run.
        if read is None or os.path.getsize(path) <= read:
            return
        point = checkpoint.load(session_id) or point


def sweep(send) -> None:
    """Upload every session whose checkpoint offset is behind its file."""
    for item in sorted(checkpoint.records_dir().glob("*.json")):
        try:
            point = json.loads(item.read_text())
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
