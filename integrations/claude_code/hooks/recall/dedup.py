"""Per-session dedup: the keys (note ids, decision keys) a session was already shown, in `data_dir()/sessions/`.

Best-effort: a failed write loses only the ids it carried. Writers of one session take a
short flock on its lock file and merge only their new ids, so overlapping hooks keep each
other's ids and a merge never brings back ids a reset removed.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

from hook_common import session_file


def _path(session_id: str) -> Path:
    return session_file(session_id, ".json")


def decision_key(scope: str, number: int) -> str:
    return f"decision:{scope}#{number}"


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def load(session_id: str) -> set[str]:
    try:
        return set(json.loads(_path(session_id).read_text()))
    except (OSError, ValueError, TypeError):
        return set()


def add(session_id: str, ids: list[str]) -> None:
    """Merge ids into the session's shown set."""
    if not ids:
        return
    path = _path(session_id)
    try:
        with _locked(path):
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(sorted(load(session_id) | set(ids))))
            os.replace(tmp, path)
    except OSError:
        pass


def reset(session_id: str) -> None:
    """Forget every key the session was shown and remove its lock file."""
    path = _path(session_id)
    try:
        with _locked(path):
            path.unlink(missing_ok=True)
        # shortcut: a writer racing this unlink may lock the removed file and lose its keys; harmless for a dedup hint.
        path.with_suffix(".lock").unlink(missing_ok=True)
    except OSError:
        pass
