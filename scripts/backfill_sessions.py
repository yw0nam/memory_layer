#!/usr/bin/env python3
"""One-time backfill of local Claude Code transcripts into the session record.

Walks `<projects>/<project dir>/*.jsonl` — the main-session transcripts, never a
file under a `subagents/` folder or deeper — and for each one, in ascending
modification time, runs the plugin's session record uploader with a checkpoint
directory of its own, so a rerun resumes where it stopped and a finished
transcript sends nothing. The session id is the file stem; the cwd comes from
the first transcript entry that has one, and a transcript without any is
skipped.

Usage: backfill_sessions.py --url MEMORY_BASE_URL [--projects DIR]
       [--since YYYY-MM-DD] [--state DIR] [--dry-run]

Environment: MEMORY_BASE_API_KEY, MEMORY_BASE_AUTHOR. Prints one line per
transcript — the session id and counts only, never transcript content. Stops at
the first failure that is not a 400 rejection, with exit 1 and the file name.
Stdlib only; Python 3.9 compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
from datetime import datetime
from pathlib import Path

HOOKS = Path(__file__).resolve().parents[1] / "integrations" / "claude_code" / "hooks"
sys.path.insert(0, str(HOOKS))

from hook_common import Config  # noqa: E402
from hook_common import post  # noqa: E402
from session_record import uploader  # noqa: E402

DEFAULT_STATE = "~/.cache/memory-base-backfill"
SINCE_FORMAT = "%Y-%m-%d"


class Send:
    """The uploader's send: posts each chunk to /session_events, counting the outcome.

    A dry run counts the events without posting. A 400 rejection is counted and
    skipped as the uploader does; any other failure is remembered so the run
    can stop after the transcript.
    """

    def __init__(self, cfg: Config, dry: bool):
        self.cfg, self.dry = cfg, dry
        self.events = self.stored = self.known = self.rejected = 0
        self.failure: Exception | None = None

    def __call__(self, body: dict) -> dict:
        self.events += len(body["events"])
        if self.dry:
            return {"stored": 0, "known": 0, "conflicts": []}
        try:
            reply = post(self.cfg, "/session_events", body, uploader.POST_TIMEOUT)
            self.stored += reply.get("stored") or 0
            self.known += reply.get("known") or 0
            return reply
        except urllib.error.HTTPError as exc:
            if exc.code == 400:
                self.rejected += len(body["events"])
            else:
                self.failure = exc
            raise
        except Exception as exc:
            self.failure = exc
            raise


def _cwd_of(path: Path) -> str | None:
    """The cwd of the first transcript entry that has one, else None."""
    with open(path, "rb") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            cwd = entry.get("cwd") if isinstance(entry, dict) else None
            if isinstance(cwd, str) and cwd:
                return cwd
    return None


def _mtime(path: Path) -> float:
    return path.stat().st_mtime


def _transcripts(projects: Path, since: float | None) -> list[Path]:
    """The main-session transcripts ascending by modification time; `since` keeps the newer."""
    files = list(projects.glob("*/*.jsonl"))
    if since is not None:
        files = [path for path in files if _mtime(path) >= since]
    return sorted(files, key=_mtime)


def parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--projects", default="~/.claude/projects", help="the Claude Code projects directory"
    )
    parser.add_argument("--url", required=True, help="memory-base REST base URL")
    parser.add_argument(
        "--since", help="only transcripts modified on/after YYYY-MM-DD (local time)"
    )
    parser.add_argument(
        "--state", default=DEFAULT_STATE, help="this backfill's checkpoint directory"
    )
    parser.add_argument("--dry-run", action="store_true", help="count the events without sending")
    args = parser.parse_args(argv)
    if args.since:
        try:
            datetime.strptime(args.since, SINCE_FORMAT)
        except ValueError:
            parser.error("--since must be YYYY-MM-DD")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("MEMORY_BASE_API_KEY")
    author = os.environ.get("MEMORY_BASE_AUTHOR")
    if not api_key or not author:
        print("error: MEMORY_BASE_API_KEY and MEMORY_BASE_AUTHOR must be set", file=sys.stderr)
        return 2
    cfg = Config(url=args.url.rstrip("/"), api_key=api_key, author=author)
    since = datetime.strptime(args.since, SINCE_FORMAT).timestamp() if args.since else None
    if args.dry_run:
        # A dry run's checkpoints land in a throwaway directory, so a real run resends.
        os.environ["CLAUDE_PLUGIN_DATA"] = tempfile.mkdtemp(prefix="memory-base-backfill-dry-")
    else:
        os.environ["CLAUDE_PLUGIN_DATA"] = str(Path(args.state).expanduser())
    files = _transcripts(Path(args.projects).expanduser(), since)
    totals = dict(events=0, stored=0, known=0, rejected=0, files=0, skipped=0)
    for path in files:
        cwd = _cwd_of(path)
        if cwd is None:
            print(f"{path.stem} skipped (no cwd)")
            totals["skipped"] += 1
            continue
        send = Send(cfg, args.dry_run)
        uploader.upload(str(path), path.stem, cwd, send)
        if send.failure is not None:
            print(f"error: {path.name}: {send.failure}", file=sys.stderr)
            return 1
        print(
            f"{path.stem} events={send.events} stored={send.stored}"
            f" known={send.known} rejected={send.rejected}"
        )
        totals["files"] += 1
        for key in ("events", "stored", "known", "rejected"):
            totals[key] += getattr(send, key)
    verb = "would send" if args.dry_run else "sent"
    print(
        f"total {verb} events={totals['events']} stored={totals['stored']}"
        f" known={totals['known']} rejected={totals['rejected']}"
        f" files={totals['files']} skipped={totals['skipped']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
