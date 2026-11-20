"""scripts/backfill_sessions.py against a fake REST layer and transcripts on disk.

No test here reaches a memory-base server: the REST client is a fake inside the process.
"""

from __future__ import annotations

import json
import os
import urllib.error

import pytest

import backfill_sessions as bs

BASE = "http://memory.test"


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_BASE_API_KEY", "test-key")
    monkeypatch.setenv("MEMORY_BASE_AUTHOR", "claude-code")
    # The script overwrites this inside main(); the fixture value is restored afterwards.
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "unused"))


@pytest.fixture
def rest(monkeypatch):
    return FakeRest(monkeypatch)


class FakeRest:
    """Stores events by id as the contract says; `statuses` answers the next posts with
    those HTTP errors; every call is recorded."""

    def __init__(self, monkeypatch):
        self.rows, self.posts, self.statuses = {}, [], []
        monkeypatch.setattr(bs, "post", self)

    def __call__(self, cfg, path, body, timeout):
        self.posts.append(body)
        if self.statuses:
            raise urllib.error.HTTPError(path, self.statuses.pop(0), "no", {}, None)
        stored = known = 0
        for event in body["events"]:
            old = self.rows.get(event["id"])
            if old is None:
                self.rows[event["id"]] = event
                stored += 1
            elif old == event:
                known += 1
        return {"stored": stored, "known": known, "conflicts": []}

    def sent(self):
        return [event["content"] for body in self.posts for event in body["events"]]


def _line(uid, text, cwd=None, kind="assistant"):
    entry = {
        "type": kind,
        "uuid": uid,
        "timestamp": "2026-11-20T10:00:00Z",
        "message": {"role": kind, "content": [{"type": "text", "text": text}]},
    }
    if cwd is not None:
        entry["cwd"] = cwd
    return (json.dumps(entry) + "\n").encode()


def _write(path, *lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(lines))


def _run(projects, tmp_path, rest, *extra, capsys=None):
    argv = [
        "--projects",
        str(projects),
        "--url",
        BASE,
        "--state",
        str(tmp_path / "state"),
        *extra,
    ]
    code = bs.main(argv)
    captured = capsys.readouterr() if capsys else None
    return code, (captured.out if captured else ""), (captured.err if captured else "")


def _set_mtime(path, offset):
    stamp = 1_700_000_000 + offset
    os.utime(path, (stamp, stamp))


# ---- discovery ----------------------------------------------------------------


def test_discovery_takes_only_project_dir_jsonl_files(env, tmp_path, rest, capsys):
    projects = tmp_path / "projects"
    _write(projects / "alpha" / "main.jsonl", _line("u1", "kept", cwd="/repo/one"))
    _write(projects / "alpha" / "subagents" / "side.jsonl", _line("u1", "sidechain"))
    _write(projects / "alpha" / "nested" / "deep.jsonl", _line("u1", "deeper"))
    _write(projects / "loose.jsonl", _line("u1", "no project dir"))

    code, out, _ = _run(projects, tmp_path, rest, capsys=capsys)

    assert code == 0
    assert rest.sent() == ["kept"]
    assert "side.jsonl" not in out and "deep.jsonl" not in out and "loose" not in out


def test_transcripts_run_in_ascending_mtime_order(env, tmp_path, rest, capsys):
    projects = tmp_path / "projects"
    _write(projects / "alpha" / "newer.jsonl", _line("u1", "newer text", cwd="/repo/one"))
    _write(projects / "beta" / "older.jsonl", _line("u1", "older text", cwd="/repo/one"))
    _set_mtime(projects / "alpha" / "newer.jsonl", 600)
    _set_mtime(projects / "beta" / "older.jsonl", 0)

    code, out, _ = _run(projects, tmp_path, rest, capsys=capsys)

    assert code == 0
    assert rest.sent() == ["older text", "newer text"]
    assert out.index("older") < out.index("newer")


# ---- cwd ----------------------------------------------------------------------


def test_an_entry_without_a_cwd_takes_the_first_one_the_transcript_has(env, tmp_path, rest, capsys):
    projects = tmp_path / "projects"
    _write(
        projects / "alpha" / "s1.jsonl",
        _line("u1", "before the cwd"),
        _line("u2", "with the cwd", cwd="/repo/one"),
        _line("u3", "another cwd", cwd="/repo/two"),
    )

    code, out, _ = _run(projects, tmp_path, rest, capsys=capsys)

    assert code == 0
    assert [(e["content"], e["cwd"]) for e in rest.rows.values()] == [
        ("before the cwd", "/repo/one"),
        ("with the cwd", "/repo/one"),
        ("another cwd", "/repo/two"),
    ]
    # /repo/one does not exist, so the repo scope is null.
    assert {event["repo"] for event in rest.rows.values()} == {None}


def test_a_transcript_without_any_cwd_is_skipped(env, tmp_path, rest, capsys):
    projects = tmp_path / "projects"
    _write(projects / "alpha" / "s1.jsonl", _line("u1", "kept", cwd="/repo/one"))
    _write(projects / "alpha" / "s2.jsonl", _line("u1", "no cwd anywhere"))

    code, out, _ = _run(projects, tmp_path, rest, capsys=capsys)

    assert code == 0
    assert rest.sent() == ["kept"]
    assert "s2 skipped (no cwd)" in out
    assert out.rstrip().endswith("skipped=1")


# ---- resume and dry run --------------------------------------------------------


def test_a_rerun_of_a_finished_transcript_sends_nothing(env, tmp_path, rest, capsys):
    projects = tmp_path / "projects"
    _write(
        projects / "alpha" / "s1.jsonl",
        _line("u1", "first", cwd="/repo/one"),
        _line("u2", "second", cwd="/repo/one"),
    )

    code, out, _ = _run(projects, tmp_path, rest, capsys=capsys)
    first_posts = len(rest.posts)
    code_again, out_again, _ = _run(projects, tmp_path, rest, capsys=capsys)

    assert code == code_again == 0
    assert len(rest.posts) == first_posts
    assert "s1 events=2 stored=2 known=0 rejected=0" in out
    assert "s1 events=0 stored=0 known=0 rejected=0" in out_again


def test_a_dry_run_posts_nothing_and_a_real_run_still_sends(env, tmp_path, rest, capsys):
    projects = tmp_path / "projects"
    _write(projects / "alpha" / "s1.jsonl", _line("u1", "kept", cwd="/repo/one"))

    code, out, _ = _run(projects, tmp_path, rest, "--dry-run", capsys=capsys)

    assert code == 0
    assert rest.posts == []
    assert "s1 events=1 stored=0 known=0 rejected=0" in out

    code_real, _, _ = _run(projects, tmp_path, rest, capsys=capsys)
    assert code_real == 0
    assert rest.sent() == ["kept"]


# ---- failure ------------------------------------------------------------------


def test_a_non_400_failure_stops_with_exit_1_and_the_file_name(env, tmp_path, rest, capsys):
    projects = tmp_path / "projects"
    _write(projects / "alpha" / "bad.jsonl", _line("u1", "fails", cwd="/repo/one"))
    _write(projects / "beta" / "next.jsonl", _line("u1", "never sent", cwd="/repo/one"))
    _set_mtime(projects / "alpha" / "bad.jsonl", 0)
    _set_mtime(projects / "beta" / "next.jsonl", 600)
    rest.statuses = [503]

    code, out, err = _run(projects, tmp_path, rest, capsys=capsys)

    assert code == 1
    assert "never sent" not in rest.sent()
    assert "bad.jsonl" in err


def test_a_line_rejected_with_400_is_counted_once_and_the_other_line_is_stored(
    env, tmp_path, rest, capsys
):
    projects = tmp_path / "projects"
    _write(
        projects / "alpha" / "s1.jsonl",
        _line("u1", "rejected", cwd="/repo/one"),
        _line("u2", "kept", cwd="/repo/one"),
    )
    rest.statuses = [400, 400]

    code, out, _ = _run(projects, tmp_path, rest, capsys=capsys)

    assert code == 0
    assert [row["content"] for row in rest.rows.values()] == ["kept"]
    assert "s1 events=2 stored=1 known=0 rejected=1" in out
