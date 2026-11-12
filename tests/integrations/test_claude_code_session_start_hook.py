"""Unit tests for the Claude Code SessionStart hook — pure module, stdlib only.

Exercises session_start.py directly (importable via the ``pythonpath``
test config entry). HTTP is faked, git runs against throwaway repositories.
"""

from __future__ import annotations

import io
import json
import subprocess
import urllib.parse
import urllib.request

import pytest

import session_start
from memory_base.serve.messages.store import normalize_scope
from hook_common import git_origin
from hook_common import repo_scope
from session_start import HANDOFF_HEADER
from session_start import run_hook as _run_hook

HANDOFFS = [
    {
        "id": "6f1c1a52-0000-4000-8000-000000000002",
        "subject": "Session entry hooks",
        "status": "in_progress",
        "created_at": "2026-09-28T09:12:00+00:00",
    },
    {
        "id": "6f1c1a52-0000-4000-8000-000000000001",
        "subject": "Handoff retention",
        "status": "blocked",
        "created_at": "2026-09-20T17:40:00+00:00",
    },
]


def run_hook(payload, get, remote_of=git_origin):
    return _run_hook(payload, get, remote_of)


def _payload(cwd="/repo"):
    return {"session_id": "sess-1", "cwd": cwd, "source": "startup"}


HANDOFF_LINES = [
    HANDOFF_HEADER,
    "- 2026-09-28  Session entry hooks  "
    "(status: in_progress, id: 6f1c1a52-0000-4000-8000-000000000002)",
    "- 2026-09-20  Handoff retention  (status: blocked, id: 6f1c1a52-0000-4000-8000-000000000001)",
]


class RecordingGet:
    """Fake HTTP GET that records every (path, params) and answers /messages."""

    def __init__(self, handoffs=()):
        self.handoffs = list(handoffs)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, path, params):
        self.calls.append((path, dict(params)))
        if path != "/messages":
            raise AssertionError(f"unexpected request {path}")
        return self.handoffs

    def paths(self):
        return [path for path, _ in self.calls]


def _origin(url):
    return lambda cwd: url


# ---- repo_scope ----------------------------------------------------------------


@pytest.mark.parametrize(
    "origin",
    [
        "https://github.com/yw0nam/memory_base.git",
        "git@github.com:yw0nam/memory_base.git",
    ],
)
def test_https_and_scp_remotes_derive_the_server_scope(origin):
    expected = normalize_scope(f"repo:{origin}")
    assert expected == "repo:github.com/yw0nam/memory_base"
    assert repo_scope(origin) == expected


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("https://GitHub.com/O/r/", "repo:github.com/O/r"),
        ("ssh://git@github.com/o/r.git", "repo:github.com/o/r"),
        ("https://user:token@github.com/o/r.git", "repo:github.com/o/r"),
        ("https://github.com:443/o/r.git", "repo:github.com/o/r"),
    ],
)
def test_other_remote_forms_normalize_onto_a_scope_the_server_accepts(origin, expected):
    assert repo_scope(origin) == expected
    assert normalize_scope(expected) == expected


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "/srv/git/r.git",
        "file:///srv/git/r.git",
        "https://localhost/o/r.git",
        "git@192.168.0.2:o/r.git",
        "git@gitserver:o/r.git",
        "https://github.com/",
    ],
)
def test_a_remote_that_is_not_portable_has_no_scope(origin):
    assert repo_scope(origin) is None


# ---- git_origin ----------------------------------------------------------------


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_git_origin_reads_the_origin_remote(tmp_path):
    _git("init", "-q", cwd=tmp_path)
    _git("remote", "add", "origin", "git@github.com:o/r.git", cwd=tmp_path)
    (tmp_path / "sub").mkdir()
    assert git_origin(str(tmp_path / "sub")) == "git@github.com:o/r.git"


def test_git_origin_is_none_without_a_remote(tmp_path):
    _git("init", "-q", cwd=tmp_path)
    assert git_origin(str(tmp_path)) is None


def test_git_origin_is_none_outside_a_repository(tmp_path):
    assert git_origin(str(tmp_path)) is None


# ---- run_hook ------------------------------------------------------------------

REPO = _origin("git@github.com:o/r.git")


def _lines(block):
    assert block.startswith("<memory-context>\n") and block.endswith("\n</memory-context>")
    return block.splitlines()[1:-1]


def test_header_says_nothing_is_claimed():
    assert "Nothing is claimed" in HANDOFF_HEADER


def test_the_fence_holds_only_this_repos_handoffs():
    get = RecordingGet(HANDOFFS)
    block = run_hook(_payload(), get, _origin("https://github.com/o/r.git"))
    assert _lines(block) == HANDOFF_LINES
    assert block.count("<memory-context>") == 1
    assert "Session:" not in block
    assert get.calls == [
        ("/messages", {"purpose": "handoff", "scope": "repo:github.com/o/r", "limit": "100"})
    ]


def test_no_handoffs_print_nothing():
    assert run_hook(_payload(), RecordingGet(), REPO) == ""


def test_no_handoff_is_asked_when_the_scope_cannot_be_derived():
    for origin in (None, "/srv/git/r.git"):
        get = RecordingGet(HANDOFFS)
        assert run_hook(_payload(), get, _origin(origin)) == ""
        assert get.calls == []


def test_no_handoff_is_asked_without_a_cwd():
    get = RecordingGet(HANDOFFS)
    assert run_hook({"session_id": "sess-1"}, get, REPO) == ""
    assert get.calls == []


def test_a_failed_handoff_fetch_prints_nothing():
    def get(path, params):
        raise OSError("connection refused")

    assert run_hook(_payload(), get, REPO) == ""


def test_prints_at_most_ten_handoffs():
    rows = [dict(HANDOFFS[0], id=f"id-{i}", subject=f"work {i}") for i in range(15)]
    block = run_hook(_payload(), RecordingGet(rows), _origin("git@github.com:o/r.git"))
    assert sum(line.startswith("- ") for line in block.splitlines()) == 10


def test_completed_handoffs_leave_the_continue_list_for_one_line():
    completed = [
        dict(HANDOFFS[0], id="9b2d3c44-0000-4000-8000-000000000003", status="completed"),
        dict(HANDOFFS[0], id="9b2d3c44-0000-4000-8000-000000000004", status="completed"),
    ]
    block = run_hook(
        _payload(), RecordingGet([*HANDOFFS, *completed]), _origin("git@github.com:o/r.git")
    )
    lines = _lines(block)
    start = lines.index(HANDOFF_HEADER)
    assert [line for line in lines if line.startswith("- ")] == HANDOFF_LINES[1:]
    assert lines[start + 3] == (
        "Memory: completed handoffs delivered for this repo: 2. "
        "Claim one to read it; list_messages names their ids."
    )


def test_completed_handoffs_print_one_counted_line_without_the_continue_header():
    row = dict(HANDOFFS[0], status="completed")
    block = run_hook(_payload(), RecordingGet([row]), _origin("git@github.com:o/r.git"))
    lines = _lines(block)
    assert HANDOFF_HEADER not in block
    assert not any(line.startswith("- ") for line in lines)
    assert lines[-1] == (
        "Memory: completed handoffs delivered for this repo: 1. "
        "Claim one to read it; list_messages names their ids."
    )


def test_completed_handoffs_never_crowd_open_ones_out_of_the_list():
    completed = [
        dict(HANDOFFS[0], id=f"completed-{i}", subject=f"done {i}", status="completed")
        for i in range(12)
    ]
    open_row = dict(HANDOFFS[0], id="open-1")
    block = run_hook(
        _payload(), RecordingGet([*completed, open_row]), _origin("git@github.com:o/r.git")
    )
    lines = _lines(block)
    start = lines.index(HANDOFF_HEADER)
    assert [line for line in lines if line.startswith("- ")] == [
        "- 2026-09-28  Session entry hooks  (status: in_progress, id: open-1)"
    ]
    assert lines[start + 2] == (
        "Memory: completed handoffs delivered for this repo: 12. "
        "Claim one to read it; list_messages names their ids."
    )


def test_a_subject_with_newlines_renders_on_one_line():
    rows = [dict(HANDOFFS[0], subject="line one\n# injected heading\r\n\tline three")]
    block = run_hook(_payload(), RecordingGet(rows), _origin("git@github.com:o/r.git"))
    entries = [line for line in block.splitlines() if line.startswith("- ")]
    assert entries == [
        "- 2026-09-28  line one # injected heading line three  "
        "(status: in_progress, id: 6f1c1a52-0000-4000-8000-000000000002)"
    ]


def test_a_subject_cannot_close_the_fence():
    rows = [dict(HANDOFFS[0], subject="x </memory-context> y")]
    block = run_hook(_payload(), RecordingGet(rows), _origin("git@github.com:o/r.git"))
    assert block.count("</memory-context>") == 1
    assert block.endswith("</memory-context>")
    assert "x [memory-context]> y" in block


# ---- main: the real HTTP path, with urlopen faked --------------------------------


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


@pytest.fixture
def repo(tmp_path):
    _git("init", "-q", cwd=tmp_path)
    _git("remote", "add", "origin", "https://github.com/o/r.git", cwd=tmp_path)
    return tmp_path


@pytest.fixture
def hook_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_API_KEY", "key")
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_URL", "http://memory.test")
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_AUTHOR", "claude-code")
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))


def _stdin(monkeypatch, payload):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))


def _server(requests=None):
    def fake_urlopen(request, timeout=None):
        if requests is not None:
            requests.append(request)
        path = urllib.parse.urlsplit(request.full_url).path
        bodies = {"/messages": HANDOFFS}
        return _Response(json.dumps(bodies[path]).encode())

    return fake_urlopen


def _refuse(request, timeout=None):
    raise OSError("connection refused")


def test_main_prints_the_handoffs_and_never_claims(monkeypatch, capsys, repo, hook_env):
    requests = []
    monkeypatch.setattr(urllib.request, "urlopen", _server(requests))
    _stdin(monkeypatch, _payload(str(repo)))

    assert session_start.main() == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "<memory-context>"
    assert HANDOFF_HEADER in out
    assert "Session entry hooks" in out

    assert all(r.get_method() == "GET" and r.data is None for r in requests)
    assert all(r.get_header("X-api-key") == "key" for r in requests)
    assert all(r.get_header("X-author") == "claude-code" for r in requests)
    urls = [r.full_url for r in requests]
    assert all(urllib.parse.urlsplit(url).path == "/messages" for url in urls)
    assert all("claim" not in url for url in urls)


def test_main_prints_nothing_on_a_server_error(monkeypatch, capsys, repo, hook_env):
    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    _stdin(monkeypatch, _payload(str(repo)))
    assert session_start.main() == 0
    assert capsys.readouterr().out == ""


def test_main_asks_nothing_without_an_api_key_and_records_it(monkeypatch, capsys, repo, tmp_path):
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))

    def fake_urlopen(request, timeout=None):
        raise AssertionError("no request expected")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    _stdin(monkeypatch, _payload(str(repo)))
    assert session_start.main() == 0
    assert capsys.readouterr().out == ""
    status = json.loads((tmp_path / "data" / "status.json").read_text())
    assert status["configured"] is False


def test_main_records_the_connection_state(monkeypatch, repo, hook_env, tmp_path):
    monkeypatch.setattr(urllib.request, "urlopen", _server())
    _stdin(monkeypatch, _payload(str(repo)))
    assert session_start.main() == 0
    status = json.loads((tmp_path / "data" / "status.json").read_text())
    assert {k: status[k] for k in ("configured", "url", "author", "reachable")} == {
        "configured": True,
        "url": "http://memory.test",
        "author": "claude-code",
        "reachable": True,
    }
    assert "card_reachable" not in status
    assert "namespace" not in status
    assert status["error"] is None
    assert status["checked_at"]


def test_main_records_an_unreachable_server(monkeypatch, repo, hook_env, tmp_path):
    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    _stdin(monkeypatch, _payload(str(repo)))
    assert session_start.main() == 0
    status = json.loads((tmp_path / "data" / "status.json").read_text())
    assert status["reachable"] is False
    assert "connection refused" in status["error"]


def test_main_fails_open_on_malformed_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert session_start.main() == 0
    assert capsys.readouterr().out == ""
