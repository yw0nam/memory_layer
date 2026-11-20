"""Unit tests for the UserPromptSubmit recall hook — HTTP faked in process, and the real
process against local stub sockets."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import prompt_recall
from hook_common import data_dir

PLUGIN = Path(__file__).resolve().parents[3] / "integrations" / "claude_code"
PROMPT = "How do we retopologize the hair?"


def _payload(prompt=PROMPT, session="sess-1"):
    return {"session_id": session, "cwd": "/repo", "prompt": prompt}


def _note(n, text=None):
    return {
        "id": f"n{n}",
        "kind": "work",
        "tags": ["blender"],
        "date": "2026-11-01",
        "score": 0.8,
        "text": text or f"Note {n}.",
        "author": "claude-code",
    }


class FakePost:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def __call__(self, path, body, timeout):
        self.calls.append((path, body))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture(autouse=True)
def state_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))


def _run(post, payload=None):
    return prompt_recall.run(payload or _payload(), post, time.monotonic())


@pytest.mark.parametrize(
    "prompt",
    [
        "fix the typo",
        "<memory-context>\nMemory: a long fenced block of handoffs\n</memory-context>\n",
        "<memory-recall>\nNote: a long fenced block of notes\n</memory-recall>\nok go",
        "/review the whole branch against the spec",
    ],
    ids=["short", "fence-only", "short-after-fence", "slash-command"],
)
def test_a_short_fence_only_or_slash_prompt_sends_no_request(prompt):
    post = FakePost([_note(1)])

    assert _run(post, _payload(prompt=prompt)) == ""
    assert post.calls == []


def test_a_session_sees_each_note_once_and_a_note_cut_by_the_budget_comes_back():
    post = FakePost([_note(n, f"Note {n} " + "z" * 1740) for n in (1, 2, 3)])

    first = _run(post)
    second = _run(post)
    third = _run(post)
    other = _run(post, _payload(session="sess-2"))

    assert "Note 1" in first and "Note 2" in first and "Note 3" not in first
    assert "Note 3" in second and "Note 1" not in second
    assert third == ""
    assert "Note 1" in other
    log = (data_dir() / "recall.log").read_text().splitlines()
    assert len(log) == 4
    assert "n1,n2" in log[0] and "n3" in log[1]
    assert all("retopologize" not in line for line in log)


def test_a_server_failure_emits_nothing():
    assert _run(FakePost(OSError("connection refused"))) == ""


# ---- the real hook process -------------------------------------------------------


def _env(url, tmp_path):
    return {
        **os.environ,
        "CLAUDE_PLUGIN_ROOT": str(PLUGIN),
        "CLAUDE_PLUGIN_OPTION_URL": url,
        "CLAUDE_PLUGIN_OPTION_API_KEY": "key",
        "CLAUDE_PLUGIN_OPTION_AUTHOR": "claude-code",
        "CLAUDE_PLUGIN_DATA": str(tmp_path / "data"),
    }


@pytest.fixture
def search_server():
    reply = json.dumps([_note(1, "Hooks stay stdlib-only.")]).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200 if self.path == "/search" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_the_registered_hook_prints_the_recall_as_additional_context(search_server, tmp_path):
    [entry] = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]["UserPromptSubmit"]
    [hook] = entry["hooks"]
    assert hook["timeout"] == 4

    result = subprocess.run(
        hook["command"],
        shell=True,
        input=json.dumps(_payload()),
        capture_output=True,
        text=True,
        env=_env(search_server, tmp_path),
        timeout=10,
    )

    assert result.returncode == 0
    output = json.loads(result.stdout)["hookSpecificOutput"]
    assert output["hookEventName"] == "UserPromptSubmit"
    assert "Hooks stay stdlib-only." in output["additionalContext"]


def test_a_stalled_server_ends_the_hook_at_its_deadline_with_nothing_printed(tmp_path):
    # A listening socket that never answers; the cutoff is pushed out so only the watchdog ends it.
    stalled = socket.socket()
    stalled.bind(("127.0.0.1", 0))
    stalled.listen()
    code = "import prompt_recall as h; h.DEADLINE = 0.5; h.CUTOFF = 30; h.main()"
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PLUGIN / "hooks",
            input=json.dumps(_payload()),
            capture_output=True,
            text=True,
            env=_env(f"http://127.0.0.1:{stalled.getsockname()[1]}", tmp_path),
            timeout=10,
        )
    finally:
        stalled.close()

    assert result.returncode == 0
    assert result.stdout == ""
    assert time.monotonic() - started < prompt_recall.DEADLINE
