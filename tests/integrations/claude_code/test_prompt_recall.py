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
from recall import tail

PLUGIN = Path(__file__).resolve().parents[3] / "integrations" / "claude_code"
PROMPT = "How do we retopologize the hair?"
ORIGIN = "git@github.com:o/r.git"
SCOPE = "repo:github.com/o/r"


def _payload(prompt=PROMPT, session="sess-1", transcript=""):
    return {"session_id": session, "cwd": "/repo", "prompt": prompt, "transcript_path": transcript}


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


def _decision(n, decision=None):
    return {
        "number": n,
        "title": f"Decision title {n}",
        "decided_on": "2026-10-30",
        "decision": decision or f"Decision text {n}.",
        "why": None,
        "links": [],
        "score": 0.8,
    }


def _reply(notes=(), decisions=(), errors=()):
    return {"notes": list(notes), "decisions": list(decisions), "errors": list(errors)}


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


def _run(post, payload=None, origin=None):
    return prompt_recall.run(payload or _payload(), post, time.monotonic(), lambda cwd: origin)


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
    post = FakePost(_reply([_note(1)]))

    assert _run(post, _payload(prompt=prompt)) == ""
    assert post.calls == []


def test_a_slash_command_sends_no_request_even_after_an_assistant_message():
    tail.save("sess-1", "A long assistant message.")
    post = FakePost(_reply([_note(1)]))

    assert _run(post, _payload(prompt="/review the branch")) == ""
    assert post.calls == []


def test_a_short_reply_recalls_with_the_previous_assistant_text_from_the_tail_file():
    tail.save("sess-1", "Shall I retopologize the hair with the old rig?")
    post = FakePost(_reply([_note(1)]))

    assert "Note 1." in _run(post, _payload(prompt="yes"))
    [(path, body)] = post.calls
    assert path == "/recall"
    assert body["query"] == "Shall I retopologize the hair with the old rig?\n\nyes"


def test_a_short_reply_falls_back_to_the_transcript_tail(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Use the old rig?"}]},
            }
        )
        + "\n"
    )
    post = FakePost(_reply([_note(1)]))

    _run(post, _payload(prompt="ok", transcript=str(transcript)))

    assert post.calls[0][1]["query"] == "Use the old rig?\n\nok"


def test_a_short_reply_without_an_assistant_message_sends_no_request():
    post = FakePost(_reply([_note(1)]))

    assert _run(post, _payload(prompt="yes", transcript="/missing.jsonl")) == ""
    assert post.calls == []


def test_the_request_carries_the_scope_of_a_portable_origin_only():
    post = FakePost(_reply())

    _run(post, origin=ORIGIN)
    _run(post, origin="/srv/git/r.git")

    assert post.calls[0][1]["scope"] == SCOPE
    assert "scope" not in post.calls[1][1]


def test_decisions_come_before_notes():
    post = FakePost(_reply([_note(1)], [_decision(3)]))

    context = _run(post, origin=ORIGIN)

    assert context.index("D-3 Decision title 3") < context.index("Note n1")


def test_five_asked_then_the_shown_dropped_then_two_decisions_and_three_notes():
    post = FakePost(_reply([_note(n) for n in range(1, 6)], [_decision(n) for n in range(1, 6)]))

    first = _run(post, origin=ORIGIN)
    second = _run(post, origin=ORIGIN)
    third = _run(post, origin=ORIGIN)

    assert [f"D-{n} " in first for n in range(1, 6)] == [True, True, False, False, False]
    assert [f"Note n{n} " in first for n in range(1, 6)] == [True, True, True, False, False]
    assert [f"D-{n} " in second for n in range(1, 6)] == [False, False, True, True, False]
    assert [f"Note n{n} " in second for n in range(1, 6)] == [False, False, False, True, True]
    assert "D-5 " in third and "Note" not in third


def test_a_session_sees_each_item_once_and_a_note_cut_by_the_budget_comes_back():
    post = FakePost(_reply([_note(n, f"Note {n} " + "z" * 1740) for n in (1, 2, 3)]))

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


def test_the_log_line_names_the_errors_and_the_emitted_keys_but_never_the_texts():
    tail.save("sess-1", "A secret assistant sentence.")
    post = FakePost(_reply([], [_decision(3)], errors=["notes"]))

    _run(post, _payload(prompt="a secret prompt sentence"), origin=ORIGIN)

    [line] = (data_dir() / "recall.log").read_text().splitlines()
    assert f"emitted=decision:{SCOPE}#3" in line and "errors=notes" in line
    assert "secret" not in line


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
    reply = json.dumps(_reply([_note(1, "Hooks stay stdlib-only.")])).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200 if self.path == "/recall" else 404)
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
