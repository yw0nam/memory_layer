"""Unit tests for the Claude Code SessionEnd capture hook — stdlib module, fake server.

Transcript parsing is exercised directly; the upload runs against a local
http.server that records every request, never the deployment.
"""

from __future__ import annotations

import io
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import capture_hook
from capture_hook import repo_name, transcript_turns

T0 = "2026-09-29T10:00:00.000Z"
T1 = "2026-09-29T10:00:05.000Z"
T2 = "2026-09-29T10:01:00.000Z"
T3 = "2026-09-29T10:02:00.000Z"
T4 = "2026-09-29T10:03:00.000Z"
EPOCH_T0 = 1790676000.0


def entry(kind, content, ts=T1, msg_id=None, **extra):
    message = {"role": kind, "content": content}
    if msg_id is not None:
        message["id"] = msg_id
    return {"type": kind, "message": message, "timestamp": ts, **extra}


def text(value):
    return {"type": "text", "text": value}


def lines(*entries):
    return [json.dumps(e) for e in entries]


# ---- transcript parsing -----------------------------------------------------------


def test_tool_blocks_and_harness_noise_are_dropped():
    turns, started, ended = transcript_turns(
        lines(
            {"type": "summary", "summary": "old", "timestamp": T0},
            entry("user", "<command-name>/clear</command-name>", T0),
            entry("user", "<local-command-stdout>cleared</local-command-stdout>", T0),
            entry("user", "<local-command-caveat>Caveat: local commands</local-command-caveat>"),
            entry("user", "<system-reminder>be careful</system-reminder>", T0),
            entry("user", "<task-notification><task-id>a1</task-id></task-notification>", T0),
            entry("user", [text("[Request interrupted by user]")], T0),
            entry("user", "Base directory for this skill: /x", T0, isMeta=True),
            entry("user", "This session is being continued...", T0, isCompactSummary=True),
            entry("user", "How should the staging images be built?", T1),
            entry(
                "assistant",
                [{"type": "thinking", "thinking": "hmm"}, text("Build them multi-arch.")],
                T2,
                "m1",
            ),
            entry("assistant", [{"type": "tool_use", "id": "t1", "name": "Bash"}], T2, "m1"),
            entry("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}], T2),
            entry("user", [{"type": "image", "source": {}}, text("and the deploy?")], T3),
            entry("assistant", [text("Drain the backend first.")], T4, "m2"),
        )
    )
    assert turns == [
        {"role": "user", "text": "How should the staging images be built?"},
        {"role": "assistant", "text": "Build them multi-arch."},
        {"role": "user", "text": "and the deploy?"},
        {"role": "assistant", "text": "Drain the backend first."},
    ]
    assert started == pytest.approx(EPOCH_T0 + 5)
    assert ended == pytest.approx(EPOCH_T0 + 180)


def test_assistant_entries_of_one_model_message_merge_into_one_turn():
    turns, _, _ = transcript_turns(
        lines(
            entry("user", "Explain the reranker floor."),
            entry("assistant", [text("The floor is 0.25.")], T2, "m1"),
            entry("assistant", [{"type": "tool_use", "id": "t1", "name": "Read"}], T2, "m1"),
            entry("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}], T2),
            entry("assistant", [text("It applies after reranking.")], T2, "m1"),
            entry("assistant", [text("A new message.")], T3, "m2"),
        )
    )
    assert turns == [
        {"role": "user", "text": "Explain the reranker floor."},
        {"role": "assistant", "text": "The floor is 0.25.\n\nIt applies after reranking."},
        {"role": "assistant", "text": "A new message."},
    ]


def test_user_entries_are_never_merged():
    turns, _, _ = transcript_turns(
        lines(entry("user", "first question"), entry("user", [text("second question")]))
    )
    assert turns == [
        {"role": "user", "text": "first question"},
        {"role": "user", "text": "second question"},
    ]


def test_unreadable_lines_are_skipped():
    turns, _, _ = transcript_turns(
        ["not json", "", json.dumps(entry("user", "hello there")), '{"type": "user"}']
    )
    assert turns == [{"role": "user", "text": "hello there"}]


@pytest.mark.parametrize(
    "resumed",
    [
        [entry("user", "one more thing", T3), entry("assistant", [text("sure")], T4, "m3")],
        [entry("assistant", [text("continuing")], T3, "m3"), entry("user", "thanks", T4)],
    ],
)
def test_a_resumed_transcript_keeps_the_earlier_turns_as_a_prefix(resumed):
    first = [
        entry("user", "start here", T0),
        entry("assistant", [text("started")], T1, "m1"),
        entry("user", "and then?", T2),
    ]
    if resumed[0]["type"] == "assistant":
        first = first[:2]
    before, _, _ = transcript_turns(lines(*first))
    after, _, _ = transcript_turns(lines(*first, *resumed))
    assert after[: len(before)] == before
    assert len(after) == len(before) + 2


# ---- repo name --------------------------------------------------------------------


def test_repo_name_is_the_git_toplevel_basename(tmp_path):
    repo = tmp_path / "my-service"
    (repo / "src" / "pkg").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    assert repo_name(str(repo / "src" / "pkg")) == "my-service"


def test_repo_name_outside_git_is_the_cwd_basename(tmp_path):
    plain = tmp_path / "notes-dir"
    plain.mkdir()
    assert repo_name(str(plain)) == "notes-dir"


# ---- the upload against a fake server ----------------------------------------------


class FakeServer:
    def __init__(self, status=201):
        self.status = status
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                server.requests.append(
                    {
                        "path": self.path,
                        "key": self.headers.get("X-API-Key"),
                        "body": json.loads(self.rfile.read(length)),
                    }
                )
                payload = json.dumps({"id": "conv:1", "created": True, "job_id": "j"}).encode()
                self.send_response(server.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                return None

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    fake = FakeServer()
    yield fake
    fake.close()


@pytest.fixture
def hook_env(monkeypatch, tmp_path, server):
    monkeypatch.setenv("MEMORY_BASE_URL", server.url)
    monkeypatch.setenv("MEMORY_BASE_API_KEY", "capture-key")
    monkeypatch.setenv("MEMORY_CAPTURE_LOG", str(tmp_path / "capture.jsonl"))
    monkeypatch.delenv("MEMORY_BASE_CAPTURE_NAMESPACE", raising=False)
    return tmp_path


def _write_transcript(path: Path, *entries) -> Path:
    path.write_text("\n".join(lines(*entries)) + "\n")
    return path


def _run_main(monkeypatch, payload) -> int:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    return capture_hook.main()


def _log_rows(tmp_path) -> list[dict]:
    path = tmp_path / "capture.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _payload(transcript: Path, cwd: Path) -> dict:
    return {
        "session_id": "sess-42",
        "transcript_path": str(transcript),
        "cwd": str(cwd),
        "reason": "prompt_input_exit",
        "hook_event_name": "SessionEnd",
    }


def test_the_session_is_posted_as_one_conversation(monkeypatch, hook_env, server):
    cwd = hook_env / "project-x"
    cwd.mkdir()
    transcript = _write_transcript(
        hook_env / "t.jsonl",
        entry("user", "How should the staging images be built?", T1),
        entry("assistant", [text("Build them multi-arch.")], T2, "m1"),
    )
    assert _run_main(monkeypatch, _payload(transcript, cwd)) == 0
    (request,) = server.requests
    assert request["path"] == "/conversations"
    assert request["key"] == "capture-key"
    assert request["body"] == {
        "origin": "claude_code",
        "external_session_id": "sess-42",
        "namespace": "dev",
        "started_at": pytest.approx(EPOCH_T0 + 5),
        "ended_at": pytest.approx(EPOCH_T0 + 60),
        "turns": [
            {"role": "user", "text": "How should the staging images be built?"},
            {"role": "assistant", "text": "Build them multi-arch."},
        ],
        "metadata": {"repo": "project-x", "cwd": str(cwd)},
    }
    (row,) = _log_rows(hook_env)
    assert row["decision"] == "uploaded"
    assert row["turns"] == 2


def test_the_namespace_comes_from_the_environment(monkeypatch, hook_env, server):
    monkeypatch.setenv("MEMORY_BASE_CAPTURE_NAMESPACE", "work")
    transcript = _write_transcript(
        hook_env / "t.jsonl",
        entry("user", "first question here"),
        entry("assistant", [text("an answer")], T2, "m1"),
    )
    _run_main(monkeypatch, _payload(transcript, hook_env))
    assert server.requests[0]["body"]["namespace"] == "work"


def test_fewer_than_two_turns_sends_nothing(monkeypatch, hook_env, server):
    transcript = _write_transcript(
        hook_env / "t.jsonl",
        entry("user", "<command-name>/exit</command-name>"),
        entry("user", "just one real turn"),
    )
    assert _run_main(monkeypatch, _payload(transcript, hook_env)) == 0
    assert server.requests == []
    assert _log_rows(hook_env)[0]["decision"] == "skipped"


def test_a_server_error_exits_zero_and_is_logged(monkeypatch, hook_env, server):
    server.status = 500
    transcript = _write_transcript(
        hook_env / "t.jsonl",
        entry("user", "first question here"),
        entry("assistant", [text("an answer")], T2, "m1"),
    )
    assert _run_main(monkeypatch, _payload(transcript, hook_env)) == 0
    assert len(server.requests) == 1
    (row,) = _log_rows(hook_env)
    assert row["decision"] == "error"
    assert "500" in row["error"]


def test_an_unreachable_server_exits_zero(monkeypatch, hook_env, server):
    monkeypatch.setenv("MEMORY_BASE_URL", "http://127.0.0.1:9")
    transcript = _write_transcript(
        hook_env / "t.jsonl",
        entry("user", "first question here"),
        entry("assistant", [text("an answer")], T2, "m1"),
    )
    assert _run_main(monkeypatch, _payload(transcript, hook_env)) == 0
    assert _log_rows(hook_env)[0]["decision"] == "error"


def test_a_missing_transcript_exits_zero(monkeypatch, hook_env, server):
    assert _run_main(monkeypatch, _payload(hook_env / "gone.jsonl", hook_env)) == 0
    assert server.requests == []
    assert _log_rows(hook_env)[0]["decision"] == "error"


def test_malformed_stdin_exits_zero(monkeypatch, hook_env, server):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))
    assert capture_hook.main() == 0
    assert server.requests == []


def test_no_api_key_sends_nothing(monkeypatch, hook_env, server):
    monkeypatch.delenv("MEMORY_BASE_API_KEY")
    monkeypatch.setenv("MEMORY_BASE_ENV", str(hook_env / "missing-env"))
    transcript = _write_transcript(
        hook_env / "t.jsonl",
        entry("user", "first question here"),
        entry("assistant", [text("an answer")], T2, "m1"),
    )
    assert _run_main(monkeypatch, _payload(transcript, hook_env)) == 0
    assert server.requests == []
