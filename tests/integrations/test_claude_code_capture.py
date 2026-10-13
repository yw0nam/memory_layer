"""Unit tests for the Claude Code turn capture: the shared poster, the two hooks, and emit."""

from __future__ import annotations

import io
import json
import re
import tempfile
import uuid
from pathlib import Path

import pytest

import hook_common
import prompt_submit
import stop
from hook_common import Config
from hook_common import Refused
from hook_common import Unreachable
from hook_common import post_turn

PLUGIN = Path(__file__).parents[2] / "integrations" / "claude_code"
CFG = Config("http://memory.test", "key", "claude-code", None)
UNREACHABLE = "memory-base unreachable; turns are not being saved"
RECOVERED = "memory-base is reachable again; turns are being saved"


class FakeServer:
    """Stands in for hook_common.post: records bodies and replays queued outcomes."""

    def __init__(self):
        self.bodies = []
        self.outcomes = []

    def __call__(self, cfg, path, body, timeout):
        assert path == "/turns"
        assert timeout == hook_common.TURN_TIMEOUT
        self.bodies.append(body)
        outcome = self.outcomes.pop(0) if self.outcomes else {"seq": 1}
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def server(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(hook_common, "post", fake)
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(hook_common, "git_origin", lambda cwd: "git@github.com:o/r.git")
    clock = {"now": 1000.0}
    monkeypatch.setattr(hook_common, "_now", lambda: clock["now"])
    fake.clock = clock
    return fake


def _marker(tmp_path, session="s1"):
    return tmp_path / "data" / "sessions" / f"{session}.json"


def test_a_turn_posts_its_role_origin_session_uuid_and_repo_scope(server):
    assert post_turn(CFG, "s1", "user", "hello there", "/work/r") is None
    (body,) = server.bodies
    assert uuid.UUID(body.pop("id"))
    assert body == {
        "origin": "claude_code",
        "session_id": "s1",
        "role": "user",
        "content": "hello there",
        "repo": "repo:github.com/o/r",
    }


def test_the_namespace_is_sent_only_when_configured(server):
    post_turn(Config("u", "k", "claude-code", "work"), "s1", "user", "x", "/w")
    assert server.bodies[0]["namespace"] == "work"


def test_the_repo_is_left_out_without_a_portable_origin(server, monkeypatch):
    monkeypatch.setattr(hook_common, "git_origin", lambda cwd: None)
    post_turn(CFG, "s1", "user", "x", "/w")
    assert "repo" not in server.bodies[0]


def test_unreachable_notifies_once_and_skips_requests_for_sixty_seconds(server, tmp_path):
    server.outcomes = [Unreachable("down")]
    assert post_turn(CFG, "s1", "user", "a", "/w") == UNREACHABLE
    assert json.loads(_marker(tmp_path).read_text())["notified"] is True

    server.clock["now"] += 59
    assert post_turn(CFG, "s1", "assistant", "b", "/w") is None
    assert len(server.bodies) == 1

    server.clock["now"] += 2
    server.outcomes = [Unreachable("down")]
    assert post_turn(CFG, "s1", "user", "c", "/w") is None
    assert len(server.bodies) == 2


def test_the_first_success_after_a_failure_says_so_once_and_clears_the_marker(server, tmp_path):
    server.outcomes = [Unreachable("down")]
    post_turn(CFG, "s1", "user", "a", "/w")
    server.clock["now"] += 61
    assert post_turn(CFG, "s1", "user", "b", "/w") == RECOVERED
    assert not _marker(tmp_path).exists()
    assert post_turn(CFG, "s1", "user", "c", "/w") is None


def test_a_refusal_prints_its_reason_once_and_does_not_pause_capture(server, tmp_path):
    server.outcomes = [Refused(401, "invalid key"), Refused(401, "invalid key")]
    first = post_turn(CFG, "s1", "user", "a", "/w")
    assert first == "memory-base refused this turn (401: invalid key); check the API key in /config"
    assert post_turn(CFG, "s1", "user", "b", "/w") is None
    assert len(server.bodies) == 2
    assert "failed_at" not in json.loads(_marker(tmp_path).read_text())


def test_a_refusal_message_is_trimmed(server):
    server.outcomes = [Refused(413, "x" * 500)]
    message = post_turn(CFG, "s1", "user", "a", "/w")
    assert message == f"memory-base refused this turn (413: {'x' * 120})"


# ---- the hooks ----------------------------------------------------------------


@pytest.fixture
def hook_env(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("MEMORY_PREFETCH_LOG", str(tmp_path / "prefetch.jsonl"))
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_URL", "http://memory.test")
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_API_KEY", "key")
    monkeypatch.setattr(prompt_submit, "_fetch_from_server", lambda cfg: lambda query: [])


def _run(module, monkeypatch, payload):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    return module.main()


@pytest.mark.parametrize("prompt", ["hi", "a much longer question about the tunnel setup"])
def test_every_prompt_is_captured_as_a_user_turn(server, hook_env, monkeypatch, prompt):
    payload = {"session_id": "s1", "prompt": prompt, "cwd": "/w"}
    assert _run(prompt_submit, monkeypatch, payload) == 0
    assert [(b["role"], b["content"]) for b in server.bodies] == [("user", prompt)]


@pytest.mark.parametrize("prompt", ["/compact", "  !git status"])
def test_slash_and_bang_prompts_are_not_captured(server, hook_env, monkeypatch, prompt):
    assert _run(prompt_submit, monkeypatch, {"session_id": "s1", "prompt": prompt}) == 0
    assert server.bodies == []


def test_the_capture_line_and_the_prefetch_block_leave_as_one_json_object(
    server, hook_env, monkeypatch, capsys
):
    server.outcomes = [Unreachable("down")]
    monkeypatch.setattr(
        prompt_submit,
        "_fetch_from_server",
        lambda cfg: lambda query: [{"date": "2026-01-01", "score": 0.9, "text": "a note"}],
    )
    payload = {"session_id": "s1", "prompt": "x" * 30, "cwd": "/w"}
    assert _run(prompt_submit, monkeypatch, payload) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["systemMessage"] == UNREACHABLE
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "a note" in out["hookSpecificOutput"]["additionalContext"]


def test_stop_captures_the_final_answer(server, hook_env, monkeypatch):
    payload = {"session_id": "s1", "last_assistant_message": "done", "cwd": "/w"}
    assert _run(stop, monkeypatch, payload) == 0
    assert [(b["role"], b["content"]) for b in server.bodies] == [("assistant", "done")]


@pytest.mark.parametrize("payload", [{"session_id": "s1"}, {"last_assistant_message": ""}])
def test_stop_without_an_answer_posts_nothing(server, hook_env, monkeypatch, payload):
    assert _run(stop, monkeypatch, payload) == 0
    assert server.bodies == []


def test_the_hooks_do_nothing_without_a_key(server, monkeypatch, capsys):
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_API_KEY", raising=False)
    assert _run(stop, monkeypatch, {"session_id": "s1", "last_assistant_message": "x"}) == 0
    assert server.bodies == []
    assert capsys.readouterr().out == ""


# ---- emit ---------------------------------------------------------------------


def test_emit_prints_plain_text_for_context_alone(capsys):
    hook_common.emit("UserPromptSubmit", "ctx", None)
    assert capsys.readouterr().out == "ctx\n"


def test_emit_prints_one_json_object_with_a_system_message(capsys):
    hook_common.emit("UserPromptSubmit", "ctx", "line")
    assert json.loads(capsys.readouterr().out) == {
        "hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "ctx"},
        "systemMessage": "line",
    }
    hook_common.emit("Stop", None, "line")
    assert json.loads(capsys.readouterr().out) == {"systemMessage": "line"}


def test_emit_prints_nothing_when_both_are_absent(capsys):
    hook_common.emit("Stop", None, None)
    assert capsys.readouterr().out == ""


# ---- the plugin files ---------------------------------------------------------


def test_hooks_and_mcp_reference_only_the_plugin_root_and_user_config():
    for name in ("hooks/hooks.json", ".mcp.json"):
        text = (PLUGIN / name).read_text()
        assert set(re.findall(r"\$\{([^}]+)\}", text)) <= {
            "CLAUDE_PLUGIN_ROOT",
            "user_config.mcp_url",
            "user_config.api_key",
        }
    events = json.loads((PLUGIN / "hooks/hooks.json").read_text())["hooks"]
    assert set(events) == {"SessionStart", "UserPromptSubmit", "Stop"}
