"""Unit tests for the Claude Code prefetch hook — pure module, stdlib only.

Exercises prefetch_hook.py directly (importable via the ``pythonpath`` test
config entry, mirroring how tests/integrations/test_hermes_client.py imports
the Hermes client). The HTTP fetch is injected, never real.
"""

from __future__ import annotations

import json

import pytest

from prefetch_hook import _resolve_api_key
from prefetch_hook import build_context_block
from prefetch_hook import is_trivial_prompt
from prefetch_hook import run_hook

HITS = [
    {"date": "2026-08-22", "score": 0.97, "text": "tunnel terminates at 127.0.0.1:8770"},
    {"date": "2026-07-27", "score": 0.91, "text": "OTLP tokens differ between hosts"},
]


def _payload(prompt: str, session: str = "sess-1") -> dict:
    return {"session_id": session, "prompt": prompt, "cwd": "/repo"}


@pytest.fixture
def env(tmp_path):
    return {"state_dir": tmp_path / "state", "log_path": tmp_path / "log.jsonl"}


# ---- is_trivial_prompt -----------------------------------------------------


def test_trivial_short_prompt():
    assert is_trivial_prompt("ㅇㅋ 진행해")


def test_trivial_slash_command():
    assert is_trivial_prompt("/compact")


def test_trivial_harness_markers():
    assert is_trivial_prompt("<task-notification>...</task-notification>")
    assert is_trivial_prompt("<command-name>/foo</command-name>")
    assert is_trivial_prompt("Caveat: the messages below were generated...")
    assert is_trivial_prompt("!git status")


def test_real_question_is_not_trivial():
    assert not is_trivial_prompt("why does the TTS server reject non-ascii voice names?")


# ---- build_context_block ---------------------------------------------------


def test_block_formats_hits_with_dates():
    block = build_context_block(HITS)
    assert block.startswith("<memory-context")
    assert block.endswith("</memory-context>")
    assert "- [2026-08-22] tunnel terminates at 127.0.0.1:8770" in block
    assert "- [2026-07-27] OTLP tokens differ between hosts" in block


def test_block_empty_for_no_hits():
    assert build_context_block([]) == ""


def test_block_truncates_at_line_boundary():
    hits = [{"date": "2026-08-01", "score": 0.9, "text": "x" * 400} for _ in range(10)]
    block = build_context_block(hits)
    assert len(block) <= 1500
    for line in block.splitlines():
        assert not line.endswith("x") or line.endswith("x" * 400)


# ---- run_hook --------------------------------------------------------------


def test_trivial_prompt_skips_search(env):
    def fetch(query):
        raise AssertionError("no search expected")

    assert run_hook(_payload("ㅇㅋ"), fetch, **env) == ""


def test_injects_block_and_logs(env):
    out = run_hook(_payload("how do the n8n signals reach the mac?"), lambda q: HITS, **env)
    assert "8770" in out
    rows = [json.loads(line) for line in env["log_path"].read_text().splitlines()]
    assert rows[-1]["decision"] == "injected"
    assert rows[-1]["session_id"] == "sess-1"


def test_same_session_never_reinjects_same_note(env):
    q = "how do the n8n signals reach the mac?"
    assert "8770" in run_hook(_payload(q), lambda _: HITS, **env)
    assert run_hook(_payload(q), lambda _: HITS, **env) == ""


def test_other_session_gets_the_note_again(env):
    q = "how do the n8n signals reach the mac?"
    run_hook(_payload(q, session="a"), lambda _: HITS, **env)
    assert "8770" in run_hook(_payload(q, session="b"), lambda _: HITS, **env)


def test_fetch_error_fails_open(env):
    def fetch(query):
        raise OSError("connection refused")

    assert run_hook(_payload("how do the n8n signals reach the mac?"), fetch, **env) == ""
    rows = [json.loads(line) for line in env["log_path"].read_text().splitlines()]
    assert rows[-1]["decision"] == "error"


def test_empty_result_logs_empty(env):
    assert run_hook(_payload("how do the n8n signals reach the mac?"), lambda _: [], **env) == ""
    rows = [json.loads(line) for line in env["log_path"].read_text().splitlines()]
    assert rows[-1]["decision"] == "empty"


# ---- _resolve_api_key --------------------------------------------------------


@pytest.fixture
def key_env(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMORY_BASE_API_KEY", raising=False)
    monkeypatch.delenv("MEMORY_BASE_ENV", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_key_env_var_wins(key_env, monkeypatch):
    monkeypatch.setenv("MEMORY_BASE_API_KEY", "from-env")
    assert _resolve_api_key() == "from-env"


def test_key_from_explicit_env_file(key_env, monkeypatch, tmp_path):
    env_file = tmp_path / "custom.env"
    env_file.write_text("OTHER=x\nMEMORY_BASE_API_KEY=from-file\n")
    monkeypatch.setenv("MEMORY_BASE_ENV", str(env_file))
    assert _resolve_api_key() == "from-file"


def test_key_defaults_to_config_dir_env_file(key_env):
    default = key_env / ".config" / "memory-base" / "env"
    default.parent.mkdir(parents=True)
    default.write_text("MEMORY_BASE_API_KEY=from-default\n")
    assert _resolve_api_key() == "from-default"


def test_key_empty_when_nothing_configured(key_env):
    assert _resolve_api_key() == ""


def test_query_truncated_before_fetch(env):
    seen = {}

    def fetch(query):
        seen["q"] = query
        return []

    run_hook(_payload("y" * 5000), fetch, **env)
    assert len(seen["q"]) <= 500
