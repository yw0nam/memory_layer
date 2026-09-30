"""Unit coverage for the headless Claude Code model client (fake subprocess)."""

from __future__ import annotations

import asyncio
import json

import pytest

from memory_base.eval import claude_code


def test_the_claude_code_client_runs_a_tool_less_headless_session(monkeypatch):
    sent = {}

    class Proc:
        returncode = 0

        async def communicate(self, data):
            sent["stdin"] = data.decode()
            result = {
                "result": " yes ",
                "num_turns": 1,
                "is_error": False,
                "usage": {"input_tokens": 90, "cache_read_input_tokens": 10, "output_tokens": 7},
                "modelUsage": {"claude-sonnet-5-5": {"thinkingTokens": 3}},
            }
            return json.dumps(result).encode(), b""

    async def spawn(*argv, **kwargs):
        sent["argv"], sent["env"] = argv, kwargs["env"]
        return Proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    client = claude_code.ClaudeCodeModel(
        model="claude-sonnet-5-5", effort="high", system_prompt="S"
    )
    assert asyncio.run(client.complete("the prompt", max_tokens=10)) == ("yes", 100, 7)
    assert sent["stdin"] == "the prompt"
    argv = sent["argv"]
    for flag, value in [
        ("--model", "claude-sonnet-5-5"),
        ("--tools", ""),
        ("--setting-sources", ""),
        ("--system-prompt", "S"),
    ]:
        assert argv[argv.index(flag) + 1] == value
    assert "--strict-mcp-config" in argv
    assert sent["env"]["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "1"
    assert client.timeout == claude_code.CALL_TIMEOUT_SECONDS


def test_the_claude_code_client_refuses_a_reply_from_another_model_or_a_tool_turn(monkeypatch):
    def result(**overrides):
        base = {"result": "yes", "num_turns": 1, "is_error": False, "usage": {}}
        base["modelUsage"] = {"claude-sonnet-5-5": {}}
        return {**base, **overrides}

    for bad in (
        result(modelUsage={"claude-opus-5-5": {}}),
        result(num_turns=2),
        result(is_error=True),
    ):

        class Proc:
            returncode = 0

            async def communicate(self, data, payload=bad):
                return json.dumps(payload).encode(), b""

        async def spawn(*argv, **kwargs):
            return Proc()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        client = claude_code.ClaudeCodeModel(
            model="claude-sonnet-5-5", effort="high", system_prompt="S"
        )
        with pytest.raises(RuntimeError):
            asyncio.run(client.complete("p", max_tokens=10))


def test_strip_fence_removes_a_surrounding_json_fence_and_whitespace():
    assert claude_code.strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert claude_code.strip_fence('  \n```\n{"a": 1}\n```  \n') == '{"a": 1}'
    assert claude_code.strip_fence('{"a": 1}') == '{"a": 1}'
    assert claude_code.strip_fence('  {"a": 1}\n') == '{"a": 1}'
