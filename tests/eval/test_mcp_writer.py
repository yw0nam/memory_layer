"""Unit coverage for the LongMemEval writer that saves through the real MCP tools."""

from __future__ import annotations

import asyncio
import json

import pytest

from memory_base.eval import mcp_writer


def _line(event: dict) -> str:
    return json.dumps(event)


def _tool_use(tool_id: str, name: str, payload: dict) -> str:
    content = [{"type": "tool_use", "id": tool_id, "name": name, "input": payload}]
    return _line({"type": "assistant", "message": {"content": content}})


def _tool_result(tool_id: str, text: str, *, is_error: bool = False) -> str:
    block = {"type": "tool_result", "tool_use_id": tool_id, "is_error": is_error, "content": text}
    return _line({"type": "user", "message": {"content": [block]}})


INIT = _line(
    {
        "type": "system",
        "subtype": "init",
        "tools": ["mcp__memory-base__search_memory", "mcp__memory-base__save_memory"],
        "mcp_servers": [{"name": "memory-base", "status": "connected"}],
    }
)
RESULT = _line(
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 4,
        "duration_ms": 9100,
        "usage": {
            "input_tokens": 5,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 1000,
            "output_tokens": 40,
        },
        "modelUsage": {"claude-sonnet-5-5": {}},
        "permission_denials": [],
    }
)


def test_the_prompt_is_the_published_instruction_the_author_and_the_session():
    turns = [
        {"role": "user", "content": "I adopted a cat.", "has_answer": True},
        {"role": "assistant", "content": "Congratulations!"},
    ]
    prompt = mcp_writer.writer_prompt("2023/05/20 (Sat) 02:21", turns)
    assert prompt == (
        f"{mcp_writer.INSTRUCTION}\n"
        f"{mcp_writer.AUTHOR_LINE}\n\n"
        "Session date: 2023-05-20T02:21:00\n\n"
        "User: I adopted a cat.\n\n"
        "Assistant: Congratulations!"
    )
    assert "has_answer" not in prompt
    assert mcp_writer.AUTHOR in mcp_writer.AUTHOR_LINE


def test_the_stream_becomes_a_session_record():
    saved = {"id": "note:lme-q:a", "kind": "personal", "stored": True, "superseded": "note:lme-q:b"}
    duplicate = {"id": "note:lme-q:c", "kind": "personal", "stored": False}
    lines = [
        INIT,
        _tool_use("t1", "mcp__memory-base__search_memory", {"query": "cat"}),
        _tool_result("t1", '{"result": []}'),
        _tool_use("t2", "mcp__memory-base__save_memory", {"content": "A"}),
        _tool_result("t2", json.dumps(saved)),
        _tool_use("t3", "mcp__memory-base__save_memory", {"content": "B"}),
        _tool_result("t3", [{"type": "text", "text": json.dumps(duplicate)}]),
        _tool_use("t4", "mcp__memory-base__save_memory", {"content": "C"}),
        _tool_result("t4", "Error executing tool save_memory: not for this gate", is_error=True),
        RESULT,
    ]
    record = mcp_writer.parse_stream(lines)
    assert record["tools"] == [
        "mcp__memory-base__search_memory",
        "mcp__memory-base__save_memory",
    ]
    assert record["mcp_servers"] == ["memory-base"]
    assert [call["name"] for call in record["tool_calls"]] == [
        "search_memory",
        "save_memory",
        "save_memory",
        "save_memory",
    ]
    assert record["saves"] == [
        {"tool": "save_memory", "id": "note:lme-q:a", "stored": True,
         "superseded": "note:lme-q:b"},
        {"tool": "save_memory", "id": "note:lme-q:c", "stored": False,
         "superseded": None},
    ]  # fmt: skip
    assert record["refusals"] == [
        {
            "tool": "save_memory",
            "error": "Error executing tool save_memory: not for this gate",
        }
    ]
    assert record["usage"] == {
        "input_tokens": 5,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 1000,
        "output_tokens": 40,
    }
    assert record["num_turns"] == 4
    assert record["models"] == ["claude-sonnet-5-5"]
    assert record["is_error"] is False
    assert record["subtype"] == "success"
    assert record["permission_denials"] == 0


def test_a_stream_without_a_result_event_is_an_error():
    record = mcp_writer.parse_stream([INIT])
    assert record["is_error"] is True
    assert record["subtype"] == "no_result"


def test_provenance_follows_saves_supersedes_and_duplicates():
    provenance = {"note:b": {"s1"}, "note:c": {"s2"}}
    saves = [
        {"tool": "save_memory", "id": "note:a", "stored": True, "superseded": "note:b"},
        {"tool": "save_memory", "id": "note:c", "stored": False, "superseded": None},
    ]
    mcp_writer.record_provenance(provenance, "s3", saves, created=["note:a", "note:d"])
    assert provenance == {
        "note:a": {"s1", "s3"},
        "note:b": {"s1"},
        "note:c": {"s2", "s3"},
        "note:d": {"s3"},
    }


def test_the_row_diff_names_created_and_newly_archived_notes():
    before = {"note:a": {"archived": False}, "note:b": {"archived": False}}
    after = {
        "note:a": {"archived": True},
        "note:b": {"archived": False},
        "note:c": {"archived": False},
    }
    assert mcp_writer.diff_rows(before, after) == (["note:c"], ["note:a"])


def test_the_mcp_config_points_only_at_a_loopback_eval_backend(tmp_path):
    config = mcp_writer.mcp_config(tmp_path, "the-key", "http://127.0.0.1:18555")
    server = config["mcpServers"]["memory-base"]
    assert list(config["mcpServers"]) == ["memory-base"]
    assert server["env"] == {
        "MCP_TRANSPORT": "stdio",
        "MEMORY_API_KEY": "the-key",
        "REST_URL": "http://127.0.0.1:18555",
    }
    assert str(tmp_path) in server["args"]
    for production in ("http://localhost:8010", "http://127.0.0.1:8010", "http://10.0.0.5:18555"):
        with pytest.raises(ValueError):
            mcp_writer.mcp_config(tmp_path, "the-key", production)


def test_the_agent_runs_an_isolated_headless_session(monkeypatch, tmp_path):
    sent = {}

    class Proc:
        returncode = 0

        async def communicate(self, data):
            sent["stdin"] = data.decode()
            return "\n".join([INIT, RESULT]).encode(), b""

    async def spawn(*argv, **kwargs):
        sent["argv"], sent["env"], sent["cwd"] = argv, kwargs["env"], kwargs["cwd"]
        return Proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    agent = mcp_writer.ClaudeCodeWriter(model="sonnet")
    config = mcp_writer.mcp_config(tmp_path, "k", "http://127.0.0.1:18555")
    record = asyncio.run(agent.run("the prompt", config))
    argv = sent["argv"]
    assert sent["stdin"] == "the prompt"
    assert argv[:4] == ("claude", "-p", "--model", "sonnet")
    for flag in ("--strict-mcp-config", "--no-session-persistence"):
        assert flag in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert argv[argv.index("--allowedTools") + 1] == "mcp__memory-base"
    assert json.loads(argv[argv.index("--settings") + 1]) == {"autoMemoryEnabled": False}
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--system-prompt" not in argv
    assert sent["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert sent["cwd"] != str(tmp_path)
    assert record["num_turns"] == 4
    assert record["seconds"] >= 0


def test_the_eval_api_gates_with_the_benchmark_key(monkeypatch):
    spawned = {}

    class Proc:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout):
            return 0

    def popen(argv, **kwargs):
        spawned["env"] = kwargs["env"]
        return Proc()

    class Health:
        status_code = 200

    monkeypatch.setenv("ZAI_API_KEY", "production-key")
    monkeypatch.setenv(mcp_writer.GATE_KEY_ENV, "benchmark-key")
    monkeypatch.setattr(mcp_writer.subprocess, "Popen", popen)
    monkeypatch.setattr(mcp_writer.httpx, "get", lambda *a, **k: Health())
    with mcp_writer.EvalApi():
        pass
    assert spawned["env"]["ZAI_API_KEY"] == "benchmark-key"


def test_questions_run_concurrently_up_to_the_limit(monkeypatch, tmp_path):
    in_flight, peak, done = 0, 0, []

    async def write_question(question, turns, writer, api_url, data_dir):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        done.append(question["question_id"])

    monkeypatch.setattr(mcp_writer, "write_question", write_question)
    monkeypatch.setattr(mcp_writer, "session_turns", lambda questions: {})
    monkeypatch.setattr(mcp_writer, "session_units", lambda question: [])
    questions = [{"question_id": f"q{i}"} for i in range(5)]
    args = mcp_writer.build_parser().parse_args(
        ["--dataset", "d.json", "--data-dir", str(tmp_path), "--concurrency", "2"]
    )
    asyncio.run(mcp_writer._write_all(questions, args, "http://127.0.0.1:1"))
    assert peak == 2
    assert sorted(done) == [f"q{i}" for i in range(5)]
