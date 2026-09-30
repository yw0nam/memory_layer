"""Unit coverage for the LongMemEval answer/judge client (fake model client)."""

from __future__ import annotations

import asyncio

import pytest
from longmemeval import answer

from memory_base.eval import longmemeval as lme

DATE = "2023/05/20 (Sat) 02:21"


def make_question(qid, qtype="multi-session"):
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": f"question {qid}?",
        "question_date": "2023/06/01 (Thu) 10:00",
        "answer": f"reference {qid}",
        "answer_session_ids": ["s1"],
        "haystack_session_ids": ["s1"],
        "haystack_dates": [DATE],
        "haystack_sessions": [[{"role": "user", "content": "hi"}]],
    }


def make_packet(qid):
    return {
        "question_id": qid,
        "question_type": "multi-session",
        "question": f"question {qid}?",
        "question_date": "2023/06/01 (Thu) 10:00",
        "hits": [{"id": "n1", "date": DATE, "score": 0.9, "text": f"fact {qid}", "sessions": []}],
    }


class FakeModel:
    model = "glm-5.3-flash"
    thinking = "disabled"

    def __init__(self, failures=0):
        self.calls = []
        self.failures = failures

    async def complete(self, prompt, *, max_tokens):
        self.calls.append((prompt, max_tokens))
        if self.failures:
            self.failures -= 1
            raise TimeoutError("endpoint stalled")
        reply = "yes" if prompt.startswith("I will give you a question") else "an answer"
        return reply, 100, 5


def write_packets(data_dir, qids, run="baseline"):
    lme.append_jsonl(lme.packets_path(data_dir, run), [make_packet(qid) for qid in qids])


def run(stage, data_dir, questions, client, run_name="baseline"):
    prompts = answer.pending_prompts(stage, data_dir, run_name, questions)
    out = lme.stage_output_path(data_dir, stage, run_name)
    return asyncio.run(answer.run_stage(stage, prompts, out, client=client, concurrency=2))


def test_answer_stage_sends_the_upstream_prompt_and_records_usage(tmp_path):
    questions = {qid: make_question(qid) for qid in ("q1", "q2")}
    write_packets(tmp_path, questions)
    client = FakeModel()
    summary = run("answer", tmp_path, questions, client)
    expected = {lme.answer_prompt(make_packet(qid)) for qid in questions}
    assert {prompt for prompt, _ in client.calls} == expected
    assert {tokens for _, tokens in client.calls} == {answer.MAX_TOKENS["answer"]}
    rows = lme.read_jsonl(lme.stage_output_path(tmp_path, "answer", "baseline"))
    assert {row["question_id"] for row in rows} == {"q1", "q2"}
    for row in rows:
        assert row["prompt_sha256"] == lme.prompt_sha(
            lme.answer_prompt(make_packet(row["question_id"]))
        )
        assert (row["in_tok"], row["out_tok"], row["model"]) == (100, 5, "glm-5.3-flash")
        assert row["seconds"] >= 0
    assert summary == {"completed": 2, "failed": 0, "failures": []}


def test_a_rerun_only_answers_questions_without_a_row_for_their_current_prompt(tmp_path):
    questions = {qid: make_question(qid) for qid in ("q1", "q2", "q3")}
    write_packets(tmp_path, ["q1", "q2"])
    run("answer", tmp_path, questions, FakeModel())
    write_packets(tmp_path, ["q3"])
    client = FakeModel()
    run("answer", tmp_path, questions, client)
    assert [prompt for prompt, _ in client.calls] == [lme.answer_prompt(make_packet("q3"))]


def test_judge_stage_grades_each_current_answer_with_the_upstream_template(tmp_path):
    questions = {qid: make_question(qid) for qid in ("q1", "q2")}
    write_packets(tmp_path, ["q1"])
    run("answer", tmp_path, questions, FakeModel())
    client = FakeModel()
    run("judge", tmp_path, questions, client)
    assert client.calls == [
        (lme.judge_prompt(questions["q1"], "an answer"), answer.MAX_TOKENS["judge"])
    ]
    answers = lme.read_jsonl(lme.stage_output_path(tmp_path, "answer", "baseline"))
    judgments = lme.read_jsonl(lme.stage_output_path(tmp_path, "judge", "baseline"))
    report = lme.qa_accuracy(["q1"], questions, answers, judgments)
    assert report["overall"] == {"correct": 1, "judged": 1, "accuracy": 1.0}
    assert answer.pending_prompts("judge", tmp_path, "baseline", questions) == {}


def test_failed_calls_are_retried_and_a_question_that_keeps_failing_writes_nothing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(answer, "RETRY_BACKOFF_SECONDS", 0)
    questions = {"q1": make_question("q1")}
    write_packets(tmp_path, ["q1"])
    client = FakeModel(failures=2)
    summary = run("answer", tmp_path, questions, client)
    assert len(client.calls) == 3
    assert summary["completed"] == 1
    tmp_other = tmp_path / "other"
    tmp_other.mkdir()
    write_packets(tmp_other, ["q1"])
    summary = run("answer", tmp_other, questions, FakeModel(failures=answer.ATTEMPTS))
    assert summary["failed"] == 1
    assert lme.read_jsonl(lme.stage_output_path(tmp_other, "answer", "baseline")) == []


def test_an_empty_reply_is_retried_and_never_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(answer, "RETRY_BACKOFF_SECONDS", 0)
    questions = {"q1": make_question("q1")}
    write_packets(tmp_path, ["q1"])
    client = FakeModel()
    replies = iter([("", 100, 10), ("", 100, 10), ("an answer", 100, 5)])

    async def complete(prompt, *, max_tokens):
        client.calls.append((prompt, max_tokens))
        return next(replies)

    client.complete = complete
    summary = run("answer", tmp_path, questions, client)
    assert len(client.calls) == 3
    assert summary["completed"] == 1
    rows = lme.read_jsonl(lme.stage_output_path(tmp_path, "answer", "baseline"))
    assert [r["text"] for r in rows] == ["an answer"]


def test_each_run_reads_and_writes_its_own_files(tmp_path):
    questions = {"q1": make_question("q1")}
    write_packets(tmp_path, ["q1"], run="gate-off")
    run("answer", tmp_path, questions, FakeModel(), run_name="gate-off")
    assert lme.stage_output_path(tmp_path, "answer", "gate-off").exists()
    assert not lme.stage_output_path(tmp_path, "answer", "baseline").exists()


def test_the_model_client_requires_the_zai_provider():
    with pytest.raises(RuntimeError):
        answer.ChatModel.from_env({"OPENAI_API_KEY": "k"}, model="m")
    client = answer.ChatModel.from_env({"ZAI_API_KEY": "k"}, model="glm-5.3-flash")
    assert client.model == "glm-5.3-flash"


def test_the_stage_manifest_records_the_code_revision_from_the_start_of_the_run(
    tmp_path, monkeypatch
):
    import json

    questions = [make_question(f"q{index:03d}") for index in range(500)]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(questions))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_packets(data_dir, ["q001"])
    client = FakeModel()
    client.provider = "zai"
    monkeypatch.setattr(answer.ChatModel, "from_env", lambda env, model: client)
    revisions_captured_in_order(monkeypatch)
    manifest = tmp_path / "manifest.json"
    common = ["--dataset", str(dataset), "--data-dir", str(data_dir), "--manifest", str(manifest)]
    answer.main(["answer", *common])
    written = lme.read_manifest(manifest)
    assert written["answer"]["code"] == {"commit": "start", "dirty": False}
    assert written["answer"]["in_tok"] == 100
    assert "upstream" not in written


def revisions_captured_in_order(monkeypatch):
    """code_revision reports "start" until the dataset is loaded and "end" afterwards."""
    loaded = []
    load_dataset = lme.load_dataset

    def load(path):
        loaded.append(path)
        return load_dataset(path)

    def revision():
        return {"commit": "end" if loaded else "start", "dirty": False}

    monkeypatch.setattr(lme, "load_dataset", load)
    monkeypatch.setattr(lme, "code_revision", revision)


def test_the_claude_code_client_runs_a_tool_less_headless_session(monkeypatch):
    import json

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
    client = answer.ClaudeCodeModel(model="claude-sonnet-5-5", effort="high", system_prompt="S")
    assert asyncio.run(client.complete("the prompt", max_tokens=10)) == ("yes", 100, 7)
    assert sent["stdin"] == "the prompt"
    argv = sent["argv"]
    for flag, value in [
        ("--model", "claude-sonnet-5-5"),
        ("--tools", ""),
        ("--setting-sources", ""),
    ]:
        assert argv[argv.index(flag) + 1] == value
    assert "--strict-mcp-config" in argv
    assert sent["env"]["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "1"


def test_the_claude_code_client_refuses_a_reply_from_another_model_or_a_tool_turn(monkeypatch):
    import json

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
        client = answer.ClaudeCodeModel(model="claude-sonnet-5-5", effort="high", system_prompt="S")
        with pytest.raises(RuntimeError):
            asyncio.run(client.complete("p", max_tokens=10))
