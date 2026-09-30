"""Unit coverage for the LongMemEval answer/judge client (fake model client)."""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from longmemeval import answer

from memory_base.eval import hit_judge
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
    system_prompt = None

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
    monkeypatch.setattr(answer.ChatModel, "from_env", lambda env, model, system_prompt=None: client)
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


CANDIDATES = "candidates-gate-off"


def hits_packet(qid, texts):
    packet = make_packet(qid)
    packet["hits"] = [
        {"id": f"n{i}", "date": DATE, "score": 0.9 - i / 100, "text": text, "sessions": []}
        for i, text in enumerate(texts)
    ]
    return packet


def valid_reply(prompt):
    count = len(re.findall(r"^\[\d+\] ", prompt, flags=re.M))
    labels = {str(i): "useful" if i == 0 else "related" for i in range(count)}
    return json.dumps({"labels": labels, "min_prefix": 1})


class FakeJudge:
    model = "claude-sonnet-5-5"

    def __init__(self, malformed=False):
        self.prompts = []
        self.malformed = malformed

    async def complete(self, prompt, *, max_tokens):
        self.prompts.append((prompt, max_tokens))
        return ("not json" if self.malformed else valid_reply(prompt)), 50, 8


def judge_hits(data_dir, questions, client):
    return asyncio.run(answer.run_judge_hits(data_dir, CANDIDATES, questions, client, 2))


def test_judge_hits_labels_every_judged_packet_and_skips_abstention_and_empty_ones(tmp_path):
    packets = [
        hits_packet("q1", ["fact a", "fact b"]),
        hits_packet("q2_abs", ["fact c"]),
        hits_packet("q3", []),
        hits_packet("q4", [f"fact {i}" for i in range(12)]),
    ]
    lme.append_jsonl(lme.packets_path(tmp_path, CANDIDATES), packets)
    questions = {p["question_id"]: make_question(p["question_id"]) for p in packets}
    client = FakeJudge()
    summary = judge_hits(tmp_path, questions, client)
    assert summary == {"completed": 2, "failed": 0, "failures": []}
    assert {tokens for _, tokens in client.prompts} == {answer.MAX_TOKENS["judge-hits"]}
    rows = {
        r["question_id"]: r for r in lme.read_jsonl(lme.hit_judgments_path(tmp_path, CANDIDATES))
    }
    assert set(rows) == {"q1", "q4"}
    assert rows["q1"]["texts"] == ["fact a", "fact b"]
    assert rows["q1"]["labels"] == ["useful", "related"]
    assert rows["q1"]["min_prefix"] == 1
    assert rows["q1"]["run"] == CANDIDATES
    assert rows["q1"]["model"] == "claude-sonnet-5-5"
    assert (rows["q1"]["in_tok"], rows["q1"]["out_tok"]) == (50, 8)
    prompt = hit_judge.hit_judge_prompt(packets[0], questions["q1"])
    assert rows["q1"]["prompt_sha256"] == lme.prompt_sha(prompt)
    assert rows["q4"]["texts"] == [f"fact {i}" for i in range(10)]
    assert len(rows["q4"]["labels"]) == 10


def test_a_judge_hits_rerun_skips_current_rows_and_rejudges_a_changed_packet(tmp_path):
    path = lme.packets_path(tmp_path, CANDIDATES)
    lme.append_jsonl(path, [hits_packet("q1", ["fact a"]), hits_packet("q2", ["fact b"])])
    questions = {qid: make_question(qid) for qid in ("q1", "q2")}
    judge_hits(tmp_path, questions, FakeJudge())
    client = FakeJudge()
    assert judge_hits(tmp_path, questions, client)["completed"] == 0
    assert client.prompts == []
    lme.write_jsonl_atomic(path, [hits_packet("q1", ["fact a"]), hits_packet("q2", ["fact z"])])
    client = FakeJudge()
    judge_hits(tmp_path, questions, client)
    assert len(client.prompts) == 1 and "fact z" in client.prompts[0][0]


def test_judge_hits_retries_a_malformed_reply_and_records_an_error_row(tmp_path, monkeypatch):
    monkeypatch.setattr(answer, "RETRY_BACKOFF_SECONDS", 0)
    lme.append_jsonl(lme.packets_path(tmp_path, CANDIDATES), [hits_packet("q1", ["fact a"])])
    questions = {"q1": make_question("q1")}
    client = FakeJudge(malformed=True)
    summary = judge_hits(tmp_path, questions, client)
    assert len(client.prompts) == answer.ATTEMPTS
    assert summary["failed"] == 1
    [row] = lme.read_jsonl(lme.hit_judgments_path(tmp_path, CANDIDATES))
    assert row["question_id"] == "q1" and row["texts"] == ["fact a"] and row["error"]
    client = FakeJudge()
    assert judge_hits(tmp_path, questions, client)["completed"] == 1


class RecordingClaudeCode:
    made = []

    def __init__(self, model, effort, system_prompt):
        self.model, self.effort, self.system_prompt = model, effort, system_prompt
        self.provider, self.thinking = "claude-code", f"effort {effort}"
        RecordingClaudeCode.made.append(self)

    async def complete(self, prompt, *, max_tokens):
        return valid_reply(prompt), 50, 8


def test_the_judge_hits_stage_defaults_to_medium_effort_and_records_its_manifest(
    tmp_path, monkeypatch
):
    questions = [make_question(f"q{index:03d}") for index in range(500)]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(questions))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    lme.append_jsonl(lme.packets_path(data_dir, CANDIDATES), [hits_packet("q001", ["fact a"])])
    error_row = {"question_id": "q001", "run": CANDIDATES, "texts": ["fact a"], "error": "x"}
    lme.append_jsonl(lme.hit_judgments_path(data_dir, CANDIDATES), [error_row])
    RecordingClaudeCode.made.clear()
    monkeypatch.setattr(answer, "ClaudeCodeModel", RecordingClaudeCode)
    revisions_captured_in_order(monkeypatch)
    manifest = tmp_path / "manifest.json"
    common = ["--dataset", str(dataset), "--data-dir", str(data_dir), "--manifest", str(manifest)]
    common += ["--backend", "claude-code"]
    answer.main(["judge-hits", *common, "--gate", "off", "--read", "candidates"])
    made = RecordingClaudeCode.made[-1]
    assert (made.model, made.effort) == ("claude-sonnet-5-5", "medium")
    assert made.system_prompt == hit_judge.HIT_JUDGE_SYSTEM_PROMPT
    section = lme.read_manifest(manifest)[f"judge-hits-{CANDIDATES}"]
    assert section["templates_sha256"] == lme.prompt_sha(hit_judge.HIT_JUDGE_SYSTEM_PROMPT)
    assert section["thinking"] == "effort medium"
    assert (section["rows"], section["in_tok"]) == (1, 50)
    assert not lme.stage_output_path(data_dir, "judge", CANDIDATES).exists()
    answer.main(["answer", *common])
    assert RecordingClaudeCode.made[-1].effort == "high"
    assert lme.read_manifest(manifest)["answer"]["thinking"] == "effort high"


def test_the_zai_client_sends_a_system_turn_only_when_it_has_a_system_prompt():
    sent = []

    async def create(**kwargs):
        sent.append(kwargs["messages"])
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=1)
        message = SimpleNamespace(content=" ok ")
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=message)])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert asyncio.run(answer.ChatModel(fake, "m").complete("p", max_tokens=5)) == ("ok", 3, 1)
    asyncio.run(answer.ChatModel(fake, "m", system_prompt="S").complete("p", max_tokens=5))
    assert sent == [
        [{"role": "user", "content": "p"}],
        [{"role": "system", "content": "S"}, {"role": "user", "content": "p"}],
    ]
