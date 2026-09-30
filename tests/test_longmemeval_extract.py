"""Unit coverage for the LongMemEval note extractor (fake model client, fake gate)."""

from __future__ import annotations

import asyncio
import json

import pytest
from longmemeval import extract

from memory_base.serve.notes import ContentVerdict

DATE_A = "2023/05/20 (Sat) 02:21"
DATE_B = "2023/05/22 (Mon) 09:00"


class FakeClient:
    temperature = 0
    thinking = "disabled"

    def __init__(self, replies=None, failures=0):
        self.calls = []
        self.replies = replies or {}
        self.failures = failures

    async def complete(self, messages):
        self.calls.append(messages[-1]["content"])
        if self.failures:
            self.failures -= 1
            return "not json", 10, 1
        for marker, notes in self.replies.items():
            if marker in messages[-1]["content"]:
                return json.dumps({"notes": notes}), 100, 20
        return json.dumps({"notes": []}), 100, 2


async def accept(content, kind):
    return ContentVerdict(accepted="chatter" not in content, reason="judged")


def unit(sid, date, text):
    return (sid, date, [{"role": "user", "content": text}, {"role": "assistant", "content": "ok"}])


def run(units, data_dir, client, gate=accept):
    return asyncio.run(
        extract.extract_units(units, data_dir, client=client, gate=gate, concurrency=2)
    )


def test_notes_and_session_records_are_written_per_unit(tmp_path):
    client = FakeClient(
        {
            "bike": [
                {"content": "The user owns a red bike.", "kind": "note"},
                {"content": "Some chatter.", "kind": "note"},
            ]
        }
    )
    summary = run([unit("s1", DATE_A, "my bike"), unit("s2", DATE_A, "hello")], tmp_path, client)
    notes = extract.read_notes(tmp_path)
    sessions = extract.read_sessions(tmp_path)
    assert {(n["session_id"], n["gate"]) for n in notes} == {("s1", "stored"), ("s1", "refused")}
    assert all(n["date"] == DATE_A and n["gate_reason"] == "judged" for n in notes)
    assert {(s["session_id"], s["notes"]) for s in sessions} == {("s1", 2), ("s2", 0)}
    assert all(s["in_tok"] == 100 and s["seconds"] >= 0 for s in sessions)
    assert summary["completed"] == 2
    assert summary["failed"] == 0


def test_resume_skips_units_already_completed_including_zero_note_ones(tmp_path):
    run([unit("s1", DATE_A, "hello")], tmp_path, FakeClient())
    client = FakeClient()
    run([unit("s1", DATE_A, "hello"), unit("s2", DATE_A, "hi")], tmp_path, client)
    assert len(client.calls) == 1
    assert "hi" in client.calls[0]
    assert len(extract.read_sessions(tmp_path)) == 2


def test_the_extraction_unit_is_the_session_and_date_pair(tmp_path):
    client = FakeClient()
    run([unit("s1", DATE_A, "hello"), unit("s1", DATE_B, "hello")], tmp_path, client)
    assert len(client.calls) == 2
    assert {s["date"] for s in extract.read_sessions(tmp_path)} == {DATE_A, DATE_B}
    assert DATE_A in client.calls[0] or DATE_A in client.calls[1]


def test_resume_discards_a_partial_line_and_notes_of_uncompleted_units(tmp_path):
    notes_path = tmp_path / extract.NOTES_FILE
    sessions_path = tmp_path / extract.SESSIONS_FILE
    done = {"session_id": "s1", "date": DATE_A, "notes": 1}
    kept = {"session_id": "s1", "date": DATE_A, "content": "kept", "kind": "note"}
    orphan = {"session_id": "s2", "date": DATE_A, "content": "orphan", "kind": "note"}
    notes_path.write_text(json.dumps(kept) + "\n" + json.dumps(orphan) + "\n" + '{"session_')
    sessions_path.write_text(json.dumps(done) + "\n" + '{"session_id": "s2", "da')
    assert extract.prepare_resume(tmp_path) == {("s1", DATE_A)}
    assert notes_path.read_text() == json.dumps(kept) + "\n"
    assert sessions_path.read_text() == json.dumps(done) + "\n"


def test_gate_failures_are_retried_and_never_recorded_as_a_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "RETRY_BACKOFF_SECONDS", 0)
    attempts = []

    async def flaky(content, kind):
        attempts.append(content)
        if len(attempts) < 3:
            raise TimeoutError("gate down")
        return ContentVerdict(accepted=True, reason="fine")

    client = FakeClient({"bike": [{"content": "The user owns a red bike.", "kind": "note"}]})
    run([unit("s1", DATE_A, "bike")], tmp_path, client, gate=flaky)
    assert len(attempts) == 3
    [note] = extract.read_notes(tmp_path)
    assert note["gate"] == "stored"
    [session] = extract.read_sessions(tmp_path)
    assert session["gate_retries"] == 2


def test_a_unit_whose_gate_stays_down_is_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "RETRY_BACKOFF_SECONDS", 0)

    async def down(content, kind):
        raise TimeoutError("gate down")

    client = FakeClient({"bike": [{"content": "The user owns a red bike.", "kind": "note"}]})
    summary = run([unit("s1", DATE_A, "bike")], tmp_path, client, gate=down)
    assert summary["failed"] == 1
    assert extract.read_notes(tmp_path) == []
    assert extract.read_sessions(tmp_path) == []


def test_a_gate_content_filter_refusal_stores_the_note_unjudged_like_production(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(extract, "RETRY_BACKOFF_SECONDS", 0)
    attempts = []

    async def filtered(content, kind):
        attempts.append(content)
        raise provider_error("1301")

    client = FakeClient({"bike": [{"content": "The user owns a red bike.", "kind": "note"}]})
    summary = run([unit("s1", DATE_A, "bike")], tmp_path, client, gate=filtered)
    assert summary["failed"] == 0
    assert len(attempts) == 1
    [note] = extract.read_notes(tmp_path)
    assert note["gate"] == "stored"
    assert note["gate_reason"] == "content gate unavailable: content_filter"
    [session] = extract.read_sessions(tmp_path)
    assert (session["stored"], session["gate_retries"]) == (1, 0)


def test_with_the_gate_off_notes_are_recorded_unjudged_without_a_gate_call(tmp_path):
    client = FakeClient({"bike": [{"content": "The user owns a red bike.", "kind": "note"}]})
    run([unit("s1", DATE_A, "bike")], tmp_path, client, gate=None)
    (note,) = extract.read_notes(tmp_path)
    (session,) = extract.read_sessions(tmp_path)
    assert (note["gate"], note["gate_reason"]) == ("unjudged", "gate off")
    assert (session["stored"], session["gate_calls"]) == (0, 0)


def test_the_gate_off_flag_skips_the_gate_and_leaves_it_out_of_the_manifest(tmp_path, monkeypatch):
    from memory_base.eval import longmemeval as lme
    from memory_base.serve import notes as notes_module

    dataset = tmp_path / "dataset.json"
    qid = single_session_dataset(dataset)
    client = FakeClient({"bike": [{"content": "The user owns a red bike.", "kind": "note"}]})
    client.model, client.provider = "glm-5.3-flash", "zai"
    monkeypatch.setattr(extract.OpenAIExtractor, "from_env", lambda env, model: client)

    async def never(content, kind):
        raise AssertionError("gate called")

    monkeypatch.setattr(notes_module, "judge_note_content", never)
    manifest = tmp_path / "manifest.json"
    extract.main(
        ["--dataset", str(dataset), "--data-dir", str(tmp_path / "data"),
         "--manifest", str(manifest), "--questions", qid, "--gate", "off"]
    )  # fmt: skip
    section = lme.read_manifest(manifest)["extract"]
    assert section["gate"] is None
    assert [n["gate"] for n in extract.read_notes(tmp_path / "data")] == ["unjudged"]


def test_invalid_extractor_output_is_retried_then_parsed(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "RETRY_BACKOFF_SECONDS", 0)
    client = FakeClient({"bike": [{"content": "The user owns a red bike.", "kind": "note"}]})
    client.failures = 1
    run([unit("s1", DATE_A, "bike")], tmp_path, client)
    assert len(client.calls) == 2
    [session] = extract.read_sessions(tmp_path)
    assert session["extract_retries"] == 1


def test_notes_the_save_path_would_refuse_are_refused_without_a_gate_call(tmp_path):
    gated = []

    async def gate(content, kind):
        gated.append(content)
        return ContentVerdict(accepted=True, reason="fine")

    token = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
    client = FakeClient(
        {
            "bike": [
                {"content": "x" * 4001, "kind": "note"},
                {"content": "The user rides.", "kind": "fact"},
                {"content": f"The user's token is {token}.", "kind": "note"},
                {"content": "  ", "kind": "note"},
            ]
        }
    )
    run([unit("s1", DATE_A, "bike")], tmp_path, client, gate=gate)
    notes = extract.read_notes(tmp_path)
    assert gated == []
    assert [n["gate"] for n in notes] == ["refused", "refused", "refused"]
    assert notes[0]["gate_reason"].startswith("validation:")
    assert notes[1]["gate_reason"].startswith("validation:")
    assert notes[2]["gate_reason"].startswith("credential:")


def test_parse_extraction_rejects_a_reply_without_a_notes_list():
    with pytest.raises(ValueError):
        extract.parse_extraction('{"facts": []}')
    assert extract.parse_extraction('{"notes": [{"content": "a", "kind": "note"}]}') == [
        {"content": "a", "kind": "note"}
    ]


def test_render_session_lists_turns_by_role():
    turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert extract.render_session(turns) == "user: hi\nassistant: hello"


def test_prompt_carries_the_session_date_and_text():
    messages = extract.build_messages(DATE_A, [{"role": "user", "content": "my bike"}])
    assert DATE_A in messages[-1]["content"]
    assert "user: my bike" in messages[-1]["content"]
    assert messages[0]["role"] == "system"


def test_the_extractor_requires_the_zai_provider():
    with pytest.raises(RuntimeError):
        extract.OpenAIExtractor.from_env({"OPENAI_API_KEY": "k"}, model="m")
    client = extract.OpenAIExtractor.from_env({"ZAI_API_KEY": "k"}, model="glm-5.3-flash")
    assert client.model == "glm-5.3-flash"


def revisions_captured_in_order(monkeypatch):
    from memory_base.eval import longmemeval as lme

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


def single_session_dataset(path):
    questions = [
        {
            "question_id": f"q{index:03d}",
            "question_type": "multi-session",
            "question": "q?",
            "question_date": "2023/06/01 (Thu) 10:00",
            "answer": "a",
            "answer_session_ids": [f"s{index}"],
            "haystack_session_ids": [f"s{index}"],
            "haystack_dates": [DATE_A],
            "haystack_sessions": [[{"role": "user", "content": "my bike"}]],
        }
        for index in range(500)
    ]
    path.write_text(json.dumps(questions))
    from memory_base.eval import longmemeval as lme

    return lme.select_subset(questions)[0]["question_id"]


def test_the_manifest_records_the_code_revision_from_the_start_of_the_run(tmp_path, monkeypatch):
    from memory_base.eval import longmemeval as lme
    from memory_base.serve import notes as notes_module

    dataset = tmp_path / "dataset.json"
    qid = single_session_dataset(dataset)
    client = FakeClient({"bike": [{"content": "The user owns a red bike.", "kind": "note"}]})
    client.model, client.provider = "glm-5.3-flash", "zai"
    monkeypatch.setattr(extract.OpenAIExtractor, "from_env", lambda env, model: client)
    monkeypatch.setattr(extract, "record_gate_usage", lambda: None)
    monkeypatch.setattr(notes_module, "judge_note_content", accept)
    revisions_captured_in_order(monkeypatch)
    manifest = tmp_path / "manifest.json"
    extract.main(
        ["--dataset", str(dataset), "--data-dir", str(tmp_path / "data"),
         "--manifest", str(manifest), "--questions", qid]
    )  # fmt: skip
    assert lme.read_manifest(manifest)["extract"]["code"] == {"commit": "start", "dirty": False}


def provider_error(code):
    import httpx
    import openai

    response = httpx.Response(400, request=httpx.Request("POST", "http://zai.test"))
    body = {"code": code, "message": "System detected potentially unsafe content."}
    return openai.BadRequestError(f"Error code: 400 - {body}", response=response, body=body)


class RaisingClient:
    temperature = 0
    thinking = "disabled"

    model = "glm-5.3-flash"
    provider = "zai"

    def __init__(self, error):
        self.error = error
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        raise self.error


def test_a_content_filter_refusal_completes_the_unit_with_no_notes_and_no_retry(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(extract, "RETRY_BACKOFF_SECONDS", 0)
    client = RaisingClient(provider_error("1301"))
    summary = run([unit("s1", DATE_A, "hello")], tmp_path, client)
    assert client.calls == 1
    assert summary["completed"] == 1
    assert summary["failed"] == 0
    assert extract.read_notes(tmp_path) == []
    [session] = extract.read_sessions(tmp_path)
    assert session["provider_refused"] == "content_filter"
    assert (session["session_id"], session["date"], session["notes"]) == ("s1", DATE_A, 0)
    assert all(session[name] == 0 for name in ("in_tok", "out_tok", "gate_calls"))
    assert extract.prepare_resume(tmp_path) == {("s1", DATE_A)}


def test_any_other_provider_error_stays_a_retried_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "RETRY_BACKOFF_SECONDS", 0)
    client = RaisingClient(provider_error("1214"))
    summary = run([unit("s1", DATE_A, "hello")], tmp_path, client)
    assert client.calls == extract.EXTRACT_ATTEMPTS
    assert summary["failed"] == 1
    assert extract.read_sessions(tmp_path) == []


def test_the_extract_manifest_counts_provider_refused_units(tmp_path, monkeypatch):
    import argparse

    monkeypatch.setattr(extract, "RETRY_BACKOFF_SECONDS", 0)
    run([unit("s1", DATE_A, "hello")], tmp_path, RaisingClient(provider_error("1301")))
    run([unit("s2", DATE_A, "hello")], tmp_path, FakeClient())
    selected = [
        {
            "question_id": "q1",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": [DATE_A, DATE_A],
            "haystack_sessions": [[], []],
        }
    ]
    args = argparse.Namespace(
        data_dir=tmp_path, questions=None, concurrency=5, prompt="agent", gate="on"
    )
    client = RaisingClient(None)
    manifest = extract._extract_manifest(
        args, selected, client, {"failed": 0}, {"commit": "c", "dirty": False}
    )
    assert manifest["units"] == {
        "selected": 2,
        "completed": 2,
        "provider_refused": 1,
        "failed_this_run": 0,
    }


def test_the_digest_prompt_carries_the_session_and_is_pinned_apart_from_the_agent_prompt():
    turns = [{"role": "user", "content": "my bike"}]
    messages = extract.build_messages(DATE_A, turns, prompt="digest")
    assert DATE_A in messages[1]["content"] and "user: my bike" in messages[1]["content"]
    assert "Storing nothing is the correct and common outcome" in messages[1]["content"]
    assert messages[1]["content"] != extract.build_messages(DATE_A, turns)[1]["content"]
    assert extract.prompt_sha256("digest") != extract.prompt_sha256("agent")


def test_units_are_extracted_with_the_selected_prompt(tmp_path):
    client = FakeClient()
    asyncio.run(
        extract.extract_units(
            [unit("s1", DATE_A, "hi")], tmp_path, client=client, gate=accept, prompt="digest"
        )
    )
    assert len(client.calls) == 1 and "Storing nothing" in client.calls[0]


def test_the_claude_code_extractor_sends_the_system_and_user_turns_and_strips_a_fence(
    monkeypatch,
):
    from longmemeval import answer

    seen = {}

    async def complete(self, prompt, *, max_tokens):
        seen["system"], seen["prompt"] = self.system_prompt, prompt
        return '```json\n{"notes": []}\n```', 10, 2

    monkeypatch.setattr(answer.ClaudeCodeModel, "complete", complete)
    client = extract.ClaudeCodeExtractor(model="claude-sonnet-5-5", effort="high")
    messages = extract.build_messages(DATE_A, [{"role": "user", "content": "hi"}])
    text, in_tok, out_tok = asyncio.run(client.complete(messages))
    assert extract.parse_extraction(text) == []
    assert (seen["system"], seen["prompt"]) == (messages[0]["content"], messages[1]["content"])
    assert (in_tok, out_tok) == (10, 2)


def test_the_personal_prompt_carries_the_personal_policy_without_the_store_nothing_default():
    turns = [{"role": "user", "content": "my bike"}]
    content = extract.build_messages(DATE_A, turns, prompt="personal")[1]["content"]
    assert DATE_A in content and "user: my bike" in content
    assert "Personal memory policy" in content
    assert "Storing nothing" not in content
    assert len({extract.prompt_sha256(p) for p in ("agent", "digest", "personal")}) == 3


def test_the_digest_and_personal_prompts_are_the_packaged_extraction_prompts():
    from memory_base.eval import extraction

    assert extract.PROMPTS["digest"] == extraction.load_prompt("digest")
    assert extract.PROMPTS["personal"] == extraction.load_prompt("personal")
    assert extract.parse_extraction is extraction.parse_extraction
    messages = extract.build_messages(DATE_A, [{"role": "user", "content": "my bike"}], "digest")
    assert messages[0]["content"] == extraction.EXTRACTION_SYSTEM_PROMPT
    assert "[0] user: my bike" in messages[1]["content"]


def test_the_packaged_prompt_shas_are_pinned():
    assert extract.prompt_sha256("digest") == (
        "d36c16c1e5f1e23450d550ceacb5503edcdbf01f473f350cd2f98763261d24c1"
    )
    assert extract.prompt_sha256("personal") == (
        "9609641776fd4d3c014da668bbb45c83ec12a74206957fb692c9b00f200e9c92"
    )
