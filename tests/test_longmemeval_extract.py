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
    monkeypatch.setattr(extract, "GATE_BACKOFF_SECONDS", 0)
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
    monkeypatch.setattr(extract, "GATE_BACKOFF_SECONDS", 0)

    async def down(content, kind):
        raise TimeoutError("gate down")

    client = FakeClient({"bike": [{"content": "The user owns a red bike.", "kind": "note"}]})
    summary = run([unit("s1", DATE_A, "bike")], tmp_path, client, gate=down)
    assert summary["failed"] == 1
    assert extract.read_notes(tmp_path) == []
    assert extract.read_sessions(tmp_path) == []


def test_invalid_extractor_output_is_retried_then_parsed(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "GATE_BACKOFF_SECONDS", 0)
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
