"""Pins the source-adapter contract in ``memory_base.adapters``. No DB/LLM/embedding
access.
"""

from __future__ import annotations

import pytest

from memory_base.adapters.base import Message, Session, SourceAdapter
from memory_base.ingest.history import build_rows, build_transcript, parse_distillation

SESSION_ID = "fake-session-1"

FAKE_MESSAGES = [
    Message(
        role="user",
        text=(
            "Why does the burst gate reject short assistant replies even when "
            "they contain a stack trace?"
        ),
        timestamp=1_700_000_000.0,
        session_id=SESSION_ID,
    ),
    Message(
        role="assistant",
        text=(
            "The gate requires at least 200 chars per burst before scoring, so "
            "short stack traces without surrounding prose get filtered out even "
            "when they are useful. Consider lowering min_chars or adding an "
            "error-signal bypass for bursts with a tool_error."
        ),
        timestamp=1_700_000_010.0,
        session_id=SESSION_ID,
        tool_names=("Read",),
    ),
    Message(
        role="user",
        text="That matches what I am seeing. Should the bypass check for the word Traceback specifically?",
        timestamp=1_700_000_020.0,
        session_id=SESSION_ID,
    ),
    Message(
        role="assistant",
        text=(
            "Checking for 'Traceback' plus a nonzero tool_error flag is a "
            "reasonable minimal signal. It avoids a broad keyword list while "
            "catching the common Python case, and keeps the heuristic explainable."
        ),
        timestamp=1_700_000_030.0,
        session_id=SESSION_ID,
    ),
    Message(
        role="user",
        text="Agreed, let's ship that as a follow-up once the adapter refactor lands.",
        timestamp=1_700_000_040.0,
        session_id=SESSION_ID,
    ),
]


def _fake_session() -> Session:
    assert len(FAKE_MESSAGES) >= 5
    transcript = build_transcript(FAKE_MESSAGES)
    assert len(transcript) >= 500
    return Session(
        session_id=SESSION_ID,
        messages=FAKE_MESSAGES,
        transcript=transcript,
        ts_last_active=FAKE_MESSAGES[-1].timestamp,
        tool_names=[name for m in FAKE_MESSAGES for name in m.tool_names],
        tool_error_count=sum(m.tool_error for m in FAKE_MESSAGES),
    )


class FakeAdapter(SourceAdapter):
    source_type = "fake"

    def discover(self):
        return []

    def parse(self, text, file):
        return [_fake_session()]

    def has_social(self, burst):
        return False


class _MissingHasSocialAdapter(SourceAdapter):
    source_type = "incomplete"

    def discover(self):
        return []

    def parse(self, text, file):
        return []

    # has_social intentionally not implemented


# ---- contract shape ---------------------------------------------------


def test_incomplete_adapter_raises_type_error_on_instantiation():
    with pytest.raises(TypeError):
        _MissingHasSocialAdapter()


def test_fake_adapter_satisfies_contract():
    adapter = FakeAdapter()
    assert adapter.source_type == "fake"
    assert adapter.discover() == []
    assert adapter.has_social(burst=None) is False
    sessions = adapter.parse("", file=None)
    assert len(sessions) == 1
    assert sessions[0].session_id == SESSION_ID


# ---- source-agnostic core: build_rows ----------------------------------


def test_build_rows_is_source_agnostic_and_pure():
    session = _fake_session()
    distillation = parse_distillation(
        {
            "one_line_question": "How should the burst gate treat short error bursts?",
            "summary": "Discussed lowering min_chars or bypassing on tool_error for short stack traces.",
            "resolution": "Bypass planned as a follow-up once the adapter refactor lands.",
            "references": ["ingest/history.py"],
            "decisions": [
                "Use Traceback text plus tool_error as the bypass signal for the burst gate."
            ],
        }
    )

    rows = build_rows(session, distillation, FakeAdapter(), dfs={}, n=1)

    assert rows
    session_rows = [row for row in rows if row["id"].endswith(":session")]
    assert len(session_rows) == 1
    session_row = session_rows[0]

    # build_rows is pure (no network/DB): there is no computed "embedding"
    # key here — embedding is added later by a separate, effectful step.
    # source_type labels rows without hardcoding any specific source.
    assert set(session_row) == {
        "id",
        "source_ref",
        "kind",
        "session_id",
        "raw",
        "distilled",
        "timestamp",
        "idf",
        "metadata",
        "source_type",
    }
    assert session_row["source_type"] == "fake"
    assert session_row["session_id"] == SESSION_ID
    assert session_row["kind"] == "session"
    assert session_row["distilled"] == distillation.text
    assert session_row["raw"] == session.transcript

    assert all(row["source_type"] == "fake" for row in rows)
    assert all("embedding" not in row for row in rows)
    assert "claude_code" not in repr(rows)


# ---- registry -----------------------------------------------------------


def test_registry_is_empty_awaiting_future_corpus_adapters():
    from memory_base.adapters import ADAPTERS

    assert ADAPTERS == {}


# ---- emit_bursts toggle -------------------------------------------------


class NoBurstAdapter(FakeAdapter):
    emit_bursts = False


def _burst_distillation():
    return parse_distillation(
        {
            "one_line_question": "How should the burst gate treat short error bursts?",
            "summary": "Discussed lowering min_chars or bypassing on tool_error.",
            "resolution": "Bypass planned as a follow-up.",
            "references": ["ingest/history.py"],
            "decisions": ["Use Traceback text plus tool_error as the bypass signal."],
        }
    )


def test_emit_bursts_defaults_true_on_the_abc():
    assert FakeAdapter.emit_bursts is True


def test_build_rows_honors_emit_bursts_toggle():
    session = _fake_session()
    distillation = _burst_distillation()

    # n < 20 makes passes_burst_gate return True unconditionally, so these
    # bursts genuinely pass the gate under the default (emit_bursts) adapter.
    emitted = build_rows(session, distillation, FakeAdapter(), dfs={}, n=1)
    assert any(row["kind"] == "burst" for row in emitted)

    suppressed = build_rows(session, distillation, NoBurstAdapter(), dfs={}, n=1)
    assert not any(row["kind"] == "burst" for row in suppressed)
    assert {row["kind"] for row in suppressed} == {"session", "decision"}
