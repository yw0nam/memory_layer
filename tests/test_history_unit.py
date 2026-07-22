"""Unit tests for the pure functions in src/history_index.py not already
covered by tests/test_history_parse.py: mean_idf, tokenize, build_transcript,
group_sessions. No DB/LLM/embedding access.
"""

from __future__ import annotations

import math

import pytest

from memory_base.ingest.history import (
    TRUNCATION_MARKER,
    Message,
    build_transcript,
    group_sessions,
    mean_idf,
    tokenize,
)


# ---- mean_idf -------------------------------------------------------------


def test_mean_idf_known_small_corpus():
    dfs = {"apple": 1, "banana": 2}
    document_count = 3
    result = mean_idf("apple banana", document_count, dfs)
    expected = (
        (math.log((document_count + 1) / (dfs["apple"] + 1)) + 1)
        + (math.log((document_count + 1) / (dfs["banana"] + 1)) + 1)
    ) / 2
    assert result == pytest.approx(expected)


def test_mean_idf_unseen_token_defaults_df_to_zero():
    result = mean_idf("unseen_token", document_count=5, dfs={})
    expected = math.log((5 + 1) / (0 + 1)) + 1
    assert result == pytest.approx(expected)


def test_mean_idf_empty_text_returns_zero():
    assert mean_idf("", document_count=10, dfs={"x": 1}) == 0.0


def test_mean_idf_whitespace_and_symbols_only_returns_zero():
    assert mean_idf("   !! ?? ", document_count=10, dfs={}) == 0.0


# ---- tokenize ---------------------------------------------------------


def test_tokenize_extracts_snake_case_identifier():
    tokens = tokenize("some_snake_case value")
    assert "some_snake_case" in tokens


def test_tokenize_lowercases_camelcase_identifier():
    tokens = tokenize("FooBarBaz")
    # regex matches the whole identifier as one token; text is lowered first
    assert "foobarbaz" in tokens
    assert "FooBarBaz" not in tokens


def test_tokenize_excludes_single_char_identifiers_and_symbols():
    tokens = tokenize("a I x ! @ # $ 123")
    assert tokens == set()


def test_tokenize_extracts_korean_two_plus_chars_excludes_one_char():
    tokens = tokenize("가 나다 라마바")
    assert "가" not in tokens
    assert "나다" in tokens
    assert "라마바" in tokens


# ---- build_transcript -----------------------------------------------------


def test_build_transcript_no_truncation_under_limit():
    msgs = [Message(role="user", text="hello", timestamp=0.0, session_id="s1")]
    transcript = build_transcript(msgs)
    assert transcript == "USER: hello"
    assert TRUNCATION_MARKER not in transcript


def test_build_transcript_truncates_head_60_tail_40_over_limit():
    long_text = "x" * 200_000
    msgs = [Message(role="user", text=long_text, timestamp=0.0, session_id="s1")]
    transcript = build_transcript(msgs, max_chars=100_000)

    assert TRUNCATION_MARKER in transcript
    head, _, tail = transcript.partition(TRUNCATION_MARKER)
    full = f"USER: {long_text}"
    assert head == full[: int(100_000 * 0.6)]
    assert tail == full[-int(100_000 * 0.4) :]
    assert len(head) == 60_000
    assert len(tail) == 40_000


# ---- group_sessions ---------------------------------------------------


def test_group_sessions_groups_by_session_id():
    msgs = [
        Message(role="user", text="a", timestamp=100.0, session_id="s1"),
        Message(role="assistant", text="b", timestamp=200.0, session_id="s1"),
        Message(role="user", text="c", timestamp=50.0, session_id="s2"),
    ]
    sessions = group_sessions(msgs)
    by_id = {s.session_id: s for s in sessions}
    assert set(by_id) == {"s1", "s2"}
    assert len(by_id["s1"].messages) == 2
    assert len(by_id["s2"].messages) == 1


def test_group_sessions_ts_last_active_is_max_timestamp():
    msgs = [
        Message(role="user", text="a", timestamp=100.0, session_id="s1"),
        Message(role="assistant", text="b", timestamp=200.0, session_id="s1"),
        Message(role="user", text="c", timestamp=150.0, session_id="s1"),
    ]
    sessions = group_sessions(msgs)
    assert sessions[0].ts_last_active == 200.0


def test_group_sessions_fallback_timestamp_when_all_none():
    msgs = [Message(role="user", text="a", timestamp=None, session_id="s1")]
    sessions = group_sessions(msgs, fallback_timestamp=999.0)
    assert sessions[0].ts_last_active == 999.0


def test_group_sessions_ignores_none_timestamps_when_computing_max():
    msgs = [
        Message(role="user", text="a", timestamp=None, session_id="s1"),
        Message(role="assistant", text="b", timestamp=42.0, session_id="s1"),
    ]
    sessions = group_sessions(msgs, fallback_timestamp=0.0)
    assert sessions[0].ts_last_active == 42.0
