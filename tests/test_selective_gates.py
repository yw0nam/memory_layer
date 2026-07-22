"""TDD red state for two rounds of history_index.py gate work:

Issue #7 (selective ingest: build_corpus_df / triage_heuristic / the
original strict-AND passes_burst_gate) is implemented and green.

Issue #9 (this round) replaces the strict-AND burst gate with a weighted
combination (score = mean_idf + 1.0-if-social >= 4.0) and adds decision
extraction (parse_distillation / Distillation.decisions). Neither
`burst_score` nor `parse_distillation` exist in history_index.py yet, so
importing them below raises ImportError -- that is the expected "red"
state for this step, and it also means every test in this module reports
as a collection error (not just the gate/decision ones) until Step B adds
both names. Step B should turn this file green without weakening the
assertions below; passes_burst_gate's *behavior* has also changed (see the
inverted/replaced cases in the gate section), so Step B must update its
implementation, not just its signature.

DB/LLM/network-free by design.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from history_index import (  # noqa: E402
    Burst,
    Distillation,
    Message,
    Session,
    build_corpus_df,
    burst_score,
    parse_distillation,
    passes_burst_gate,
    triage_heuristic,
)


def _msg(role: str, text: str, ts: float = 0.0, session_id: str = "s1") -> Message:
    return Message(role=role, text=text, timestamp=ts, session_id=session_id)


def _session(messages: list[Message], session_id: str = "s1") -> Session:
    return Session(session_id=session_id, messages=messages, transcript="", ts_last_active=0.0)


def _burst(social_weight: float, text: str = "irrelevant") -> Burst:
    return Burst(role="assistant", text=text, messages=[], social_weight=social_weight)


# ---- build_corpus_df -------------------------------------------------------
# build_corpus_df(transcripts: Iterable[str]) -> tuple[dict[str, int], int]
# df = count of transcripts whose *unique* token set contains the token.
# N = number of transcripts passed in.


def test_build_corpus_df_hand_computed_small_corpus():
    transcripts = ["apple banana", "apple cherry", "cherry cherry"]
    dfs, n = build_corpus_df(transcripts)
    assert n == 3
    assert dfs["apple"] == 2
    assert dfs["banana"] == 1
    assert dfs["cherry"] == 2


def test_build_corpus_df_empty_iterable():
    dfs, n = build_corpus_df([])
    assert dfs == {}
    assert n == 0


def test_build_corpus_df_counts_n_even_for_transcript_with_no_tokens():
    dfs, n = build_corpus_df(["", "apple"])
    assert n == 2
    assert dfs == {"apple": 1}


def test_build_corpus_df_duplicate_tokens_within_transcript_count_once():
    dfs, n = build_corpus_df(["apple apple apple banana"])
    assert n == 1
    assert dfs["apple"] == 1
    assert dfs["banana"] == 1


# ---- triage_heuristic -------------------------------------------------------
# triage_heuristic(session: Session) -> "keep" | "skip" | "borderline"
# Priority order (first match wins):
#   1. zero assistant text messages -> "skip"
#   2. <=2 user messages AND user text total < 200 chars -> "skip"
#   3. total text length >= 5000 chars OR message count >= 20 -> "keep"
#   4. otherwise -> "borderline"


def test_triage_heuristic_skip_when_no_assistant_messages():
    # Would satisfy rule 3 on length alone, but rule 1 (no assistant) wins.
    messages = [_msg("user", "x" * 6000)]
    assert triage_heuristic(_session(messages)) == "skip"


def test_triage_heuristic_skip_simple_command_session():
    messages = [_msg("user", "run tests"), _msg("assistant", "ok, running now")]
    assert triage_heuristic(_session(messages)) == "skip"


def test_triage_heuristic_skip_with_exactly_two_short_user_messages():
    messages = [_msg("user", "hi"), _msg("user", "there"), _msg("assistant", "hello!")]
    assert triage_heuristic(_session(messages)) == "skip"


def test_triage_heuristic_rule2_takes_precedence_over_rule3():
    # user side is short (rule 2 fires), even though total length (via a
    # huge assistant reply) would otherwise satisfy rule 3's >=5000 keep.
    messages = [_msg("user", "short question"), _msg("assistant", "y" * 6000)]
    assert triage_heuristic(_session(messages)) == "skip"


def test_triage_heuristic_user_count_over_two_bypasses_rule2():
    messages = [
        _msg("user", "hi"),
        _msg("user", "ok"),
        _msg("user", "go"),
        _msg("assistant", "sure"),
    ]
    # 3 user messages (>2) -> rule 2 doesn't apply regardless of char sum.
    # total chars tiny, message count 4 (<20) -> falls through to borderline.
    assert triage_heuristic(_session(messages)) == "borderline"


def test_triage_heuristic_user_chars_at_200_boundary_not_skip():
    messages = [_msg("user", "u" * 200), _msg("assistant", "a")]
    # exactly 200 chars is NOT < 200 -> rule 2 does not fire.
    assert triage_heuristic(_session(messages)) == "borderline"


def test_triage_heuristic_keep_when_total_length_at_least_5000():
    messages = [_msg("user", "a" * 3000), _msg("assistant", "b" * 2000)]
    # total == 5000 (boundary, >=5000) and user_chars=3000 avoids rule 2.
    assert triage_heuristic(_session(messages)) == "keep"


def test_triage_heuristic_keep_when_message_count_at_least_20():
    messages = [_msg("user", "hi", ts=i) for i in range(19)] + [_msg("assistant", "hello")]
    # 19 user (>2, so rule 2 doesn't apply) + 1 assistant = 20 messages.
    assert triage_heuristic(_session(messages)) == "keep"


def test_triage_heuristic_borderline_when_no_rule_matches():
    messages = [_msg("user", "u" * 250), _msg("assistant", "a" * 250)]
    # user_chars=250 (not <200) avoids rule 2; total=500 (<5000) and
    # count=2 (<20) avoid rule 3 -> borderline.
    assert triage_heuristic(_session(messages)) == "borderline"


# ---- burst_score -------------------------------------------------------
# burst_score(mean_idf_value: float, has_social: bool) -> float
# score = mean_idf_value + (1.0 if has_social else 0.0)


def test_burst_score_social_signal_adds_exactly_one_point():
    with_social = burst_score(2.5, has_social=True)
    without_social = burst_score(2.5, has_social=False)
    assert with_social == pytest.approx(3.5)
    assert without_social == pytest.approx(2.5)
    assert with_social - without_social == pytest.approx(1.0)


# ---- passes_burst_gate (Issue #9: weighted sum, replaces strict AND) --------
# passes_burst_gate(burst, mean_idf_value, document_count) -> bool
#
#   if document_count < 20: return True   # bootstrap: full bypass, idf AND
#                                          # social both untrusted below N=20
#   return burst_score(mean_idf_value, burst.social_weight > 1.0) >= 4.0
#
# Below N=20, social is no longer required (this inverts the old
# strict-AND `bootstrap_bypass_still_requires_social_signal` case). Above
# N=20, IDF alone can clear the 4.0 threshold with no social signal at all
# (this inverts the old `fails_without_social_signal` case) -- the AND gate
# is gone.


def test_passes_burst_gate_idf_and_social_both_pass():
    # idf=4.5 + social(+1.0)=5.5 -- comfortably over threshold either way.
    assert passes_burst_gate(_burst(social_weight=1.5), mean_idf_value=4.5, document_count=50) is True


def test_passes_burst_gate_idf_alone_above_threshold_passes_without_social():
    # Inverts the old strict-AND "fails_without_social_signal" case: under
    # the weighted gate, IDF alone (4.5, no social bonus) already clears 4.0.
    assert passes_burst_gate(_burst(social_weight=1.0), mean_idf_value=4.5, document_count=50) is True


def test_passes_burst_gate_fails_without_sufficient_idf_when_corpus_large():
    # idf=1.0 + social(+1.0)=2.0 -- social bonus alone can't rescue a weak
    # IDF score.
    assert passes_burst_gate(_burst(social_weight=1.5), mean_idf_value=1.0, document_count=50) is False


def test_passes_burst_gate_idf_alone_reaches_threshold_at_4_0():
    assert passes_burst_gate(_burst(social_weight=1.0), mean_idf_value=4.0, document_count=50) is True


def test_passes_burst_gate_idf_alone_just_under_threshold_at_3_9_fails():
    assert passes_burst_gate(_burst(social_weight=1.0), mean_idf_value=3.9, document_count=50) is False


def test_passes_burst_gate_idf_plus_social_reaches_threshold_at_3_0():
    # 3.0 + 1.0 (social bonus) == 4.0 -> passes.
    assert passes_burst_gate(_burst(social_weight=1.5), mean_idf_value=3.0, document_count=50) is True


def test_passes_burst_gate_idf_plus_social_just_under_threshold_at_2_9_fails():
    # 2.9 + 1.0 (social bonus) == 3.9 -> still fails.
    assert passes_burst_gate(_burst(social_weight=1.5), mean_idf_value=2.9, document_count=50) is False


def test_passes_burst_gate_social_weight_exactly_1_0_gives_no_bonus():
    # social_weight must be strictly > 1.0 to count as a social signal.
    # idf=3.5 alone is under threshold; only the >1.0 case gets the +1.0
    # bonus needed to clear 4.0.
    assert passes_burst_gate(_burst(social_weight=1.0), mean_idf_value=3.5, document_count=50) is False
    assert passes_burst_gate(_burst(social_weight=1.5), mean_idf_value=3.5, document_count=50) is True


def test_passes_burst_gate_bootstrap_bypasses_idf_when_corpus_small():
    assert passes_burst_gate(_burst(social_weight=1.5), mean_idf_value=0.1, document_count=10) is True


def test_passes_burst_gate_bootstrap_bypasses_social_requirement_too():
    # Inverts the old strict-AND "bootstrap_bypass_still_requires_social_signal"
    # case: below N=20 the bypass is total -- weak IDF AND no social signal
    # still passes, since neither is trustworthy yet at this corpus size.
    assert passes_burst_gate(_burst(social_weight=1.0), mean_idf_value=0.1, document_count=10) is True


def test_passes_burst_gate_document_count_19_is_bootstrap_bypass():
    # Same (weak) inputs as the N=20 case below, differing only in N, to
    # pin the exact boundary where the bootstrap bypass stops applying.
    assert passes_burst_gate(_burst(social_weight=1.0), mean_idf_value=0.0, document_count=19) is True


def test_passes_burst_gate_document_count_20_is_not_bootstrap_bypass():
    # document_count == 20 is NOT < 20 -> weighted score is evaluated for
    # real, and 0.0 (no idf, no social) doesn't clear 4.0.
    assert passes_burst_gate(_burst(social_weight=1.0), mean_idf_value=0.0, document_count=20) is False


# ---- parse_distillation / Distillation.decisions (Issue #9) ----------------
# parse_distillation(parsed: dict) -> Distillation
#
# Extracted from _distill's inline JSON-parsing logic, plus a new
# `decisions: list[str]` field: missing key or non-list value -> [];
# each item is str()-converted, stripped, empty results dropped; capped at
# 10 entries. Distillation.text (the session-row embedding target) must
# NOT include decisions -- they are stored as separate `chunk_kind="decision"`
# rows, not duplicated into the session row.


def _base_parsed(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "one_line_question": "질문?",
        "summary": "요약입니다.",
        "resolution": "해결됨",
        "references": ["a.py"],
    }
    base.update(overrides)
    return base


def test_parse_distillation_happy_path_sets_core_fields_and_decisions():
    result = parse_distillation(_base_parsed(decisions=["X를 하기로 함"]))
    assert isinstance(result, Distillation)
    assert result.one_line_question == "질문?"
    assert result.summary == "요약입니다."
    assert result.resolution == "해결됨"
    assert result.references == ["a.py"]
    assert result.decisions == ["X를 하기로 함"]


def test_parse_distillation_missing_decisions_key_defaults_to_empty_list():
    result = parse_distillation(_base_parsed())  # no "decisions" key at all
    assert result.decisions == []


def test_parse_distillation_non_list_decisions_defaults_to_empty_list():
    result = parse_distillation(_base_parsed(decisions="이건 리스트가 아님"))
    assert result.decisions == []


def test_parse_distillation_items_are_stringified_stripped_and_emptied_out():
    result = parse_distillation(_base_parsed(decisions=[123, "  공백 정리됨  ", "", "   "]))
    assert result.decisions == ["123", "공백 정리됨"]


def test_parse_distillation_truncates_to_ten_decisions():
    decisions = [f"결정{i}" for i in range(15)]
    result = parse_distillation(_base_parsed(decisions=decisions))
    assert len(result.decisions) == 10
    assert result.decisions == decisions[:10]


def test_distillation_text_excludes_decisions():
    result = parse_distillation(_base_parsed(decisions=["세션 행에 포함되면 안 됨"]))
    assert "세션 행에 포함되면 안 됨" not in result.text
    assert result.text == "질문?\n요약입니다.\n해결됨\na.py"
