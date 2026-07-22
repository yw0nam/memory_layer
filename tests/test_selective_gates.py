"""TDD Step A (red) for Issue #7 selective-ingest gates in src/history_index.py.

Pins down the contracts for three not-yet-implemented pure functions:
build_corpus_df, triage_heuristic, passes_burst_gate. history_index.py does
not define them yet, so importing here raises ImportError -- that failure
is the expected "red" state for this step. Step B (implementation) should
turn this file green without changing the assertions below.

DB/LLM/network-free by design.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from history_index import (  # noqa: E402
    Burst,
    Message,
    Session,
    build_corpus_df,
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


# ---- passes_burst_gate -------------------------------------------------------
# passes_burst_gate(burst, mean_idf_value, document_count) -> bool
# (document_count < 20 or mean_idf_value >= 4.0) and burst.social_weight > 1.0


def test_passes_burst_gate_idf_and_social_both_pass():
    assert passes_burst_gate(_burst(social_weight=1.5), mean_idf_value=4.5, document_count=50) is True


def test_passes_burst_gate_fails_without_social_signal():
    assert passes_burst_gate(_burst(social_weight=1.0), mean_idf_value=4.5, document_count=50) is False


def test_passes_burst_gate_fails_without_sufficient_idf_when_corpus_large():
    assert passes_burst_gate(_burst(social_weight=1.5), mean_idf_value=1.0, document_count=50) is False


def test_passes_burst_gate_bootstrap_bypasses_idf_when_corpus_small():
    assert passes_burst_gate(_burst(social_weight=1.5), mean_idf_value=0.1, document_count=10) is True


def test_passes_burst_gate_bootstrap_bypass_still_requires_social_signal():
    assert passes_burst_gate(_burst(social_weight=1.0), mean_idf_value=0.1, document_count=10) is False


def test_passes_burst_gate_document_count_boundary_20_is_not_bootstrap():
    # document_count == 20 is NOT < 20 -> idf gate applies and fails.
    assert passes_burst_gate(_burst(social_weight=1.5), mean_idf_value=1.0, document_count=20) is False


def test_passes_burst_gate_idf_boundary_4_0_passes():
    assert passes_burst_gate(_burst(social_weight=1.5), mean_idf_value=4.0, document_count=50) is True


def test_passes_burst_gate_social_weight_boundary_1_0_fails():
    # social_weight must be strictly > 1.0.
    assert passes_burst_gate(_burst(social_weight=1.0), mean_idf_value=10.0, document_count=50) is False
