"""Unit tests for the pure functions in src/search.py: no DB/network access.

_rrf_fuse, _apply_time_decay, and _dedup_cap operate purely on Hit objects /
plain dicts, so they're exercised directly with hand-built inputs.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from memory_base.common import vector_literal
from memory_base.retrieval.search import (
    FUSED_TOP,
    PER_FILE_CAP,
    RRF_K,
    Hit,
    _apply_time_decay,
    _dedup_cap,
    _rrf_fuse,
)


# ---- _rrf_fuse ---------------------------------------------------------


def test_rrf_fuse_formula_k60():
    # doc "a" ranked 1st in both lists; "b"/"c" ranked 2nd in only one.
    scores = _rrf_fuse([["a", "b"], ["a", "c"]])
    assert scores["a"] == pytest.approx(2 / (RRF_K + 1))
    assert scores["b"] == pytest.approx(1 / (RRF_K + 2))
    assert scores["c"] == pytest.approx(1 / (RRF_K + 2))


def test_rrf_fuse_consensus_beats_single_strong_vote():
    # "consensus" appears 2nd in both lists; "x"/"y" appear 1st in only one
    # each. Two moderate votes should outscore one strong vote.
    scores = _rrf_fuse([["x", "consensus"], ["y", "consensus"]])
    single_strong_vote = scores["x"]
    consensus = scores["consensus"]
    assert consensus > single_strong_vote


def test_rrf_fuse_empty_lists_yields_empty_scores():
    assert _rrf_fuse([]) == {}
    assert _rrf_fuse([[], []]) == {}


# ---- _apply_time_decay --------------------------------------------------


def test_time_decay_half_life_90_days():
    now = time.time()
    fresh = Hit(source="code", ref="a", text="", ts=now, rrf=1.0)
    ninety_days_old = Hit(source="code", ref="b", text="", ts=now - 90 * 86400, rrf=1.0)
    _apply_time_decay([fresh, ninety_days_old])
    assert fresh.rrf == pytest.approx(1.0, rel=1e-3)
    assert ninety_days_old.rrf == pytest.approx(0.5, rel=1e-3)


def test_time_decay_age_zero_means_no_decay():
    now = time.time()
    hit = Hit(source="code", ref="a", text="", ts=now, rrf=0.7)
    _apply_time_decay([hit])
    assert hit.rrf == pytest.approx(0.7, rel=1e-3)


def test_time_decay_180_days_quarters_score():
    now = time.time()
    hit = Hit(source="code", ref="a", text="", ts=now - 180 * 86400, rrf=1.0)
    _apply_time_decay([hit])
    assert hit.rrf == pytest.approx(0.25, rel=1e-3)


# ---- _dedup_cap ----------------------------------------------------------


def _hit(ref, rrf, filename=None):
    meta = {"filename": filename} if filename else {}
    return Hit(source="code", ref=ref, text="", ts=0.0, rrf=rrf, meta=meta)


def test_dedup_cap_caps_same_file_to_per_file_cap():
    hits = [_hit(f"a.py:L{i}", rrf=float(10 - i), filename="a.py") for i in range(5)]
    out = _dedup_cap(hits)
    assert len(out) == PER_FILE_CAP == 3
    # keeps the highest-rrf chunks for that file
    assert [h.rrf for h in out] == [10.0, 9.0, 8.0]


def test_dedup_cap_overall_fused_top_cap():
    hits = [_hit(f"f{i}.py:L1", rrf=float(30 - i), filename=f"f{i}.py") for i in range(30)]
    out = _dedup_cap(hits)
    assert len(out) == FUSED_TOP == 20


def test_dedup_cap_keeps_rrf_descending_order():
    hits = [
        _hit("b.py:L1", rrf=0.3, filename="b.py"),
        _hit("a.py:L1", rrf=0.9, filename="a.py"),
        _hit("c.py:L1", rrf=0.6, filename="c.py"),
    ]
    out = _dedup_cap(hits)
    assert [h.rrf for h in out] == [0.9, 0.6, 0.3]


def test_dedup_cap_falls_back_to_ref_when_no_filename_meta():
    # history hits have no meta["filename"] -> dedup key falls back to ref
    hits = [Hit(source="history", ref="sess-1", text="", ts=0.0, rrf=1.0, meta={})]
    out = _dedup_cap(hits)
    assert len(out) == 1
    assert out[0].ref == "sess-1"


# ---- Hit dataclass --------------------------------------------------------


def test_hit_score_returns_rerank_score_when_set():
    h = Hit(source="code", ref="a", text="", ts=0.0, rrf=0.3, rerank_score=0.9)
    assert h.score == 0.9


def test_hit_score_falls_back_to_rrf_when_rerank_is_none():
    h = Hit(source="code", ref="a", text="", ts=0.0, rrf=0.4, rerank_score=None)
    assert h.score == 0.4


def test_hit_defaults_and_meta_dict_independence():
    h1 = Hit(source="code", ref="a", text="", ts=0.0)
    h2 = Hit(source="code", ref="b", text="", ts=0.0)
    assert h1.rrf == 0.0
    assert h1.rerank_score is None
    h1.meta["x"] = 1
    assert h2.meta == {}  # default_factory gives each Hit its own dict


# ---- vector_literal -------------------------------------------------------


def test_vector_literal_format():
    vec = np.array([1.0, 0.5, 0.25], dtype=np.float16)
    result = vector_literal(vec)
    assert result == "[1.000000,0.500000,0.250000]"


def test_vector_literal_empty():
    vec = np.array([], dtype=np.float16)
    assert vector_literal(vec) == "[]"
