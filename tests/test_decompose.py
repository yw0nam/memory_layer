"""Unit tests for the decomposition loop (no DB/LLM)."""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np
import pytest

from memory_base.retrieval.decompose import (
    DEEP_MAX_HOPS,
    HOP_CANDIDATE_CAP,
    SUB_QUESTION_CAP,
    _Candidate,
    _LLMError,
    _Timeout,
    _atom_rows,
    _call_timeout,
    _collapse_atoms,
    _deep_loop,
    _hop_retrieve,
    _llm_complete,
    _memory_backup,
    _parse_proposal,
    _parse_selection,
    _propose,
    _remaining,
    _select,
    deep_search,
)


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list = []

    async def create(self, *, model, messages, response_format=None, temperature=None):
        self.calls.append((model, messages, response_format))
        if not self._responses:
            raise RuntimeError("no more responses")
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return FakeResponse(resp)


class FakeChat:
    def __init__(self, completions: FakeCompletions):
        self.completions = completions


class FakeLLM:
    def __init__(self, responses: list):
        self.chat = FakeChat(FakeCompletions(responses))


class FakeEmbedder:
    def __init__(self, fail: bool = False):
        self._fail = fail
        self.calls: list = []

    async def embed(self, text: str, *, query: bool = False):
        self.calls.append((text, query))
        if self._fail:
            raise RuntimeError("embedder failure")
        return np.zeros(2048, dtype=np.float16)


class FakeConnection:
    def __init__(self, fetch_results: list | None = None, fetchval_results: list | None = None):
        self._fetch = list(fetch_results or [])
        self._fetchval = list(fetchval_results or [True])
        self.queries: list = []

    async def fetchval(self, query, *args):
        self.queries.append((query, args))
        if self._fetchval:
            return self._fetchval.pop(0)
        return True

    async def fetch(self, query, *args):
        self.queries.append((query, args))
        if self._fetch:
            return self._fetch.pop(0)
        return []


def _far_future_deadline() -> float:
    return time.monotonic() + 3600


def _expired_deadline() -> float:
    return time.monotonic() - 1


def _make_atom_row(
    parent_id: str = "p1",
    atom_id: str = "a1",
    cosine: float = 0.9,
    question: str = "atom q",
    text: str = "parent text",
    kind: str = "doc",
    ref: str = "doc.md#chunk-0",
    source_ref: str = "doc.md",
    tags: list | None = None,
    archived_at: float | None = None,
):
    return {
        "atom_id": atom_id,
        "matched_question": question,
        "atom_cosine": cosine,
        "id": parent_id,
        "source_ref": source_ref,
        "chunk_kind": kind,
        "content_raw": text,
        "distilled": text,
        "ts_last_active": 100.0,
        "metadata": {"search_ref": ref, "tags": tags or []},
        "archived_at": archived_at,
    }


def _make_history_row(
    row_id: str = "p1",
    text: str = "history text",
    kind: str = "doc",
    ref: str = "doc.md#chunk-0",
    source_ref: str = "doc.md",
    tags: list | None = None,
    ts: float = 100.0,
    idf: float = 0.5,
    archived_at: float | None = None,
):
    return {
        "id": row_id,
        "source_ref": source_ref,
        "chunk_kind": kind,
        "metadata": {"search_ref": ref, "tags": tags or []},
        "distilled": text,
        "content_raw": text,
        "ts_last_active": ts,
        "idf_score": idf,
        "archived_at": archived_at,
    }


# --- Parsing tests ---


class TestParseProposal:
    def test_valid_continue_true(self):
        text = json.dumps({"continue": True, "sub_questions": ["q1", "q2"]})
        flag, sqs = _parse_proposal(text)
        assert flag is True
        assert sqs == ["q1", "q2"]

    def test_valid_continue_false(self):
        text = json.dumps({"continue": False, "sub_questions": []})
        flag, sqs = _parse_proposal(text)
        assert flag is False
        assert sqs == []

    def test_caps_at_sub_question_limit(self):
        text = json.dumps({"continue": True, "sub_questions": ["a", "b", "c", "d"]})
        _, sqs = _parse_proposal(text)
        assert len(sqs) == SUB_QUESTION_CAP

    def test_filters_non_string_entries(self):
        text = json.dumps({"continue": True, "sub_questions": ["valid", 42, "", "  ", "ok"]})
        _, sqs = _parse_proposal(text)
        assert sqs == ["valid", "ok"]

    def test_rejects_non_object(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_proposal("[1,2,3]")

    def test_rejects_missing_continue(self):
        with pytest.raises(ValueError, match="continue"):
            _parse_proposal(json.dumps({"sub_questions": ["q"]}))

    def test_rejects_wrong_continue_type(self):
        with pytest.raises(ValueError, match="continue"):
            _parse_proposal(json.dumps({"continue": "yes", "sub_questions": []}))

    def test_rejects_missing_sub_questions(self):
        with pytest.raises(ValueError, match="sub_questions"):
            _parse_proposal(json.dumps({"continue": True}))

    def test_rejects_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_proposal("not json")


class TestParseSelection:
    def test_valid_index(self):
        text = json.dumps({"selected": 2})
        assert _parse_selection(text, 5) == 1

    def test_null_selection(self):
        text = json.dumps({"selected": None})
        assert _parse_selection(text, 5) is None

    def test_out_of_range_high(self):
        with pytest.raises(ValueError, match="out of range"):
            _parse_selection(json.dumps({"selected": 6}), 5)

    def test_out_of_range_zero(self):
        with pytest.raises(ValueError, match="out of range"):
            _parse_selection(json.dumps({"selected": 0}), 5)

    def test_non_integer(self):
        with pytest.raises(ValueError, match="integer"):
            _parse_selection(json.dumps({"selected": "first"}), 5)

    def test_rejects_non_object(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_selection("[1]", 5)

    def test_rejects_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_selection("bad", 5)


# --- LLM call tests ---


class TestLLMComplete:
    def test_success(self):
        llm = FakeLLM(["hello"])
        result = asyncio.run(
            _llm_complete(llm, [{"role": "user", "content": "hi"}], _far_future_deadline())
        )
        assert result == "hello"

    def test_transport_error_raises_llm_error(self):
        llm = FakeLLM([RuntimeError("connection refused")])
        with pytest.raises(_LLMError):
            asyncio.run(
                _llm_complete(llm, [{"role": "user", "content": "hi"}], _far_future_deadline())
            )

    def test_expired_deadline_raises_timeout(self):
        llm = FakeLLM(["hello"])
        with pytest.raises(_Timeout):
            asyncio.run(
                _llm_complete(llm, [{"role": "user", "content": "hi"}], _expired_deadline())
            )


class TestPropose:
    def test_success(self):
        resp = json.dumps({"continue": True, "sub_questions": ["sq1"]})
        llm = FakeLLM([resp])
        flag, sqs = asyncio.run(_propose(llm, "question", [], _far_future_deadline()))
        assert flag is True
        assert sqs == ["sq1"]

    def test_retry_then_succeed(self):
        bad = "not json"
        good = json.dumps({"continue": True, "sub_questions": ["sq1"]})
        llm = FakeLLM([bad, good])
        flag, sqs = asyncio.run(_propose(llm, "question", [], _far_future_deadline()))
        assert flag is True
        assert len(llm.chat.completions.calls) == 2

    def test_retry_then_fail(self):
        bad1 = "not json"
        bad2 = "still bad"
        llm = FakeLLM([bad1, bad2])
        with pytest.raises(_LLMError):
            asyncio.run(_propose(llm, "question", [], _far_future_deadline()))

    def test_transport_retry_then_fail(self):
        llm = FakeLLM([RuntimeError("err"), RuntimeError("err")])
        with pytest.raises(_LLMError):
            asyncio.run(_propose(llm, "question", [], _far_future_deadline()))

    def test_timeout_propagates(self):
        llm = FakeLLM(["anything"])
        with pytest.raises(_Timeout):
            asyncio.run(_propose(llm, "question", [], _expired_deadline()))


class TestSelect:
    def _candidates(self, n=3):
        return [
            _Candidate(f"p{i}", f"ref{i}", f"text{i}", 100.0, "doc", [], f"src{i}", None)
            for i in range(n)
        ]

    def test_valid_selection(self):
        resp = json.dumps({"selected": 2})
        llm = FakeLLM([resp])
        idx = asyncio.run(_select(llm, "q", [], self._candidates(), _far_future_deadline()))
        assert idx == 1

    def test_null_selection(self):
        resp = json.dumps({"selected": None})
        llm = FakeLLM([resp])
        idx = asyncio.run(_select(llm, "q", [], self._candidates(), _far_future_deadline()))
        assert idx is None

    def test_out_of_range_retry_then_fail(self):
        bad = json.dumps({"selected": 99})
        bad2 = json.dumps({"selected": 100})
        llm = FakeLLM([bad, bad2])
        with pytest.raises(_LLMError):
            asyncio.run(_select(llm, "q", [], self._candidates(), _far_future_deadline()))

    def test_retry_then_succeed(self):
        bad = json.dumps({"selected": 99})
        good = json.dumps({"selected": 1})
        llm = FakeLLM([bad, good])
        idx = asyncio.run(_select(llm, "q", [], self._candidates(), _far_future_deadline()))
        assert idx == 0


# --- Retrieval tests ---


class TestAtomRows:
    def test_exclusion_predicate_before_limit(self):
        conn = FakeConnection([[]])
        asyncio.run(_atom_rows(conn, "[1]", ["p1", "p2"], None, None, False))
        sql, args = conn.queries[0]
        assert "parent.id != ALL(" in sql
        assert sql.index("parent.id != ALL(") < sql.index("LIMIT")
        assert args[-1] == ["p1", "p2"]

    def test_filter_passthrough(self):
        conn = FakeConnection([[]])
        asyncio.run(_atom_rows(conn, "[1]", [], "decision", ["infra"], False))
        sql, args = conn.queries[0]
        assert "parent.chunk_kind =" in sql
        assert "parent.metadata->'tags' ?|" in sql
        assert "parent.archived_at IS NULL" in sql
        assert "decision" in args
        assert ["infra"] in args

    def test_include_archived_skips_archived_predicate(self):
        conn = FakeConnection([[]])
        asyncio.run(_atom_rows(conn, "[1]", [], None, None, True))
        sql, _ = conn.queries[0]
        assert "archived_at IS NULL" not in sql
        assert "parent.archived_at" in sql


class TestCollapseAtoms:
    def test_collapses_by_parent_keeping_highest_cosine(self):
        rows = [
            _make_atom_row(parent_id="p1", atom_id="a1", cosine=0.7, question="low"),
            _make_atom_row(parent_id="p1", atom_id="a2", cosine=0.95, question="high"),
            _make_atom_row(parent_id="p2", atom_id="a3", cosine=0.8, question="mid"),
        ]
        candidates = _collapse_atoms(rows)
        assert len(candidates) == 2
        assert candidates[0].parent_id == "p1"
        assert candidates[0].atom_question == "high"
        assert candidates[1].parent_id == "p2"

    def test_caps_at_hop_candidate_cap(self):
        rows = [
            _make_atom_row(parent_id=f"p{i}", atom_id=f"a{i}", cosine=1.0 - i * 0.01)
            for i in range(20)
        ]
        candidates = _collapse_atoms(rows)
        assert len(candidates) == HOP_CANDIDATE_CAP

    def test_carries_archived_state(self):
        rows = [
            _make_atom_row(parent_id="p1", atom_id="a1", archived_at=100.0),
            _make_atom_row(parent_id="p2", atom_id="a2", cosine=0.5),
        ]
        candidates = _collapse_atoms(rows)
        assert candidates[0].archived is True
        assert candidates[1].archived is False


class TestMemoryBackup:
    def test_exclusion_in_both_queries_before_limit(self):
        conn = FakeConnection(fetch_results=[[], []], fetchval_results=[True])
        asyncio.run(_memory_backup(conn, "q", "[1]", ["p1"], None, None, False))
        vec_sql = conn.queries[1][0]
        fts_sql = conn.queries[2][0]
        for sql in (vec_sql, fts_sql):
            assert "id != ALL(" in sql
            assert sql.index("id != ALL(") < sql.index("LIMIT")

    def test_returns_candidates_with_atom_question_none(self):
        row = _make_history_row(row_id="p1", text="backup text")
        conn = FakeConnection(fetch_results=[[row], []], fetchval_results=[True])
        candidates = asyncio.run(_memory_backup(conn, "q", "[1]", [], None, None, False))
        assert len(candidates) == 1
        assert candidates[0].atom_question is None
        assert candidates[0].text == "backup text"

    def test_carries_archived_state(self):
        row = _make_history_row(row_id="p1", archived_at=100.0)
        conn = FakeConnection(fetch_results=[[row], []], fetchval_results=[True])
        candidates = asyncio.run(_memory_backup(conn, "q", "[1]", [], None, None, True))
        assert candidates[0].archived is True

    def test_empty_when_table_missing(self):
        conn = FakeConnection(fetchval_results=[False])
        candidates = asyncio.run(_memory_backup(conn, "q", "[1]", [], None, None, False))
        assert candidates == []

    def test_filter_passthrough(self):
        conn = FakeConnection(fetch_results=[[], []], fetchval_results=[True])
        asyncio.run(_memory_backup(conn, "q", "[1]", [], "note", ["db"], False))
        vec_sql, vec_args = conn.queries[1]
        fts_sql, fts_args = conn.queries[2]
        for sql in (vec_sql, fts_sql):
            assert "chunk_kind =" in sql
            assert "metadata->'tags' ?|" in sql
        assert "note" in vec_args
        assert ["db"] in vec_args
        assert "note" in fts_args
        assert ["db"] in fts_args


class TestHopRetrieve:
    def test_stage_a_wins_when_nonempty(self):
        row = _make_atom_row(parent_id="p1", question="sub-q atom")
        conn = FakeConnection(
            [
                [row],
            ]
        )
        embedder = FakeEmbedder()
        candidates = asyncio.run(
            _hop_retrieve(
                conn,
                embedder,
                ["sub-q"],
                "original",
                "[1]",
                [],
                None,
                None,
                False,
                _far_future_deadline(),
            )
        )
        assert len(candidates) == 1
        assert candidates[0].atom_question == "sub-q atom"
        assert len(conn.queries) == 1

    def test_falls_through_to_stage_b(self):
        row = _make_atom_row(parent_id="p1", question="orig atom")
        conn = FakeConnection(
            [
                [],
                [row],
            ]
        )
        embedder = FakeEmbedder()
        candidates = asyncio.run(
            _hop_retrieve(
                conn,
                embedder,
                ["sub-q"],
                "original",
                "[1]",
                [],
                None,
                None,
                False,
                _far_future_deadline(),
            )
        )
        assert len(candidates) == 1
        assert len(conn.queries) == 2

    def test_falls_through_to_stage_c(self):
        hist_row = _make_history_row(row_id="p1", text="backup result")
        conn = FakeConnection(
            fetch_results=[[], [], [hist_row], []],
            fetchval_results=[True],
        )
        embedder = FakeEmbedder()
        candidates = asyncio.run(
            _hop_retrieve(
                conn,
                embedder,
                ["sub-q"],
                "original",
                "[1]",
                [],
                None,
                None,
                False,
                _far_future_deadline(),
            )
        )
        assert len(candidates) == 1
        assert candidates[0].atom_question is None
        assert candidates[0].text == "backup result"

    def test_empty_chain_returns_empty(self):
        conn = FakeConnection(
            fetch_results=[[], [], [], []],
            fetchval_results=[True],
        )
        embedder = FakeEmbedder()
        candidates = asyncio.run(
            _hop_retrieve(
                conn,
                embedder,
                ["sub-q"],
                "original",
                "[1]",
                [],
                None,
                None,
                False,
                _far_future_deadline(),
            )
        )
        assert candidates == []

    def test_timeout_during_embedding(self):
        conn = FakeConnection()
        embedder = FakeEmbedder()
        with pytest.raises(_Timeout):
            asyncio.run(
                _hop_retrieve(
                    conn,
                    embedder,
                    ["sub-q"],
                    "original",
                    "[1]",
                    [],
                    None,
                    None,
                    False,
                    _expired_deadline(),
                )
            )

    def test_skips_failed_subquestion_embedding(self):
        row = _make_atom_row(parent_id="p1")
        conn = FakeConnection([[row]])
        embedder = FakeEmbedder(fail=True)
        candidates = asyncio.run(
            _hop_retrieve(
                conn,
                embedder,
                ["sub-q"],
                "original",
                "[1]",
                [],
                None,
                None,
                False,
                _far_future_deadline(),
            )
        )
        assert len(candidates) == 1


# --- Loop tests ---


def _propose_response(continue_flag: bool, sub_questions: list[str] | None = None) -> str:
    return json.dumps(
        {
            "continue": continue_flag,
            "sub_questions": sub_questions or ["sq1"],
        }
    )


def _select_response(index: int | None) -> str:
    return json.dumps({"selected": index})


class TestDeepLoop:
    def _make_loop_args(self, llm_responses, fetch_results=None, fetchval_results=None):
        conn = FakeConnection(fetch_results=fetch_results, fetchval_results=fetchval_results)
        embedder = FakeEmbedder()
        llm = FakeLLM(llm_responses)
        return conn, embedder, llm

    def test_done_on_continue_false(self):
        llm_responses = [_propose_response(False)]
        conn, embedder, llm = self._make_loop_args(llm_responses)
        result = asyncio.run(
            _deep_loop("q", conn, embedder, llm, 3, None, None, False, _far_future_deadline())
        )
        assert result.stopped_reason == "done"
        assert result.hops_used == 0
        assert len(result.trace) == 1
        assert result.trace[0].hop == 1

    def test_max_hops(self):
        llm_responses = []
        fetch_results = []
        for i in range(3):
            llm_responses.append(_propose_response(True, [f"sq{i}"]))
            llm_responses.append(_select_response(1))
            fetch_results.append(
                [_make_atom_row(parent_id=f"p{i + 1}", atom_id=f"a{i}", cosine=0.9 - i * 0.1)]
            )

        conn = FakeConnection(fetch_results=fetch_results)
        embedder = FakeEmbedder()
        llm = FakeLLM(llm_responses)
        result = asyncio.run(
            _deep_loop("q", conn, embedder, llm, 3, None, None, False, _far_future_deadline())
        )
        assert result.stopped_reason == "max_hops"
        assert result.hops_used == 3
        assert len(result.trace) == 3

    def test_no_candidates(self):
        llm_responses = [_propose_response(True)]
        conn = FakeConnection(
            fetch_results=[[], [], [], []],
            fetchval_results=[True],
        )
        embedder = FakeEmbedder()
        llm = FakeLLM(llm_responses)
        result = asyncio.run(
            _deep_loop("q", conn, embedder, llm, 3, None, None, False, _far_future_deadline())
        )
        assert result.stopped_reason == "no_candidates"
        assert result.hops_used == 0
        assert len(result.trace) == 1
        assert result.trace[0].sub_questions == ["sq1"]

    def test_no_selection(self):
        atom_row = _make_atom_row(parent_id="p1")
        llm_responses = [
            _propose_response(True),
            _select_response(None),
        ]
        conn = FakeConnection(fetch_results=[[atom_row]])
        embedder = FakeEmbedder()
        llm = FakeLLM(llm_responses)
        result = asyncio.run(
            _deep_loop("q", conn, embedder, llm, 3, None, None, False, _far_future_deadline())
        )
        assert result.stopped_reason == "no_selection"
        assert result.hops_used == 0
        assert len(result.trace) == 1

    def test_llm_error_on_proposal(self):
        llm_responses = ["bad json", "still bad"]
        conn, embedder, llm = self._make_loop_args(llm_responses)
        result = asyncio.run(
            _deep_loop("q", conn, embedder, llm, 3, None, None, False, _far_future_deadline())
        )
        assert result.stopped_reason == "llm_error"
        assert result.hops_used == 0
        assert result.trace[0].sub_questions == []

    def test_llm_error_on_selection(self):
        atom_row = _make_atom_row(parent_id="p1")
        llm_responses = [
            _propose_response(True),
            "bad json",
            "still bad",
        ]
        conn = FakeConnection(fetch_results=[[atom_row]])
        embedder = FakeEmbedder()
        llm = FakeLLM(llm_responses)
        result = asyncio.run(
            _deep_loop("q", conn, embedder, llm, 3, None, None, False, _far_future_deadline())
        )
        assert result.stopped_reason == "llm_error"
        assert result.hops_used == 0
        assert len(result.trace) == 1
        assert result.trace[0].sub_questions == ["sq1"]

    def test_timeout_before_first_hop(self):
        conn, embedder, llm = self._make_loop_args([])
        result = asyncio.run(
            _deep_loop("q", conn, embedder, llm, 3, None, None, False, _expired_deadline())
        )
        assert result.stopped_reason == "timeout"
        assert result.hops_used == 0

    def test_chosen_parent_exclusion_across_hops(self):
        llm_responses = [
            _propose_response(True, ["sq1"]),
            _select_response(1),
            _propose_response(True, ["sq2"]),
            _select_response(1),
            _propose_response(False),
        ]
        fetch_results = [
            [_make_atom_row(parent_id="p1", atom_id="a1", cosine=0.9)],
            [_make_atom_row(parent_id="p2", atom_id="a2", cosine=0.85)],
        ]
        conn = FakeConnection(fetch_results=fetch_results)
        embedder = FakeEmbedder()
        llm = FakeLLM(llm_responses)
        result = asyncio.run(
            _deep_loop("q", conn, embedder, llm, 3, None, None, False, _far_future_deadline())
        )
        assert result.stopped_reason == "done"
        assert result.hops_used == 2
        atom_queries = [q for q in conn.queries if "atom.chunk_kind = 'atom'" in q[0]]
        assert len(atom_queries) >= 2
        second_atom_sql, second_atom_args = atom_queries[1]
        assert "parent.id != ALL(" in second_atom_sql
        assert "p1" in second_atom_args[-1]

    def test_trace_longer_than_hops_used(self):
        atom_row = _make_atom_row(parent_id="p1")
        llm_responses = [
            _propose_response(True),
            _select_response(1),
            _propose_response(False),
        ]
        conn = FakeConnection(fetch_results=[[atom_row]])
        embedder = FakeEmbedder()
        llm = FakeLLM(llm_responses)
        result = asyncio.run(
            _deep_loop("q", conn, embedder, llm, 3, None, None, False, _far_future_deadline())
        )
        assert result.hops_used == 1
        assert len(result.trace) == 2
        assert result.trace[0].selected_ref is not None
        assert result.trace[1].selected_ref is None

    def test_evidence_hop_numbering(self):
        llm_responses = [
            _propose_response(True, ["sq1"]),
            _select_response(1),
            _propose_response(True, ["sq2"]),
            _select_response(1),
            _propose_response(False),
        ]
        fetch_results = [
            [_make_atom_row(parent_id="p1", atom_id="a1")],
            [_make_atom_row(parent_id="p2", atom_id="a2")],
        ]
        conn = FakeConnection(fetch_results=fetch_results)
        embedder = FakeEmbedder()
        llm = FakeLLM(llm_responses)
        result = asyncio.run(
            _deep_loop("q", conn, embedder, llm, 3, None, None, False, _far_future_deadline())
        )
        assert result.evidence[0].hop == 1
        assert result.evidence[1].hop == 2

    def test_filter_passthrough_in_retrieval(self):
        llm_responses = [
            _propose_response(True),
            _select_response(1),
            _propose_response(False),
        ]
        atom_row = _make_atom_row(parent_id="p1")
        conn = FakeConnection(fetch_results=[[atom_row]])
        embedder = FakeEmbedder()
        llm = FakeLLM(llm_responses)
        asyncio.run(
            _deep_loop(
                "q",
                conn,
                embedder,
                llm,
                3,
                "decision",
                ["infra"],
                False,
                _far_future_deadline(),
            )
        )
        atom_sql = [q[0] for q in conn.queries if "atom.chunk_kind = 'atom'" in q[0]]
        assert len(atom_sql) >= 1
        assert "parent.chunk_kind =" in atom_sql[0]
        assert "parent.metadata->'tags' ?|" in atom_sql[0]


# --- Validation tests ---


class TestDeepSearchValidation:
    def test_max_hops_zero_rejected(self):
        with pytest.raises(ValueError, match="max_hops"):
            asyncio.run(deep_search("q", max_hops=0))

    def test_max_hops_above_max_rejected(self):
        with pytest.raises(ValueError, match="max_hops"):
            asyncio.run(deep_search("q", max_hops=DEEP_MAX_HOPS + 1))

    def test_max_hops_one_accepted(self):
        try:
            asyncio.run(deep_search("q", max_hops=1))
        except ValueError as e:
            if "max_hops" in str(e):
                pytest.fail("max_hops=1 should be accepted")
        except Exception:
            pass

    def test_max_hops_at_max_accepted(self):
        try:
            asyncio.run(deep_search("q", max_hops=DEEP_MAX_HOPS))
        except ValueError as e:
            if "max_hops" in str(e):
                pytest.fail(f"max_hops={DEEP_MAX_HOPS} should be accepted")
        except Exception:
            pass

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValueError, match="kind"):
            asyncio.run(deep_search("q", kind="atom"))

    def test_empty_tags_rejected(self):
        with pytest.raises(ValueError, match="tags"):
            asyncio.run(deep_search("q", tags=[]))

    def test_non_list_tags_rejected(self):
        with pytest.raises(ValueError, match="tags"):
            asyncio.run(deep_search("q", tags="infra"))


# --- Helper tests ---


class TestHelpers:
    def test_remaining_positive(self):
        assert _remaining(time.monotonic() + 10) > 0

    def test_remaining_expired(self):
        assert _remaining(time.monotonic() - 1) == 0.0

    def test_call_timeout_capped_by_service_timeout(self):
        from memory_base.core.config import SERVICE_TIMEOUT_SECONDS

        deadline = time.monotonic() + 9999
        assert _call_timeout(deadline) == SERVICE_TIMEOUT_SECONDS

    def test_call_timeout_uses_remaining_when_smaller(self):
        deadline = time.monotonic() + 5
        assert _call_timeout(deadline) <= 5.1
