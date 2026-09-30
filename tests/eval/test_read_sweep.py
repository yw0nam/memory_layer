"""Unit coverage for the read-settings sweep (no network, no database)."""

from __future__ import annotations

import asyncio
import json

import pytest

from memory_base.eval import longmemeval as lme
from memory_base.eval import read_sweep as sweep
from memory_base.retrieval import search as search_module

D1 = "2023/05/01 (Mon) 10:00"


def record(i, score, chars=40, sessions=(("s1", D1),)):
    return {
        "id": f"n{i}",
        "score": score,
        "text": "x" * chars,
        "date": D1,
        "sessions": [list(unit) for unit in sessions],
    }


def ranked(scores, chars=40):
    return [record(i, score, chars) for i, score in enumerate(scores)]


def ids(hits):
    return [hit["id"] for hit in hits]


def test_the_grid_holds_every_top_k_and_budget_point():
    names = [setting.name for setting in sweep.grid()]
    assert len(names) == 3 * 4 + 4 * 3
    assert "k3-f0.6" in names and "k10-f0" in names
    assert "b800-f0.4" in names and "b4000-f0" in names
    assert all(sweep.parse_setting(name).name == name for name in names)


def test_top_k_mode_cuts_the_rerank_top_then_the_floor_then_top_k():
    hits = ranked([0.9, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.9, 0.9])
    assert ids(sweep.apply_setting(sweep.parse_setting("k3-f0"), hits)) == ["n0", "n1", "n2"]
    assert ids(sweep.apply_setting(sweep.parse_setting("k10-f0.25"), hits)) == [
        "n0",
        "n1",
        "n2",
        "n3",
    ]
    # Candidates past RERANK_TOP never come back in top_k mode, whatever their score.
    assert len(sweep.apply_setting(sweep.parse_setting("k10-f0"), hits)) == search_module.RERANK_TOP


def test_budget_mode_packs_in_rank_order_and_applies_its_floor_first():
    hits = ranked([0.9, 0.8, 0.3, 0.2, 0.1], chars=400)
    assert ids(sweep.apply_setting(sweep.parse_setting("b250-f0"), hits)) == ["n0", "n1"]
    assert len(sweep.apply_setting(sweep.parse_setting("b4000-f0"), hits)) == 5
    assert ids(sweep.apply_setting(sweep.parse_setting("b4000-f0.25"), hits)) == [
        "n0",
        "n1",
        "n2",
    ]


def test_a_row_without_text_is_measured_by_its_length():
    rows = [{"id": "n0", "score": 0.9, "chars": 400}, {"id": "n1", "score": 0.8, "chars": 400}]
    assert ids(sweep.apply_setting(sweep.parse_setting("b150-f0"), rows)) == ["n0"]
    assert sweep.hit_tokens(rows[0]) == 100


def test_the_search_path_delivers_every_hit_at_its_estimated_tokens():
    hits = ranked([0.9, 0.8], chars=400)
    kept, tokens = sweep.deliver("search", hits)
    assert ids(kept) == ["n0", "n1"]
    assert tokens == 200
    assert sweep.deliver("search", []) == ([], 0)


def test_the_claude_code_path_stops_at_the_hook_block_limit():
    import prefetch_hook

    hits = ranked([0.9, 0.8, 0.7, 0.6], chars=500)
    kept, tokens = sweep.deliver("claude-code", hits)
    block = prefetch_hook.build_context_block(
        [{"date": sweep.DATE_PLACEHOLDER, "text": hit["text"]} for hit in hits]
    )
    assert len(block) <= prefetch_hook.BLOCK_LIMIT
    assert ids(kept) == ["n0", "n1"]
    assert tokens == len(block) // 4
    assert sweep.deliver("claude-code", []) == ([], 0)


def test_the_hermes_path_stops_at_the_prefetch_char_budget():
    import client as hermes_client

    hits = ranked([0.9, 0.8, 0.7, 0.6, 0.5], chars=500)
    kept, tokens = sweep.deliver("hermes", hits)
    assert ids(kept) == ["n0", "n1", "n2"]
    assert tokens * 4 <= hermes_client.PREFETCH_CHAR_BUDGET
    assert tokens > 3 * 500 // 4


def test_lme_metrics_count_junk_zero_hits_and_whole_packet_recall():
    questions = {
        "q1": {
            "question_id": "q1",
            "question_type": "multi-session",
            "answer_session_ids": ["s1", "s2"],
        },
        "q2": {
            "question_id": "q2",
            "question_type": "single-session-user",
            "answer_session_ids": ["s3"],
        },
        "q3_abs": {
            "question_id": "q3_abs",
            "question_type": "single-session-user",
            "answer_session_ids": ["s3"],
        },
    }
    packets = [
        {
            "question_id": "q1",
            "hits": [
                record(0, 0.9, sessions=[("s1", D1)]),
                record(1, 0.8, sessions=[("s9", D1)]),
                record(2, 0.7, sessions=[("s2", D1)]),
                record(3, 0.1, sessions=[("s8", D1)]),
            ],
        },
        {"question_id": "q2", "hits": [record(4, 0.2, sessions=[("s3", D1)])]},
        {"question_id": "q3_abs", "hits": [record(5, 0.9, sessions=[("s7", D1)])]},
    ]
    metrics = sweep.lme_metrics(packets, questions, sweep.parse_setting("k10-f0.25"), "search")
    assert metrics["recall_all"] == pytest.approx(0.5)
    assert metrics["junk_share"] == pytest.approx(1 / 3)
    assert metrics["zero_hit_rate"] == pytest.approx(1 / 3)
    assert metrics["mean_hits"] == pytest.approx(4 / 3)
    assert metrics["mean_tokens"] == pytest.approx(40 / 3)
    loose = sweep.lme_metrics(packets, questions, sweep.parse_setting("k10-f0"), "search")
    assert loose["recall_all"] == 1.0
    assert loose["zero_hit_rate"] == 0.0
    assert loose["junk_share"] == pytest.approx(2 / 5)


def test_probe_metrics_report_how_often_each_intent_injects_anything():
    rows = [
        {"intent": "off_topic", "hits": [record(0, 0.7)]},
        {"intent": "off_topic", "hits": [record(1, 0.3)]},
        {"intent": "memory", "hits": [record(2, 0.9), record(3, 0.65)]},
    ]
    metrics = sweep.probe_metrics(rows, sweep.parse_setting("k3-f0.6"), "search")
    assert metrics["off_topic"] == {
        "prompts": 2,
        "fire_rate": 0.5,
        "mean_hits": 0.5,
        "mean_tokens": 5.0,
    }
    assert metrics["memory"]["fire_rate"] == 1.0
    assert metrics["memory"]["mean_hits"] == 2.0


def test_replay_metrics_score_labels_expect_empty_and_junk_against_the_labels():
    rows = [
        {
            "query": "a",
            "query_class": "keyword",
            "relevant_ids": ["n0"],
            "expect_empty": False,
            "hits": [record(1, 0.9), record(0, 0.8)],
        },
        {
            "query": "b",
            "query_class": "keyword",
            "relevant_ids": [],
            "expect_empty": True,
            "hits": [record(2, 0.2)],
        },
        {
            "query": "c",
            "query_class": "keyword",
            "relevant_ids": ["gone"],
            "expect_empty": False,
            "hits": [],
        },
    ]
    corpus = {"n0", "n1", "n2"}
    metrics = sweep.replay_metrics(rows, corpus, sweep.parse_setting("k10-f0.25"), "search")
    assert metrics["scored"] == 1
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["expect_empty_passed"] == 1
    assert metrics["expect_empty_total"] == 1
    assert metrics["junk_share"] == 0.5
    assert metrics["zero_hit_rate"] == pytest.approx(2 / 3)
    strict = sweep.replay_metrics(rows, corpus, sweep.parse_setting("k10-f0"), "search")
    assert strict["expect_empty_passed"] == 0


def test_a_deployed_read_url_opens_every_transaction_read_only():
    assert (
        sweep.read_only_url("postgresql://u:p@h:5432/db")
        == "postgresql://u:p@h:5432/db?default_transaction_read_only=on"
    )
    assert (
        sweep.read_only_url("postgresql://u:p@h/db?sslmode=disable")
        == "postgresql://u:p@h/db?sslmode=disable&default_transaction_read_only=on"
    )


def test_the_candidates_read_keeps_every_fused_candidate(monkeypatch):
    from memory_base.eval import retrieval
    from memory_base.serve import namespaces

    class Hit:
        def __init__(self, i):
            self.meta = {"id": f"n{i}", "occurred_at": 1.0}
            self.ts, self.score, self.text = 1.0, 1 - i / 50, "t"

    calls = []

    async def search(query, **kwargs):
        calls.append(kwargs)
        return [Hit(i) for i in range(search_module.FUSED_TOP)]

    async def load(namespace, units, notes_by_unit, gate, writer=None):
        return lme.LoadStats(), {f"n{i}": {("s1", D1)} for i in range(search_module.FUSED_TOP)}

    async def create(namespace):
        pass

    monkeypatch.setattr(retrieval, "_search_with_retry", search)
    monkeypatch.setattr(lme, "load_question_notes", load)
    monkeypatch.setattr(namespaces, "create_namespace", create)
    question = {
        "question_id": "q1",
        "question_type": "multi-session",
        "question": "q?",
        "question_date": D1,
        "haystack_session_ids": ["s1"],
        "haystack_dates": [D1],
    }
    assert lme.run_name("baseline", "off", "candidates") == sweep.CANDIDATES_RUN
    packet = asyncio.run(lme.retrieve_question(question, {}, sweep.CANDIDATES_RUN))
    assert calls[0]["budget_tokens"] == sweep.CANDIDATES_BUDGET
    assert len(packet["hits"]) == search_module.FUSED_TOP


def test_probe_collection_searches_each_probe_in_the_question_namespace(monkeypatch):
    from memory_base.eval import retrieval
    from memory_base.serve import namespaces

    class Hit:
        def __init__(self, i):
            self.meta = {"id": f"n{i}"}
            self.score, self.text = 0.5, "note text"

    calls = []

    async def search(query, **kwargs):
        calls.append((query, kwargs))
        return [Hit(0)]

    async def load(namespace, units, notes_by_unit, gate, writer=None):
        calls.append((namespace, gate))
        return lme.LoadStats(), {}

    async def create(namespace):
        pass

    monkeypatch.setattr(retrieval, "_search_with_retry", search)
    monkeypatch.setattr(lme, "load_question_notes", load)
    monkeypatch.setattr(namespaces, "create_namespace", create)
    question = {"question_id": "q1", "haystack_session_ids": ["s1"], "haystack_dates": [D1]}
    probes = [{"intent": "off_topic", "query": "rename this variable please"}]
    rows = asyncio.run(sweep.probe_question(question, {}, "off", probes))
    assert calls[0] == ("lme-q1", "off")
    query, kwargs = calls[1]
    assert query == "rename this variable please"
    assert kwargs["namespaces"] == ["lme-q1"]
    assert kwargs["budget_tokens"] == sweep.CANDIDATES_BUDGET
    assert rows == [
        {
            "intent": "off_topic",
            "query": "rename this variable please",
            "hits": [{"id": "n0", "score": 0.5, "text": "note text"}],
        }
    ]


def test_the_probe_fixture_holds_forty_off_topic_and_twenty_memory_prompts():
    probes = sweep.load_probes()
    assert sum(p["intent"] == "off_topic" for p in probes) == 40
    assert sum(p["intent"] == "memory" for p in probes) == 20
    # The prefetch hook skips prompts under its minimum length, so every probe reaches search.
    import prefetch_hook

    assert not any(prefetch_hook.is_trivial_prompt(p["query"]) for p in probes)


def test_export_writes_the_delivered_packets_for_the_answer_stage(tmp_path):
    source = tmp_path / "lme"
    source.mkdir()
    packet = {
        "question_id": "q1",
        "question_type": "multi-session",
        "question": "q?",
        "question_date": D1,
        "run": sweep.CANDIDATES_RUN,
        "hits": ranked([0.9, 0.5, 0.1]),
    }
    lme.append_jsonl(lme.packets_path(source, sweep.CANDIDATES_RUN), [packet])
    out = tmp_path / "k3"
    sweep.export_packets(source, sweep.parse_setting("k3-f0.25"), "search", out)
    (exported,) = lme.read_jsonl(lme.packets_path(out, "gate-off"))
    assert ids(exported["hits"]) == ["n0", "n1"]
    assert exported["run"] == "gate-off"
    assert exported["read"] == {"setting": "k3-f0.25", "path": "search"}
    assert json.dumps(exported)


def test_a_deployed_snapshot_row_is_copied_through_the_table_row_type():
    executed = []

    class Conn:
        async def execute(self, sql, *args):
            executed.append((sql, args))

    rows = [{"id": "note:a", "metadata": {"tags": ["x"]}, "embedding": "[0.1,0.2]"}]
    asyncio.run(sweep.copy_snapshot(Conn(), rows, "memory"))
    ((sql, args),) = executed
    assert 'INSERT INTO "memory".memory_chunks ("id", "metadata", "embedding")' in sql
    assert 'jsonb_populate_record(NULL::"memory".memory_chunks, $1::jsonb)' in sql
    assert json.loads(args[0]) == rows[0]


def test_the_deployed_snapshot_reads_every_row_with_its_embedding_as_text():
    sql = sweep.snapshot_sql("memory")
    assert sql.lstrip().upper().startswith("SELECT")
    assert "embedding::text" in sql
    assert '"memory".memory_chunks' in sql
