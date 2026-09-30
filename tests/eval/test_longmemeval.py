"""Unit coverage for the LongMemEval retrieve/score harness (no network, no database)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memory_base.eval import longmemeval as lme
from memory_base.eval import longmemeval_prompts as prompts

TYPES = {
    "multi-session": 133,
    "temporal-reasoning": 133,
    "knowledge-update": 78,
    "single-session-user": 70,
    "single-session-assistant": 56,
    "single-session-preference": 30,
}


def make_question(qid, qtype, *, sessions=(), answers=(), question="q?", answer="a"):
    sessions = list(sessions)
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": question,
        "question_date": "2023/06/01 (Thu) 10:00",
        "answer": answer,
        "answer_session_ids": list(answers),
        "haystack_session_ids": [sid for sid, _ in sessions],
        "haystack_dates": [date for _, date in sessions],
        "haystack_sessions": [[{"role": "user", "content": sid}] for sid, _ in sessions],
    }


def synthetic_dataset():
    questions = []
    for qtype, count in TYPES.items():
        for index in range(count):
            questions.append(make_question(f"{qtype}-{index:03d}", qtype))
    return questions


def test_subset_is_seeded_deterministic_and_proportional_per_type():
    dataset = synthetic_dataset()
    first = lme.select_subset(dataset)
    second = lme.select_subset(list(dataset))
    assert [q["question_id"] for q in first] == [q["question_id"] for q in second]
    assert len(first) == 100
    counts = Counter(q["question_type"] for q in first)
    assert counts == {
        "multi-session": 27,
        "temporal-reasoning": 27,
        "knowledge-update": 15,
        "single-session-user": 14,
        "single-session-assistant": 11,
        "single-session-preference": 6,
    }
    order = {q["question_id"]: index for index, q in enumerate(dataset)}
    assert [order[q["question_id"]] for q in first] == sorted(
        order[q["question_id"]] for q in first
    )


def test_subset_changes_with_the_seed():
    dataset = synthetic_dataset()
    ids = {q["question_id"] for q in lme.select_subset(dataset, seed=0)}
    assert ids != {q["question_id"] for q in lme.select_subset(dataset, seed=1)}


def test_manifest_sections_round_trip_and_merge(tmp_path):
    path = tmp_path / "manifest.json"
    lme.update_manifest(path, "subset", {"seed": 0, "question_ids": ["a", "b"]})
    lme.update_manifest(path, "extract", {"notes": 3})
    lme.update_manifest(path, "subset", {"seed": 0, "question_ids": ["a"]})
    manifest = lme.read_manifest(path)
    assert manifest["subset"] == {"seed": 0, "question_ids": ["a"]}
    assert manifest["extract"] == {"notes": 3}
    assert json.loads(path.read_text()) == manifest


def test_dataset_dates_convert_to_iso_8601():
    assert lme.iso_datetime("2023/05/20 (Sat) 02:21") == "2023-05-20T02:21:00"
    with pytest.raises(ValueError):
        lme.iso_datetime("2023-05-20 02:21")


def test_session_units_are_unique_session_date_pairs_in_date_order():
    question = make_question(
        "q1",
        "multi-session",
        sessions=[
            ("s2", "2023/05/21 (Sun) 09:00"),
            ("s1", "2023/05/20 (Sat) 02:21"),
            ("s2", "2023/05/22 (Mon) 09:00"),
        ],
    )
    assert lme.session_units(question) == [
        ("s1", "2023/05/20 (Sat) 02:21"),
        ("s2", "2023/05/21 (Sun) 09:00"),
        ("s2", "2023/05/22 (Mon) 09:00"),
    ]


def test_read_jsonl_discards_a_partial_last_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": ')
    assert lme.read_jsonl(path) == [{"a": 1}, {"a": 2}]
    assert lme.read_jsonl(tmp_path / "missing.jsonl") == []


def fake_save_note(calls):
    async def save(content, *, tags, kind, namespace, occurred_at, allow_similar):
        calls.append((content, kind, namespace, occurred_at, allow_similar, tuple(tags)))
        note_id = f"note:{namespace}:{hashlib.sha256(content.encode()).hexdigest()[:16]}"
        seen = [c for c in calls[:-1] if c[0] == content and c[2] == namespace]
        similar = [{"id": "note:other"}] if "similar" in content else []
        return {"id": note_id, "stored": not seen, "similar": similar}

    return save


def test_loading_maps_each_note_back_to_every_session_it_came_from():
    units = [("s1", "2023/05/20 (Sat) 02:21"), ("s2", "2023/05/21 (Sun) 09:00")]
    notes_by_unit = {
        units[0]: [
            {"content": "The user owns a red bike.", "kind": "note", "gate": "stored"},
            {"content": "Refused chatter.", "kind": "note", "gate": "refused"},
        ],
        units[1]: [
            {"content": "The user owns a red bike.", "kind": "note", "gate": "stored"},
            {"content": "A similar fact.", "kind": "episode", "gate": "stored"},
        ],
    }
    calls = []
    stats, provenance = asyncio.run(
        lme.load_question_notes("lme-q1", units, notes_by_unit, save=fake_save_note(calls))
    )
    assert [c[0] for c in calls] == [
        "The user owns a red bike.",
        "The user owns a red bike.",
        "A similar fact.",
    ]
    assert all(c[2] == "lme-q1" and c[4] is True and c[5] for c in calls)
    assert calls[0][3] == "2023-05-20T02:21:00"
    bike = f"note:lme-q1:{hashlib.sha256(b'The user owns a red bike.').hexdigest()[:16]}"
    assert provenance[bike] == {units[0], units[1]}
    assert stats.submitted == 3
    assert stats.stored == 2
    assert stats.duplicates == 1
    assert stats.similar_acks == 1


def test_loading_counts_credential_and_invalid_refusals():
    from memory_base.serve.notes import CredentialNoteError

    async def save(content, **kwargs):
        if "secret" in content:
            raise CredentialNoteError("AWS Access Key")
        raise ValueError("content exceeds 4000 chars")

    units = [("s1", "2023/05/20 (Sat) 02:21")]
    notes_by_unit = {
        units[0]: [
            {"content": "has secret", "kind": "note", "gate": "stored"},
            {"content": "too long", "kind": "note", "gate": "stored"},
        ]
    }
    stats, provenance = asyncio.run(
        lme.load_question_notes("lme-q1", units, notes_by_unit, save=save)
    )
    assert stats.credential_refused == 1
    assert stats.invalid == 1
    assert stats.stored == 0
    assert provenance == {}


def test_dated_variant_embeds_date_prefixed_text_only_while_a_date_is_bound():
    seen = []

    async def embed(embedder, text):
        seen.append(text)
        return "vec"

    wrapped = lme.dated_embed_text(embed)

    async def run():
        await wrapped(None, "plain")
        token = lme.NOTE_DATE.set("2023-05-20")
        try:
            await wrapped(None, "The user owns a red bike.")
        finally:
            lme.NOTE_DATE.reset(token)

    asyncio.run(run())
    assert seen == ["plain", "2023-05-20: The user owns a red bike."]


def hit(note_id, sessions, score=0.9):
    return {
        "id": note_id,
        "date": "2023/05/20 (Sat) 02:21",
        "score": score,
        "text": note_id,
        "sessions": [[sid, "2023/05/20 (Sat) 02:21"] for sid in sessions],
    }


def test_session_ranking_is_ordered_distinct_sessions_expanding_collided_notes():
    hits = [hit("n1", ["s3"]), hit("n2", ["s1", "s3"]), hit("n3", ["s2"]), hit("n4", ["s1"])]
    assert lme.session_ranking(hits) == ["s3", "s1", "s2"]


def test_recall_all_and_ndcg_any_follow_upstream_definitions():
    ranking = ["x", "a", "y", "b"]
    assert lme.recall_all_at_k(ranking, {"a", "b"}, k=5) == 1.0
    assert lme.recall_all_at_k(ranking, {"a", "b"}, k=3) == 0.0
    assert lme.recall_all_at_k([], {"a"}, k=5) == 0.0
    # Upstream dcg weights rank 1 and rank 2 both by 1, then 1/log2(rank).
    assert lme.ndcg_any_at_k(["a"], {"a"}, k=5) == 1.0
    assert lme.ndcg_any_at_k(["x", "a"], {"a"}, k=5) == 1.0
    assert lme.ndcg_any_at_k(["x", "y", "a"], {"a"}, k=5) == pytest.approx(1 / 1.5849625)
    assert lme.ndcg_any_at_k(["x", "a", "y", "b"], {"a", "b"}, k=5) == pytest.approx(1.5 / 2)
    with pytest.raises(ValueError):
        lme.recall_all_at_k(ranking, set(), k=5)


def test_retrieval_metrics_exclude_abstention_and_count_zero_hit_packets():
    questions = {
        "q1": make_question("q1", "multi-session", answers=["s1", "s2"]),
        "q2": make_question("q2", "temporal-reasoning", answers=["s9"]),
        "q3_abs": make_question("q3_abs", "multi-session", answers=["s5"]),
    }
    packets = [
        {"question_id": "q1", "hits": [hit("n1", ["s1"]), hit("n2", ["s2"])]},
        {"question_id": "q2", "hits": []},
        {"question_id": "q3_abs", "hits": []},
    ]
    report = lme.retrieval_metrics(packets, questions)
    assert report["overall"]["count"] == 2
    assert report["overall"]["recall_all@5"] == 0.5
    assert report["overall"]["ndcg_any@10"] == 0.5
    assert report["multi-session"]["recall_all@10"] == 1.0
    assert report["temporal-reasoning"]["recall_all@10"] == 0.0
    assert report["zero_hit_packets"] == 2
    assert report["zero_hit_packets_scored"] == 1


UPSTREAM_TEMPLATE_SHA256 = {
    "qa": "fba020ba3d57982efdc9a937c1c01f897b789a608c7f88e60244121f6505e5bc",
    "temporal-reasoning": "8d33a5fdd83afeeb4592454a965eab43d1fcb2dedc042d1d3892f4254be6c273",
    "knowledge-update": "183a9b3a6197ec620940f610cdc1207201ec98c1113dd633ea685cfc322fafac",
    "single-session-preference": "741ee3bcbea7ff5e8ed359acef61d2f8ded3de021bbcff6ee13de455f2e2aa9b",
    "abstention": "5c0b365a1e1d06db36377c735432b56e122ca3c428f89faf61d43a0d5a7e050b",
    "answer": "00410c8193a84a2bdb96f86ea1c806ccdc6379f6144060ab03a5fa195f32009e",
}


def test_prompt_templates_are_byte_exact_upstream_copies():
    templates = {
        "qa": prompts.QA_TEMPLATE,
        "temporal-reasoning": prompts.TEMPORAL_TEMPLATE,
        "knowledge-update": prompts.KNOWLEDGE_UPDATE_TEMPLATE,
        "single-session-preference": prompts.PREFERENCE_TEMPLATE,
        "abstention": prompts.ABSTENTION_TEMPLATE,
        "answer": prompts.ANSWER_TEMPLATE,
    }
    digests = {name: hashlib.sha256(t.encode()).hexdigest() for name, t in templates.items()}
    assert digests == UPSTREAM_TEMPLATE_SHA256
    assert len(prompts.UPSTREAM_COMMIT) == 40


def test_judge_prompt_picks_the_template_by_type_and_abstention_suffix():
    q = make_question("q1", "knowledge-update", question="Where?", answer=7)
    assert lme.judge_prompt(q, "Paris") == prompts.KNOWLEDGE_UPDATE_TEMPLATE.format(
        "Where?", 7, "Paris"
    )
    for qtype in ("single-session-user", "single-session-assistant", "multi-session"):
        q = make_question("q1", qtype, question="Where?")
        assert lme.judge_prompt(q, "Paris") == prompts.QA_TEMPLATE.format("Where?", "a", "Paris")
    q = make_question("q1_abs", "temporal-reasoning", question="Where?")
    assert lme.judge_prompt(q, "Paris") == prompts.ABSTENTION_TEMPLATE.format(
        "Where?", "a", "Paris"
    )


def test_judge_label_parses_yes_exactly_as_upstream():
    assert lme.judge_label("Yes.") is True
    assert lme.judge_label("  yes") is True
    assert lme.judge_label("no") is False
    assert lme.judge_label("No, the response misses it.") is False
    # Upstream tests substring containment, so this counts as yes.
    assert lme.judge_label("eyes") is True


def test_answer_prompt_renders_date_sorted_hits_with_the_upstream_template():
    packet = {
        "question_id": "q1",
        "question": "What bike?",
        "question_date": "2023/06/01 (Thu) 10:00",
        "hits": [
            {**hit("n2", ["s2"]), "date": "2023/05/21 (Sun) 09:00", "text": "Second."},
            {**hit("n1", ["s1"]), "date": "2023/05/20 (Sat) 02:21", "text": "First."},
        ],
    }
    history = (
        "\n### Session 1:\nSession Date: 2023/05/20 (Sat) 02:21\nSession Content:\nFirst.\n"
        "\n### Session 2:\nSession Date: 2023/05/21 (Sun) 09:00\nSession Content:\nSecond.\n"
    )
    assert lme.answer_prompt(packet) == prompts.ANSWER_TEMPLATE.format(
        history, "2023/06/01 (Thu) 10:00", "What bike?"
    )


def reply(qid, text, model="glm-5.3-flash"):
    return {"question_id": qid, "text": text, "model": model}


def judged(question, answer_text, verdict):
    return {
        **reply(question["question_id"], verdict),
        "prompt_sha256": judge_sha(question, answer_text),
    }


def judge_sha(question, answer_text):
    return lme.prompt_sha(lme.judge_prompt(question, answer_text))


def test_current_rows_keep_the_latest_row_answering_the_current_prompt():
    expected = {"q1": "prompt one", "q2": "prompt two", "q3": "prompt three"}
    rows = [
        {**reply("q1", "first"), "prompt_sha256": lme.prompt_sha("prompt one")},
        {**reply("q1", "second"), "prompt_sha256": lme.prompt_sha("prompt one")},
        {**reply("q2", "stale"), "prompt_sha256": lme.prompt_sha("old prompt")},
        {**reply("unknown", "x"), "prompt_sha256": lme.prompt_sha("prompt one")},
    ]
    current = lme.current_rows(rows, expected)
    assert {qid: row["text"] for qid, row in current.items()} == {"q1": "second"}


def test_score_aggregates_accuracy_overall_per_type_and_over_abstention():
    questions = {
        "q1": make_question("q1", "multi-session"),
        "q2": make_question("q2", "multi-session"),
        "q3": make_question("q3", "temporal-reasoning"),
        "q4_abs": make_question("q4_abs", "temporal-reasoning"),
    }
    answers = [{**reply(qid, f"answer {qid}"), "prompt_sha256": "a"} for qid in questions]
    judgments = [
        judged(questions["q1"], "answer q1", "yes"),
        judged(questions["q2"], "answer q2", "no"),
        judged(questions["q4_abs"], "answer q4_abs", "Yes"),
    ]
    report = lme.qa_accuracy(list(questions), questions, answers, judgments)
    assert report["overall"] == {"correct": 2, "judged": 3, "accuracy": pytest.approx(2 / 3)}
    assert report["multi-session"] == {"correct": 1, "judged": 2, "accuracy": 0.5}
    assert report["temporal-reasoning"] == {"correct": 1, "judged": 1, "accuracy": 1.0}
    assert report["abstention"] == {"correct": 1, "judged": 1, "accuracy": 1.0}
    assert report["unjudged"] == ["q3"]


def test_score_ignores_a_judgment_of_a_different_answer():
    questions = {"q1": make_question("q1", "multi-session")}
    answers = [{**reply("q1", "new answer"), "prompt_sha256": "a"}]
    judgments = [judged(questions["q1"], "old answer", "yes")]
    report = lme.qa_accuracy(["q1"], questions, answers, judgments)
    assert report["overall"]["judged"] == 0
    assert report["unjudged"] == ["q1"]


def test_score_uses_the_latest_judgment_of_the_current_answer():
    questions = {"q1": make_question("q1", "multi-session")}
    answers = [{**reply("q1", "the answer"), "prompt_sha256": "a"}]
    judgments = [
        judged(questions["q1"], "the answer", "no"),
        judged(questions["q1"], "the answer", "yes"),
    ]
    report = lme.qa_accuracy(["q1"], questions, answers, judgments)
    assert report["overall"] == {"correct": 1, "judged": 1, "accuracy": 1.0}


def test_run_names_pair_gate_and_variant_without_gate_off_dated():
    assert lme.run_name("baseline", "on") == "baseline"
    assert lme.run_name("dated", "on") == "dated"
    assert lme.run_name("baseline", "off") == "gate-off"
    with pytest.raises(ValueError):
        lme.run_name("dated", "off")
    assert lme.packets_path(Path("d"), "gate-off") == Path("d/packets-gate-off.jsonl")
    assert lme.stage_output_path(Path("d"), "judge", "gate-off") == Path(
        "d/judgments-gate-off.jsonl"
    )


def test_the_prefetch_read_setting_names_its_own_runs():
    assert lme.run_name("baseline", "on", "prefetch") == "prefetch"
    assert lme.run_name("baseline", "off", "prefetch") == "prefetch-gate-off"
    assert lme.run_name("baseline", "on", "search") == "baseline"
    assert lme.packets_path(Path("d"), "prefetch-gate-off") == Path(
        "d/packets-prefetch-gate-off.jsonl"
    )
    assert lme.read_setting("prefetch-gate-off") == {"top_k": 5, "min_score": 0.6}
    assert lme.read_setting("gate-off") == {"top_k": 10, "min_score": None}
    assert set(lme.RUNS) >= {"prefetch", "prefetch-gate-off"}


def test_a_prefetch_run_searches_with_the_prefetch_floor_and_keeps_five_hits(monkeypatch):
    from memory_base.eval import retrieval
    from memory_base.serve import namespaces

    class Hit:
        def __init__(self, i):
            self.meta = {"id": f"n{i}", "occurred_at": 1.0}
            self.ts, self.score, self.text = 1.0, 1 - i / 10, "t"

    calls = []

    async def search(query, **kwargs):
        calls.append(kwargs)
        return [Hit(i) for i in range(7)]

    async def load(namespace, units, notes_by_unit, gate):
        calls.append(gate)
        return lme.LoadStats(), {f"n{i}": {("s1", D1)} for i in range(7)}

    async def create(namespace):
        pass

    monkeypatch.setattr(retrieval, "_search_with_retry", search)
    monkeypatch.setattr(lme, "load_question_notes", load)
    monkeypatch.setattr(namespaces, "create_namespace", create)
    question = make_question("q1", "single-session-user", sessions=[("s1", D1)])
    packet = asyncio.run(lme.retrieve_question(question, {}, "prefetch-gate-off"))
    assert calls[0] == "off"
    assert calls[1]["min_score"] == 0.6
    assert [h["id"] for h in packet["hits"]] == ["n0", "n1", "n2", "n3", "n4"]
    packet = asyncio.run(lme.retrieve_question(question, {}, "baseline"))
    assert calls[2] == "on"
    assert calls[3]["min_score"] is None
    assert len(packet["hits"]) == 7


def test_the_budget_read_setting_names_its_own_runs():
    assert lme.run_name("baseline", "off", "budget") == "budget-gate-off"
    assert lme.read_setting("budget-gate-off") == {"budget_tokens": 4000}
    assert lme.read_setting("budget") == {"budget_tokens": 4000}
    assert set(lme.RUNS) >= {"budget", "budget-gate-off"}


def test_a_budget_run_passes_the_budget_to_search_and_keeps_every_packed_hit(monkeypatch):
    from memory_base.eval import retrieval
    from memory_base.serve import namespaces

    class Hit:
        def __init__(self, i):
            self.meta, self.ts, self.score, self.text = {"id": f"n{i}"}, 1.0, 1 - i / 20, "t"

    calls = []

    async def search(query, **kwargs):
        calls.append(kwargs)
        return [Hit(i) for i in range(12)]

    async def load(namespace, units, notes_by_unit, gate):
        return lme.LoadStats(), {f"n{i}": {("s1", D1)} for i in range(12)}

    async def create(namespace):
        pass

    monkeypatch.setattr(retrieval, "_search_with_retry", search)
    monkeypatch.setattr(lme, "load_question_notes", load)
    monkeypatch.setattr(namespaces, "create_namespace", create)
    question = make_question("q1", "multi-session", sessions=[("s1", D1)])
    packet = asyncio.run(lme.retrieve_question(question, {}, "budget-gate-off"))
    assert calls[0]["budget_tokens"] == 4000
    assert len(packet["hits"]) == 12
    assert packet["budget_tokens"] == 4000
    packet = asyncio.run(lme.retrieve_question(question, {}, "gate-off"))
    assert calls[1].get("budget_tokens") is None
    assert len(packet["hits"]) == 10
    assert packet["budget_tokens"] is None


def test_gate_off_loading_adds_gate_refused_notes_but_not_save_path_refusals():
    units = [("s1", "2023/05/20 (Sat) 02:21")]
    notes_by_unit = {
        units[0]: [
            {"content": "Kept.", "kind": "note", "gate": "stored", "gate_reason": "ok"},
            {"content": "Gate refused.", "kind": "note", "gate": "refused", "gate_reason": "plan"},
            {
                "content": "Bad kind.",
                "kind": "plan",
                "gate": "refused",
                "gate_reason": "validation: kind must be one of ('note', 'decision', 'episode')",
            },
            {
                "content": "Secret.",
                "kind": "note",
                "gate": "refused",
                "gate_reason": "credential: GitHub Token",
            },
        ]
    }
    calls = []
    stats, _ = asyncio.run(
        lme.load_question_notes("lme-q1", units, notes_by_unit, save=fake_save_note(calls))
    )
    assert [c[0] for c in calls] == ["Kept."]
    assert stats.gate_refused == 0
    calls = []
    stats, _ = asyncio.run(
        lme.load_question_notes(
            "lme-q1", units, notes_by_unit, save=fake_save_note(calls), gate="off"
        )
    )
    assert [c[0] for c in calls] == ["Kept.", "Gate refused."]
    assert stats.gate_refused == 1
    assert stats.submitted == 2


def test_judge_audit_sample_is_seeded_and_carries_what_the_auditor_needs():
    questions = {
        f"q{i}": make_question(f"q{i}", "multi-session", answer=f"ref {i}") for i in range(30)
    }
    answers = {qid: {**reply(qid, f"answer {qid}"), "prompt_sha256": "a"} for qid in questions}
    judgments = [judged(questions[qid], f"answer {qid}", "yes") for qid in questions]
    first = lme.judge_audit_sample(list(questions), questions, answers, judgments)
    second = lme.judge_audit_sample(list(questions), questions, answers, judgments)
    assert first == second
    assert len(first) == 20
    assert len({row["question_id"] for row in first}) == 20
    row = first[0]
    assert row["reference"] == questions[row["question_id"]]["answer"]
    assert row["response"] == f"answer {row['question_id']}"
    assert row["judge_reply"] == "yes"
    assert row["judge_label"] is True
    assert row["human_label"] is None
    other = lme.judge_audit_sample(list(questions), questions, answers, judgments, seed=1)
    assert {r["question_id"] for r in other} != {r["question_id"] for r in first}


def test_judge_agreement_counts_only_hand_labeled_rows():
    rows = [
        {"judge_label": True, "human_label": True},
        {"judge_label": True, "human_label": False},
        {"judge_label": False, "human_label": False},
        {"judge_label": False, "human_label": None},
    ]
    assert lme.judge_agreement(rows) == {
        "sample": 4,
        "labeled": 3,
        "agreed": 2,
        "agreement_rate": pytest.approx(2 / 3),
    }
    assert lme.judge_agreement([])["agreement_rate"] is None


D1, D2, D9 = "2023/05/01 (Mon) 10:00", "2023/05/02 (Tue) 10:00", "2023/05/09 (Tue) 10:00"


def test_gate_rates_count_refusals_per_question_type():
    questions = {
        "q1": make_question("q1", "multi-session", sessions=[("s1", D1), ("s2", D2)]),
        "q2": make_question("q2", "temporal-reasoning", sessions=[("s2", D2)]),
    }
    notes = [
        {"session_id": "s1", "date": D1, "gate": "stored"},
        {"session_id": "s1", "date": D1, "gate": "refused"},
        {"session_id": "s2", "date": D2, "gate": "refused"},
        {"session_id": "s9", "date": D9, "gate": "stored"},
    ]
    rates = lme.gate_rates(["q1", "q2"], questions, notes)
    assert rates["overall"] == {"notes": 3, "refused": 2, "refused_rate": pytest.approx(2 / 3)}
    assert rates["multi-session"] == {
        "notes": 3,
        "refused": 2,
        "refused_rate": pytest.approx(2 / 3),
    }
    assert rates["temporal-reasoning"] == {"notes": 1, "refused": 1, "refused_rate": 1.0}


def test_throwaway_database_url_must_be_loopback():
    assert (
        lme.throwaway_db_url("pw", "54321") == "postgresql://memory:pw@127.0.0.1:54321/memory_base"
    )
    with pytest.raises(ValueError):
        lme.throwaway_db_url("pw", "not-a-port")


def test_upstream_manifest_pins_the_commit_and_every_copied_template():
    upstream = lme.upstream_manifest()
    assert upstream["commit"] == prompts.UPSTREAM_COMMIT
    assert upstream["answer_template_sha256"] == UPSTREAM_TEMPLATE_SHA256["answer"]
    assert set(upstream["judge_templates_sha256"].values()) == {
        UPSTREAM_TEMPLATE_SHA256[name] for name in UPSTREAM_TEMPLATE_SHA256 if name != "answer"
    }


def test_code_revision_ignores_the_manifest_the_run_itself_writes(tmp_path, monkeypatch):
    import subprocess

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "tracked.txt").write_text("x")
    git("add", "tracked.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    manifest = tmp_path / "docs" / "benchmarks" / "longmemeval-manifest.json"
    monkeypatch.setattr(lme, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(lme, "DEFAULT_MANIFEST", manifest)
    lme.update_manifest(manifest, "subset", {})
    assert lme.code_revision()["dirty"] is False
    (tmp_path / "tracked.txt").write_text("changed")
    assert lme.code_revision()["dirty"] is True


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


def test_retrieve_and_score_record_the_code_revision_from_the_start_of_the_run(
    tmp_path, monkeypatch
):
    dataset = synthetic_dataset()
    for question in dataset:
        question["answer_session_ids"] = ["s1"]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    qid = lme.select_subset(dataset)[0]["question_id"]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    packet = {
        "question_id": qid,
        "question": "q?",
        "question_date": "2023/06/01 (Thu) 10:00",
        "hits": [],
        "load": {"submitted": 0},
    }
    lme.append_jsonl(lme.packets_path(data_dir, "baseline"), [packet])
    manifest = tmp_path / "manifest.json"
    common = ["--dataset", str(dataset_path), "--data-dir", str(data_dir)]
    common += ["--manifest", str(manifest)]

    revisions_captured_in_order(monkeypatch)
    lme.main(["retrieve", *common, "--questions", qid])
    assert lme.read_manifest(manifest)["retrieve"]["code"]["commit"] == "start"

    revisions_captured_in_order(monkeypatch)
    lme.main(["score", *common])
    written = lme.read_manifest(manifest)
    assert written["score"]["code"]["commit"] == "start"
    assert written["upstream"] == lme.upstream_manifest()


def test_score_reports_whichever_runs_have_packets(tmp_path, monkeypatch):
    dataset = synthetic_dataset()
    for question in dataset:
        question["answer_session_ids"] = ["s1"]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    qid = lme.select_subset(dataset)[0]["question_id"]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    packet = {
        "question_id": qid,
        "question": "q?",
        "question_date": "2023/06/01 (Thu) 10:00",
        "hits": [],
        "load": {"submitted": 0},
    }
    lme.append_jsonl(lme.packets_path(data_dir, "prefetch-gate-off"), [packet])
    lme.append_jsonl(data_dir / lme.SESSIONS_FILE, [])
    manifest = tmp_path / "manifest.json"
    revisions_captured_in_order(monkeypatch)
    lme.main(
        [
            "score",
            "--dataset",
            str(dataset_path),
            "--data-dir",
            str(data_dir),
            "--manifest",
            str(manifest),
        ]
    )
    report = json.loads((data_dir / "report.json").read_text())
    assert list(report["runs"]) == ["prefetch-gate-off"]
    assert lme.read_manifest(manifest)["score"]["questions_scored"] == 1


def test_the_extraction_summary_counts_provider_refused_units(tmp_path):
    question = make_question("q1", "multi-session", sessions=[("s1", D1), ("s2", D2)])
    rows = [
        {"session_id": "s1", "date": D1, "notes": 0, "provider_refused": "content_filter"},
        {"session_id": "s2", "date": D2, "notes": 0},
    ]
    for row in rows:
        row.update({name: 0 for name in lme.SESSION_TOTALS})
    lme.append_jsonl(tmp_path / lme.SESSIONS_FILE, rows)
    summary = lme._extraction_summary(tmp_path, ["q1"], {"q1": question})
    assert summary["units_extracted"] == 2
    assert summary["provider_refused_units"] == 1


def test_a_packet_hit_is_dated_by_the_note_occurred_at():
    class Hit:
        meta = {
            "id": "n1",
            "occurred_at": datetime(2023, 5, 20, 2, 21, tzinfo=timezone.utc).timestamp(),
        }
        ts = datetime(2026, 9, 30, tzinfo=timezone.utc).timestamp()
        score = 0.9
        text = "t"

    record = lme._hit_record(Hit(), {"n1": {("s1", D1)}})
    assert record["date"] == "2023/05/20 (Sat) 02:21"
