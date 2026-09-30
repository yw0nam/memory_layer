"""Unit coverage for the eval-only agent writer (fake search, fake save, fake model)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from memory_base.eval import agent_writer as aw
from memory_base.eval import longmemeval as lme

NAMESPACE = "lme-q1"
SAVE_KWARGS = {
    "tags": ["longmemeval"],
    "kind": "note",
    "namespace": NAMESPACE,
    "occurred_at": "2023-05-20T02:21:00",
    "allow_similar": True,
}


class Hit:
    def __init__(self, note_id, text, day, score=0.9):
        ts = datetime(2023, 1, day, tzinfo=timezone.utc).timestamp()
        self.meta = {"id": note_id, "occurred_at": ts}
        self.text = text
        self.score = score


class FakeSearch:
    def __init__(self, hits=(), error=None):
        self.hits = list(hits)
        self.error = error
        self.calls = []

    async def __call__(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error is not None:
            raise self.error
        return self.hits


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def complete(self, prompt, *, max_tokens):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, 10, 2


class FakeSave:
    def __init__(self, refuse=None):
        self.calls = []
        self.refuse = refuse

    async def __call__(self, content, **kwargs):
        self.calls.append((content, kwargs))
        if self.refuse is not None and self.refuse in content:
            raise ValueError("refused")
        return {"id": f"note:{content}", "stored": True, "similar": []}


def writer(replies, hits=None, error=None):
    hits = [Hit("note:eggs", "30 dozen eggs.", 5)] if hits is None else hits
    return aw.AgentWriter(FakeModel(replies), FakeSearch(hits, error), "m", "medium")


def save_one(w, save, content="20 dozen eggs."):
    stats = lme.LoadStats()
    result = asyncio.run(w.save(save, content, stats=stats, date="2023-05-20", **SAVE_KWARGS))
    return stats, result


def supersede(index=0, content="20 dozen eggs as of 2023-05 (30 dozen as of 2023-01)."):
    return json.dumps({"action": "supersede", "index": index, "content": content})


NEW = json.dumps({"action": "new"})


def test_a_new_decision_saves_the_original_note_without_supersedes():
    w, save = writer([NEW]), FakeSave()
    stats, result = save_one(w, save)
    assert save.calls == [("20 dozen eggs.", SAVE_KWARGS)]
    assert result["id"] == "note:20 dozen eggs."
    assert (stats.agent_calls, stats.superseded, stats.agent_errors) == (1, 0, 0)
    query, kwargs = w.search.calls[0]
    assert query == "20 dozen eggs."
    assert kwargs == {"source": "memory", "namespaces": [NAMESPACE], "min_score": 0.25}
    prompt = w.model.prompts[0]
    assert "New note (2023-05-20):\n20 dozen eggs." in prompt
    assert "[0] (2023-01-05) 30 dozen eggs." in prompt


def test_a_supersede_decision_saves_the_rewrite_over_the_chosen_candidate():
    hits = [Hit("note:milk", "Milk.", 3), Hit("note:eggs", "30 dozen eggs.", 5)]
    w, save = writer(["```json\n" + supersede(1) + "\n```"], hits), FakeSave()
    stats, result = save_one(w, save)
    rewrite = "20 dozen eggs as of 2023-05 (30 dozen as of 2023-01)."
    assert save.calls == [(rewrite, {**SAVE_KWARGS, "supersedes": "note:eggs"})]
    assert result["id"] == f"note:{rewrite}"
    assert (stats.agent_calls, stats.superseded, stats.agent_errors) == (1, 1, 0)


def test_the_writer_sees_at_most_the_top_five_candidates_in_rerank_order():
    hits = [Hit(f"note:{i}", f"fact {i}", i + 1) for i in range(7)]
    w = writer([NEW], hits)
    save_one(w, FakeSave())
    prompt = w.model.prompts[0]
    assert "[4] (2023-01-05) fact 4" in prompt
    assert "fact 5" not in prompt
    assert prompt.index("[0]") < prompt.index("[1]")


def test_malformed_replies_are_retried_and_a_valid_one_is_applied():
    w, save = writer(["not json", "{}", supersede()]), FakeSave()
    stats, _ = save_one(w, save)
    assert save.calls[0][1]["supersedes"] == "note:eggs"
    assert (stats.agent_calls, stats.superseded, stats.agent_errors) == (3, 1, 0)


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps({"action": "supersede", "index": True, "content": "x"}),
        json.dumps({"action": "supersede", "index": 1, "content": "x"}),
        json.dumps({"action": "supersede", "index": -1, "content": "x"}),
        json.dumps({"action": "supersede", "index": 0, "content": "  "}),
        json.dumps({"action": "keep"}),
        json.dumps(["new"]),
        RuntimeError("claude exited 1"),
    ],
)
def test_three_failed_attempts_save_the_original_as_new(bad):
    w, save = writer([bad, bad, bad]), FakeSave()
    stats, _ = save_one(w, save)
    assert save.calls == [("20 dozen eggs.", SAVE_KWARGS)]
    assert (stats.agent_calls, stats.superseded, stats.agent_errors) == (3, 0, 1)


def test_parse_decision_accepts_both_actions_and_ignores_extra_keys():
    assert aw.parse_decision('{"action": "new", "why": "x"}', 2) == ("new", None, None)
    assert aw.parse_decision(supersede(1, "rewrite"), 2) == ("supersede", 1, "rewrite")
    with pytest.raises(ValueError):
        aw.parse_decision(supersede(2, "rewrite"), 2)


def test_no_candidates_means_no_model_call_and_a_plain_save():
    w, save = writer([], hits=[]), FakeSave()
    stats, _ = save_one(w, save)
    assert w.model.prompts == []
    assert save.calls == [("20 dozen eggs.", SAVE_KWARGS)]
    assert (stats.agent_calls, stats.superseded, stats.agent_errors) == (0, 0, 0)


def test_a_failing_search_saves_the_note_plainly_and_counts_an_error():
    w, save = writer([], error=RuntimeError("rerank down")), FakeSave()
    stats, _ = save_one(w, save)
    assert w.model.prompts == []
    assert save.calls == [("20 dozen eggs.", SAVE_KWARGS)]
    assert (stats.agent_calls, stats.agent_errors) == (0, 1)


def test_a_refused_rewrite_falls_back_to_saving_the_original_as_new():
    w, save = writer([supersede(content="rewrite refused")]), FakeSave(refuse="refused")
    stats, result = save_one(w, save)
    assert save.calls == [
        ("rewrite refused", {**SAVE_KWARGS, "supersedes": "note:eggs"}),
        ("20 dozen eggs.", SAVE_KWARGS),
    ]
    assert result["id"] == "note:20 dozen eggs."
    assert (stats.superseded, stats.agent_errors) == (0, 1)


def test_the_config_names_the_writer_and_hashes_both_prompts():
    w = writer([])
    assert w.config() == {
        "kind": "agent",
        "model": "m",
        "effort": "medium",
        "prompt_sha": lme.prompt_sha(aw.WRITER_SYSTEM_PROMPT + "\n" + aw.WRITER_PROMPT),
    }


UNITS = [("s1", "2023/05/20 (Sat) 02:21")]
NOTES = {UNITS[0]: [{"content": "20 dozen eggs.", "kind": "note", "gate": "stored"}]}


def test_loading_without_a_writer_never_searches_or_calls_the_model():
    w, save = writer([NEW]), FakeSave()
    stats, provenance = asyncio.run(lme.load_question_notes(NAMESPACE, UNITS, NOTES, save=save))
    assert w.search.calls == [] and w.model.prompts == []
    assert save.calls == [("20 dozen eggs.", SAVE_KWARGS)]
    assert (stats.agent_calls, stats.superseded, stats.agent_errors) == (0, 0, 0)
    assert provenance == {"note:20 dozen eggs.": {UNITS[0]}}


def test_loading_with_a_writer_routes_each_save_through_it_and_maps_the_rewrite():
    w, save = writer([supersede(content="rewrite")]), FakeSave()
    stats, provenance = asyncio.run(
        lme.load_question_notes(NAMESPACE, UNITS, NOTES, save=save, writer=w)
    )
    assert save.calls == [("rewrite", {**SAVE_KWARGS, "supersedes": "note:eggs"})]
    assert "New note (2023-05-20)" in w.model.prompts[0]
    assert (stats.submitted, stats.stored, stats.superseded, stats.agent_calls) == (1, 1, 1, 1)
    assert provenance == {"note:rewrite": {UNITS[0]}}
