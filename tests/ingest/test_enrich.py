"""Unit tests for generic JSON enrichment validation and retry behavior."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from memory_base.ingest import enrich


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])


def _client(responses):
    completions = FakeCompletions(responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_atom_parser_caps_deduplicates_and_drops_pronouns_and_long_questions():
    parsed = enrich.parse_atom_response(
        {
            "atom_questions": [
                "What does Project Atlas store?",
                "what does project atlas store?",
                "How does it work?",
                *[f"What feature {index} does Project Atlas support?" for index in range(20)],
                "x" * 201,
            ],
            "tags": [" Knowledge Base ", "INVALID_TAG!", 4, "knowledge base", "ai"],
        }
    )
    assert parsed is not None
    assert len(parsed["atom_questions"]) == 10
    assert parsed["atom_questions"][0] == "What does Project Atlas store?"
    assert parsed["tags"] == ["knowledge base", "ai"]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"atom_questions": "bad", "tags": ["valid tag"]},
        {"atom_questions": [], "tags": "bad"},
        {"atom_questions": [], "tags": ["!"]},
    ],
)
def test_atom_parser_rejects_wrong_shape_and_zero_valid_tags(payload):
    assert enrich.parse_atom_response(payload) is None


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"summary": "", "tags": ["valid tag"]},
        {"summary": "Summary", "tags": "bad"},
        {"summary": "Summary", "tags": []},
    ],
)
def test_summary_parser_rejects_wrong_shape_empty_summary_and_zero_tags(payload):
    assert enrich.parse_summary_response(payload) is None


def test_atom_enrichment_retries_invalid_json_once(monkeypatch):
    fake, completions = _client(
        ["not json", json.dumps({"atom_questions": [], "tags": ["knowledge base"]})]
    )
    retries = []
    monkeypatch.setattr(enrich, "llm_client", lambda: fake)
    result = asyncio.run(
        enrich.atomize_and_tag("text", "context", on_retry=lambda: retries.append(True))
    )
    assert result == {"atom_questions": [], "tags": ["knowledge base"]}
    assert completions.calls == 2
    assert retries == [True]


@pytest.mark.parametrize(
    "responses",
    [
        ["{}", "{}"],
        [json.dumps({"atom_questions": [], "tags": []})] * 2,
        [RuntimeError("down"), RuntimeError("still down")],
    ],
)
def test_atom_enrichment_persistent_failure_is_fail_closed(monkeypatch, responses):
    fake, _ = _client(responses)
    monkeypatch.setattr(enrich, "llm_client", lambda: fake)
    with pytest.raises(enrich.EnrichmentError, match="failed after retry"):
        asyncio.run(enrich.atomize_and_tag("text", "context"))


def test_summary_enrichment_retries_empty_summary(monkeypatch):
    fake, completions = _client(
        [
            json.dumps({"summary": " ", "tags": ["table"]}),
            json.dumps({"summary": "An English table card.", "tags": ["table"]}),
        ]
    )
    monkeypatch.setattr(enrich, "llm_client", lambda: fake)
    result = asyncio.run(enrich.summarize_and_tag("text", "context"))
    assert result["summary"] == "An English table card."
    assert completions.calls == 2


def test_atoms_generate_false_keeps_tagging_and_suppresses_atoms(monkeypatch):
    fake, completions = _client(
        [json.dumps({"atom_questions": ["What does Project Atlas store?"], "tags": ["atlas"]})]
    )
    monkeypatch.setenv("ATOMS_GENERATE", "false")
    monkeypatch.setattr(enrich, "llm_client", lambda: fake)
    result = asyncio.run(enrich.atomize_and_tag("text", "context"))
    assert result == {"atom_questions": [], "tags": ["atlas"]}
    assert completions.calls == 1
