"""Unit tests for generic JSON enrichment validation and retry behavior."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger

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


def test_summary_parser_normalizes_deduplicates_and_drops_malformed_tags():
    parsed = enrich.parse_summary_response(
        {"summary": " A card. ", "tags": [" Knowledge Base ", "INVALID_TAG!", 4, "knowledge base"]}
    )
    assert parsed == {"summary": "A card.", "tags": ["knowledge base"]}


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


@pytest.mark.parametrize(
    "responses",
    [
        ["{}", "{}"],
        [json.dumps({"summary": "Summary", "tags": []})] * 2,
        [RuntimeError("down"), RuntimeError("still down")],
    ],
)
def test_summary_enrichment_persistent_failure_is_fail_closed(monkeypatch, responses):
    fake, _ = _client(responses)
    monkeypatch.setattr(enrich, "llm_client", lambda: fake)
    with pytest.raises(enrich.EnrichmentError, match="failed after retry"):
        asyncio.run(enrich.summarize_and_tag("text", "context"))


def test_each_failed_attempt_logs_debug_record_with_exception(monkeypatch):
    fake, _ = _client([RuntimeError("down"), RuntimeError("still down")])
    monkeypatch.setattr(enrich, "llm_client", lambda: fake)
    records = []
    sink_id = loguru_logger.add(records.append, level="DEBUG", format="{level}|{message}")
    try:
        with pytest.raises(enrich.EnrichmentError):
            asyncio.run(enrich.summarize_and_tag("text", "context"))
    finally:
        loguru_logger.remove(sink_id)

    debug_records = [record for record in records if record.record["level"].name == "DEBUG"]
    assert len(debug_records) == 2
    assert all(record.record["exception"] is not None for record in debug_records)
    assert all("RuntimeError" in str(record) for record in debug_records)
