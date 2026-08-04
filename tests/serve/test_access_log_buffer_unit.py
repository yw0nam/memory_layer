"""Unit tests for the buffered retrieval write-back (memory_base.serve.access_log).

No DB, no network: ``memory_base.core.db.acquire`` is replaced by a connection
that records every statement, so these tests pin what the search request path
issues (nothing) and what the interval flusher issues (batched, deduplicated
writes plus an hourly retention delete).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib

import pytest
from starlette.testclient import TestClient

from memory_base.core import db
from memory_base.retrieval.decompose import DeepResult, EvidenceEntry
from memory_base.retrieval.search import Hit
from memory_base.serve import access_log, api

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

WRITE_VERBS = ("INSERT", "UPDATE", "DELETE")
NOW = 1_700_000_000.0
# tests/conftest.py stubs the lifespan flusher; these tests drive the real one.
START_FLUSHER = access_log.start_flusher
STOP_FLUSHER = access_log.stop_flusher


class RecordingConnection:
    """Collects statements instead of running them."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple]] = []

    async def execute(self, sql: str, *args):
        self.statements.append((sql, args))
        return "OK"

    async def executemany(self, sql: str, args):
        self.statements.append((sql, tuple(args)))
        return None

    def writes(self) -> list[str]:
        return [
            sql
            for sql, _ in self.statements
            if sql.strip().split()[0].upper().startswith(WRITE_VERBS)
        ]

    def matching(self, verb: str) -> list[tuple[str, tuple]]:
        return [
            (sql, args) for sql, args in self.statements if sql.strip().upper().startswith(verb)
        ]


@pytest.fixture()
def connection(monkeypatch) -> RecordingConnection:
    recording = RecordingConnection()

    @contextlib.asynccontextmanager
    async def fake_acquire(*args, **kwargs):
        yield recording

    monkeypatch.setattr(db, "acquire", fake_acquire)
    return recording


@pytest.fixture(autouse=True)
def empty_buffer(monkeypatch):
    """Every test starts with an empty buffer and no retention run yet."""
    monkeypatch.setattr(access_log, "_pending_logs", [])
    monkeypatch.setattr(access_log, "_pending_hits", {})
    monkeypatch.setattr(access_log, "_last_retention", 0.0)


def _hit(chunk_id: str) -> Hit:
    return Hit(
        source="memory",
        ref=f"ref/{chunk_id}",
        text="body",
        ts=1_700_000_000.0,
        meta={"id": chunk_id},
    )


# ---- the request path writes nothing ---------------------------------------


def test_search_request_performs_no_db_writes(monkeypatch, connection):
    async def fake_search(query, **options):
        return [_hit("chunk-a"), _hit("chunk-b")]

    monkeypatch.setattr(api, "search", fake_search)

    response = client.post("/search", json={"query": "q", "source": "memory"})

    assert response.status_code == 200
    assert connection.writes() == []
    assert {chunk: count for chunk, (count, _) in access_log._pending_hits.items()} == {
        "chunk-a": 1,
        "chunk-b": 1,
    }
    assert len(access_log._pending_logs) == 1


def test_deep_search_request_performs_no_db_writes(monkeypatch, connection):
    entry = EvidenceEntry(
        id="chunk-deep",
        ref="ref/chunk-deep",
        text="body",
        kind="note",
        tags=[],
        date=1_700_000_000.0,
        hop=1,
        atom_question=None,
    )

    async def fake_deep_search(question, **options):
        return DeepResult(evidence=[entry], trace=[], hops_used=1, stopped_reason="done")

    monkeypatch.setattr(api, "deep_search", fake_deep_search)

    response = client.post("/search/deep", json={"query": "q"})

    assert response.status_code == 200
    assert connection.writes() == []
    assert set(access_log._pending_hits) == {"chunk-deep"}


# ---- the flusher writes batched, deduplicated statements --------------------


def test_repeated_hits_on_one_chunk_collapse_into_one_update(connection):
    for _ in range(3):
        access_log.record_retrieval("q", "memory", [_hit("chunk-a")], now=NOW)
    access_log.record_retrieval("q", "memory", [_hit("chunk-b")], now=NOW)

    asyncio.run(access_log.flush(now=NOW))

    updates = connection.matching("UPDATE")
    assert len(updates) == 1
    _, args = updates[0]
    counts = dict(zip(args[0], args[1]))
    assert counts == {"chunk-a": 3, "chunk-b": 1}


def test_flush_writes_one_retrieval_log_row_per_recorded_search(connection):
    access_log.record_retrieval("first", "memory", [_hit("chunk-a")], now=NOW)
    access_log.record_retrieval("second", "all", [_hit("chunk-b")], now=NOW + 1)

    asyncio.run(access_log.flush(now=NOW + 1))

    inserts = connection.matching("INSERT")
    assert len(inserts) == 1
    _, rows = inserts[0]
    assert [row[0] for row in rows] == ["first", "second"]
    assert [row[1] for row in rows] == ["memory", "all"]


def test_flush_drains_the_buffer(connection):
    access_log.record_retrieval("q", "memory", [_hit("chunk-a")], now=NOW)
    asyncio.run(access_log.flush(now=NOW))
    connection.statements.clear()

    asyncio.run(access_log.flush(now=NOW + 100))

    assert connection.statements == []


def test_flush_survives_a_failing_write(monkeypatch):
    @contextlib.asynccontextmanager
    async def failing_acquire(*args, **kwargs):
        raise RuntimeError("pool down")
        yield  # pragma: no cover

    monkeypatch.setattr(db, "acquire", failing_acquire)
    access_log.record_retrieval("q", "memory", [_hit("chunk-a")], now=NOW)

    asyncio.run(access_log.flush(now=NOW))

    assert access_log._pending_hits == {}


# ---- retention moves off the request path ----------------------------------


def test_retention_delete_runs_at_most_once_per_hour(connection):
    access_log.record_retrieval("q", "memory", [_hit("chunk-a")], now=NOW)
    asyncio.run(access_log.flush(now=NOW))
    assert len(connection.matching("DELETE")) == 1

    access_log.record_retrieval("q", "memory", [_hit("chunk-a")], now=NOW + 100)
    asyncio.run(access_log.flush(now=NOW + 100))
    assert len(connection.matching("DELETE")) == 1

    access_log.record_retrieval("q", "memory", [_hit("chunk-a")], now=NOW + 3601)
    asyncio.run(access_log.flush(now=NOW + 3601))
    assert len(connection.matching("DELETE")) == 2


def test_retention_delete_runs_with_an_empty_buffer(connection):
    asyncio.run(access_log.flush(now=NOW))

    assert len(connection.matching("DELETE")) == 1
    assert connection.matching("INSERT") == []


# ---- flusher lifecycle ------------------------------------------------------


def test_flusher_loop_flushes_on_the_configured_interval(monkeypatch, connection):
    """A 10ms interval flushes within 100ms; the 30s default never would."""
    monkeypatch.setattr(access_log, "HIT_FLUSH_INTERVAL_SECONDS", 0.01)

    async def _run() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(access_log.flusher_loop(stop=stop))
        access_log.record_retrieval("q", "memory", [_hit("chunk-a")], now=NOW)
        await asyncio.sleep(0.1)
        stop.set()
        await task

    asyncio.run(_run())

    assert connection.matching("UPDATE")
    assert access_log._pending_hits == {}


def test_stop_flusher_flushes_pending_hits(connection):
    async def _run() -> None:
        task = START_FLUSHER()
        access_log.record_retrieval("q", "memory", [_hit("chunk-a")], now=NOW)
        await STOP_FLUSHER(task)

    asyncio.run(_run())

    assert connection.matching("UPDATE")
    assert access_log._pending_hits == {}


def test_flush_interval_is_env_configurable_with_a_default(monkeypatch):
    monkeypatch.delenv("HIT_FLUSH_INTERVAL_SECONDS", raising=False)
    try:
        assert importlib.reload(access_log).HIT_FLUSH_INTERVAL_SECONDS == 30.0
        monkeypatch.setenv("HIT_FLUSH_INTERVAL_SECONDS", "5")
        assert importlib.reload(access_log).HIT_FLUSH_INTERVAL_SECONDS == 5.0
    finally:
        monkeypatch.undo()
        importlib.reload(access_log)
