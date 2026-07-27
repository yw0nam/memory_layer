"""Unit tests for the process-lifetime schema setup guard (no DB needed)."""

from __future__ import annotations

import asyncio

import pytest

from memory_base import common, schema


class FakeConnection:
    """Stub connection that only records executed statements."""

    def __init__(self):
        self.calls = 0

    async def execute(self, query, *args):
        self.calls += 1


@pytest.fixture(autouse=True)
def _reset_once_state(monkeypatch):
    """Isolate the module-level once-guard state and PG_SCHEMA across tests."""
    monkeypatch.setattr(schema, "_prepared_schemas", set())
    monkeypatch.setattr(common, "PG_SCHEMA", "test_schema")


def test_ensure_schema_once_runs_ddl_once_for_repeated_calls(monkeypatch):
    calls = 0

    async def fake_ensure_schema(conn):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(schema, "ensure_schema", fake_ensure_schema)
    conn = FakeConnection()

    asyncio.run(schema.ensure_schema_once(conn))
    asyncio.run(schema.ensure_schema_once(conn))

    assert calls == 1


def test_ensure_schema_once_reruns_after_schema_change(monkeypatch):
    calls = 0

    async def fake_ensure_schema(conn):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(schema, "ensure_schema", fake_ensure_schema)
    conn = FakeConnection()

    asyncio.run(schema.ensure_schema_once(conn))
    monkeypatch.setattr(common, "PG_SCHEMA", "other_schema")
    asyncio.run(schema.ensure_schema_once(conn))

    assert calls == 2


def test_ensure_schema_once_coalesces_concurrent_callers(monkeypatch):
    calls = 0

    async def fake_ensure_schema(conn):
        nonlocal calls
        await asyncio.sleep(0)  # yield so concurrent callers interleave before the flag is set
        calls += 1

    monkeypatch.setattr(schema, "ensure_schema", fake_ensure_schema)
    conn = FakeConnection()

    async def run_concurrent():
        await asyncio.gather(*(schema.ensure_schema_once(conn) for _ in range(10)))

    asyncio.run(run_concurrent())

    assert calls == 1


def test_ensure_schema_itself_runs_every_time_called_directly():
    conn = FakeConnection()

    asyncio.run(schema.ensure_schema(conn))
    asyncio.run(schema.ensure_schema(conn))

    assert conn.calls == 2
