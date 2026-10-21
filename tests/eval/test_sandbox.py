"""Unit coverage for the throwaway REST API sandbox."""

from __future__ import annotations

import asyncio
import signal
from contextlib import contextmanager

import pytest

from memory_base.eval import sandbox


def _fakes(monkeypatch, events):
    @contextmanager
    def database():
        events.append("db in")
        try:
            yield {"image": "img", "url": "postgresql://fake"}
        finally:
            events.append("db out")

    class Api:
        url = "http://127.0.0.1:1"

        def __enter__(self):
            events.append("api in")
            return self

        def __exit__(self, *exc):
            events.append("api out")

    async def prepare(url):
        return "KEY123"

    monkeypatch.setattr(sandbox, "throwaway_postgres", database)
    monkeypatch.setattr(sandbox, "EvalApi", Api)
    monkeypatch.setattr(sandbox, "_prepare", prepare)


def test_main_prints_the_url_and_key_and_waits_inside_both_contexts(monkeypatch, capsys):
    events: list[str] = []
    _fakes(monkeypatch, events)
    sandbox.main(wait=lambda: events.append("wait"))
    assert events == ["db in", "api in", "wait", "api out", "db out"]
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == ["URL http://127.0.0.1:1", "KEY KEY123", "AUTHOR claude-code"]


def test_an_interrupt_stops_the_api_then_the_database_and_restores_handlers(monkeypatch, capsys):
    events: list[str] = []
    _fakes(monkeypatch, events)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    before_term = signal.getsignal(signal.SIGTERM)
    seen = {}

    def wait():
        seen["int"] = signal.getsignal(signal.SIGINT)
        seen["term"] = signal.getsignal(signal.SIGTERM)
        raise KeyboardInterrupt

    try:
        sandbox.main(wait=wait)
        after_int = signal.getsignal(signal.SIGINT)
        after_term = signal.getsignal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    assert events == ["db in", "api in", "api out", "db out"]
    assert "sandbox stopped" in capsys.readouterr().err
    assert seen["int"] is signal.default_int_handler
    assert seen["term"] not in (before_term, signal.SIG_IGN, signal.SIG_DFL)
    assert after_int == signal.SIG_IGN
    assert after_term == before_term


def test_prepare_closes_the_pool_when_minting_the_key_fails(monkeypatch):
    closed = []

    async def schema(url):
        return []

    async def new_key(label):
        raise RuntimeError("no key")

    async def close_pool():
        closed.append(True)

    monkeypatch.setattr(sandbox, "_prepare_schema", schema)
    monkeypatch.setattr(sandbox.keys, "new_key", new_key)
    monkeypatch.setattr(sandbox.db, "close_pool", close_pool)
    with pytest.raises(RuntimeError, match="no key"):
        asyncio.run(sandbox._prepare("postgresql://fake"))
    assert closed == [True]
