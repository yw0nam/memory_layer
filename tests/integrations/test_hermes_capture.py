"""The Hermes provider's session-end capture, loaded outside Hermes with a fake client.

The plugin package imports ``agent.memory_provider`` from Hermes, so the test
installs a stub for it and loads the package under a private name (its own
name collides with this repository's ``memory_base``).
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "integrations" / "hermes" / "memory_base"
MESSAGES = [
    {"role": "system", "content": "You are Natsume."},
    {"role": "user", "content": "My sister Emily moves to Busan next month."},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
    {"role": "tool", "tool_call_id": "c1", "content": "calendar lookup"},
    {"role": "assistant", "content": [{"type": "text", "text": "That is a big move."}]},
]


@pytest.fixture
def plugin(monkeypatch):
    agent = types.ModuleType("agent")
    memory_provider = types.ModuleType("agent.memory_provider")

    class MemoryProvider:
        pass

    memory_provider.MemoryProvider = MemoryProvider
    agent.memory_provider = memory_provider
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.memory_provider", memory_provider)
    spec = importlib.util.spec_from_file_location(
        "hermes_memory_base",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "hermes_memory_base", module)
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self, error=None):
        self.bodies: list[dict] = []
        self.error = error

    def store_conversation(self, body):
        self.bodies.append(body)
        if self.error is not None:
            raise self.error
        return {"id": "conv:1", "created": True, "job_id": "j"}


def _provider(plugin, monkeypatch, fake, config=None):
    monkeypatch.setattr(plugin, "_load_plugin_config", lambda: dict(config or {}))
    provider = plugin.MemoryBaseProvider()
    monkeypatch.setattr(provider, "_build_client", lambda: fake)
    return provider


def test_session_end_posts_the_user_and_assistant_turns(plugin, monkeypatch):
    fake = FakeClient()
    provider = _provider(plugin, monkeypatch, fake)
    provider.initialize("hermes-session-1")
    provider.on_session_end(MESSAGES)
    (body,) = fake.bodies
    assert body["origin"] == "hermes"
    assert body["external_session_id"] == "hermes-session-1"
    assert body["namespace"] == "personal"
    assert body["turns"] == [
        {"role": "user", "text": "My sister Emily moves to Busan next month."},
        {"role": "assistant", "text": "That is a big move."},
    ]
    assert body["started_at"] <= body["ended_at"]


def test_the_capture_namespace_comes_from_the_plugin_config(plugin, monkeypatch):
    fake = FakeClient()
    provider = _provider(plugin, monkeypatch, fake, {"capture_namespace": "family"})
    provider.initialize("s1")
    provider.on_session_end(MESSAGES)
    assert fake.bodies[0]["namespace"] == "family"


def test_a_session_switch_uploads_under_the_new_session_id(plugin, monkeypatch):
    fake = FakeClient()
    provider = _provider(plugin, monkeypatch, fake)
    provider.initialize("old-session")
    provider.on_session_switch("new-session")
    provider.on_session_end(MESSAGES)
    assert [body["external_session_id"] for body in fake.bodies] == ["new-session"]


def test_fewer_than_two_turns_uploads_nothing(plugin, monkeypatch):
    fake = FakeClient()
    provider = _provider(plugin, monkeypatch, fake)
    provider.initialize("s1")
    provider.on_session_end(MESSAGES[:2])
    assert fake.bodies == []


def test_a_failed_upload_is_logged_and_never_raised(plugin, monkeypatch, caplog):
    fake = FakeClient(error=RuntimeError("memory-base is down"))
    provider = _provider(plugin, monkeypatch, fake)
    provider.initialize("s1")
    with caplog.at_level(logging.WARNING):
        provider.on_session_end(MESSAGES)
    assert len(fake.bodies) == 1
    assert "memory-base is down" in caplog.text


def test_session_end_before_initialize_does_nothing(plugin, monkeypatch):
    fake = FakeClient()
    provider = _provider(plugin, monkeypatch, fake)
    provider.on_session_end(MESSAGES)
    assert fake.bodies == []
