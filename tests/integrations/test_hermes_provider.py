"""The Hermes provider's prefetch, loaded outside Hermes.

The plugin package imports ``agent.memory_provider`` from Hermes, so the test
installs a stub for it and loads the package under a private name (its own
name collides with this repository's ``memory_base``).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "integrations" / "hermes" / "memory_base"


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


def test_the_default_client_is_built_with_the_benchmarked_prefetch_floor(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_load_plugin_config", lambda: {})
    client = plugin.MemoryBaseProvider()._build_client()
    assert client.min_score == 0.25
    assert client.top_k == 5


def test_the_provider_does_not_capture_sessions(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_load_plugin_config", lambda: {})
    provider = plugin.MemoryBaseProvider()
    assert not hasattr(provider, "on_session_end")
    assert not hasattr(provider, "on_session_switch")
    assert not hasattr(plugin.client.MemoryBaseClient, "store_conversation")
    assert not hasattr(plugin.client, "conversation_turns")
