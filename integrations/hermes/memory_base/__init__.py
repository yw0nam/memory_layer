"""memory_base Hermes memory plugin — MemoryProvider interface.

Pre-injects memory-base content into every conversation: a session-start
digest of recent episode notes, plus a per-turn semantic prefetch over the
memory-base REST API.

Config via config.yaml (memory.memory_base):
  url          — memory-base REST API base URL (required)
  timeout      — request timeout in seconds (default: 5)
  top_k        — max prefetch search results (default: 5)
  min_score    — relevance floor for prefetch search (default: 0.6)
  api_key_env  — env var holding the API key (default: MEMORY_BASE_API_KEY)
"""

from __future__ import annotations

import os
from typing import Any

from agent.memory_provider import MemoryProvider

from . import client

_DEFAULT_TIMEOUT = 5
_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SCORE = 0.6
_DEFAULT_API_KEY_ENV = "MEMORY_BASE_API_KEY"


def _load_plugin_config() -> dict[str, Any]:
    """Read the profile-scoped ``memory.memory_base`` config subtree."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        memory_config = config.get("memory", {}) if isinstance(config, dict) else {}
        provider_config = memory_config.get("memory_base", {})
        return dict(provider_config) if isinstance(provider_config, dict) else {}
    except Exception:
        return {}


class MemoryBaseProvider(MemoryProvider):
    """Episode digest at session start, semantic prefetch every turn."""

    def __init__(self) -> None:
        self._config = _load_plugin_config()
        self._client: client.MemoryBaseClient | None = None
        self._digest = ""
        self._digest_ids: set[str] = set()

    @property
    def name(self) -> str:
        return "memory_base"

    def _url(self) -> str:
        return str(self._config.get("url") or "")

    def _api_key(self) -> str:
        env_var = str(self._config.get("api_key_env") or _DEFAULT_API_KEY_ENV)
        return os.environ.get(env_var, "")

    def is_available(self) -> bool:
        """Config presence only — no network I/O."""
        return bool(self._url() and self._api_key())

    def _build_client(self) -> client.MemoryBaseClient:
        return client.MemoryBaseClient(
            url=self._url(),
            api_key=self._api_key(),
            timeout=self._config.get("timeout", _DEFAULT_TIMEOUT),
            top_k=self._config.get("top_k", _DEFAULT_TOP_K),
            min_score=self._config.get("min_score", _DEFAULT_MIN_SCORE),
        )

    def _refresh_digest(self) -> None:
        episodes = self._client.recent_episodes(5) if self._client else []
        self._digest = client.format_digest(episodes)
        self._digest_ids = client.digest_identities(episodes)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._client = self._build_client()
        self._refresh_digest()

    def system_prompt_block(self) -> str:
        """Byte-stable within a session — the digest fetched at initialize()."""
        return self._digest

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._client:
            return ""
        try:
            return self._client.build_prefetch(query, self._digest_ids)
        except Exception:
            return ""

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._refresh_digest()

    # -- Context-only provider: no tools, no writes. -------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        pass

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        pass

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        pass


def register(ctx) -> None:
    """Register memory_base as a memory provider plugin."""
    ctx.register_memory_provider(MemoryBaseProvider())
