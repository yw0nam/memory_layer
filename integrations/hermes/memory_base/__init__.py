"""memory_base Hermes memory plugin — MemoryProvider interface.

Pre-injects a per-turn semantic prefetch over the memory-base REST API into
every conversation turn, and uploads each session's user and assistant turns
at session end so the server distills them into notes.

Config via config.yaml (memory.memory_base):
  url                — memory-base REST API base URL (required)
  timeout            — request timeout in seconds (default: 5)
  top_k              — max prefetch search results (default: 5)
  min_score          — relevance floor for prefetch search (default: 0.6)
  api_key            — API key value, takes precedence over api_key_env (optional)
  api_key_env        — env var holding the API key (default: MEMORY_BASE_API_KEY)
  capture_namespace  — namespace sessions are captured into (default: personal)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from agent.memory_provider import MemoryProvider

from . import client

_DEFAULT_TIMEOUT = 5
_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SCORE = 0.6
_DEFAULT_CAPTURE_NAMESPACE = "personal"
_MIN_CAPTURE_TURNS = 2

logger = logging.getLogger(__name__)


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
    """Semantic prefetch every turn; the session's turns are captured at its end."""

    def __init__(self) -> None:
        self._config = _load_plugin_config()
        self._client: client.MemoryBaseClient | None = None
        self._session_id = ""
        self._session_started = 0.0

    @property
    def name(self) -> str:
        return "memory_base"

    def _url(self) -> str:
        return str(self._config.get("url") or "")

    def _api_key(self) -> str:
        return client.resolve_api_key(self._config, os.environ)

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

    def initialize(self, session_id: str, **kwargs) -> None:
        self._client = self._build_client()
        self._session_id = session_id
        self._session_started = time.time()

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._client:
            return ""
        try:
            return self._client.build_prefetch(query)
        except Exception:
            return ""

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id
        self._session_started = time.time()

    # -- No tools, no per-turn writes; the whole session is captured at its end.

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
        if not self._client or not self._session_id:
            return
        try:
            turns = client.conversation_turns(messages)
            if len(turns) < _MIN_CAPTURE_TURNS:
                return
            self._client.store_conversation(
                {
                    "origin": "hermes",
                    "external_session_id": self._session_id,
                    "namespace": str(
                        self._config.get("capture_namespace") or _DEFAULT_CAPTURE_NAMESPACE
                    ),
                    "started_at": self._session_started,
                    "ended_at": max(time.time(), self._session_started),
                    "turns": turns,
                }
            )
        except Exception as exc:
            logger.warning("memory_base: session %s was not captured: %s", self._session_id, exc)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        pass


def register(ctx) -> None:
    """Register memory_base as a memory provider plugin."""
    ctx.register_memory_provider(MemoryBaseProvider())
