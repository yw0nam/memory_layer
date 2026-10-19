"""memory_base Hermes memory plugin — MemoryProvider interface.

Delivers the agent's own card and the user's card in the system prompt, fetched at each
session start, pre-injects a per-turn semantic prefetch over the memory-base REST API
into every conversation turn, and posts each primary turn to the turn lane.

Config via config.yaml (memory.memory_base):
  url                — memory-base REST API base URL (required)
  timeout            — request timeout in seconds (default: 5)
  top_k              — max prefetch search results (default: 5)
  min_score          — relevance floor for prefetch search (default: 0.25)
  api_key            — API key value, takes precedence over api_key_env (optional)
  namespace          — where this agent's turns land (default: `default`)
  api_key_env        — env var holding the API key (default: MEMORY_BASE_API_KEY)
  author             — the author slug this profile writes under and reads by default;
                       sent as the X-Author header (required, no default;
                       absent, the provider is unavailable and no card is fetched)
  user_entity        — slug of the user's entity; its card is delivered too (optional)
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
_DEFAULT_MIN_SCORE = 0.25
_ORIGIN = "hermes"
_MAX_REASONING_CHARS = 64_000
_RETRY_AFTER_SECONDS = 60
_MESSAGE_CHARS = 120
_SESSION_LINE = (
    "Session: hermes {session_id} (pass it as session_id with origin hermes "
    "when a note is distilled from this session)"
)


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


def _turn_reasoning(messages: list[dict[str, Any]] | None) -> str | None:
    """The reasoning of the assistant entries after the last user entry, cut to the bound."""
    entries = messages or []
    start = max((i for i, m in enumerate(entries) if m.get("role") == "user"), default=-1) + 1
    parts = [
        m.get("reasoning_content") or m.get("reasoning")
        for m in entries[start:]
        if m.get("role") == "assistant"
    ]
    text = "\n\n".join(p for p in parts if isinstance(p, str) and p)
    return text[:_MAX_REASONING_CHARS] or None


class MemoryBaseProvider(MemoryProvider):
    """The agent's and the user's cards at session start, semantic prefetch and turn capture."""

    def __init__(self) -> None:
        self._config = _load_plugin_config()
        self._client: client.MemoryBaseClient | None = None
        self._cards = ""
        self._session_id = ""
        self._primary = True
        self._warn = None
        self._reset_outage()

    def _reset_outage(self) -> None:
        self._unreachable = False
        self._refused: set[int] = set()
        self._retry_at = 0.0

    @property
    def name(self) -> str:
        return "memory_base"

    def _url(self) -> str:
        return str(self._config.get("url") or "")

    def _api_key(self) -> str:
        return client.resolve_api_key(self._config, os.environ)

    def is_available(self) -> bool:
        """Config presence only — no network I/O."""
        return bool(self._url() and self._api_key() and self._config.get("author"))

    def _build_client(self) -> client.MemoryBaseClient:
        return client.MemoryBaseClient(
            url=self._url(),
            api_key=self._api_key(),
            author=str(self._config.get("author") or ""),
            timeout=float(self._config.get("timeout", _DEFAULT_TIMEOUT)),
            top_k=int(self._config.get("top_k", _DEFAULT_TOP_K)),
            min_score=float(self._config.get("min_score", _DEFAULT_MIN_SCORE)),
        )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._client = self._build_client()
        self._session_id = session_id
        self._primary = kwargs.get("agent_context", "primary") == "primary"
        warn = kwargs.get("warning_callback")
        self._warn = warn if callable(warn) else None
        self._reset_outage()
        self._cards = self._build_cards()

    def _build_cards(self) -> str:
        author = str(self._config.get("author") or "")
        if not author:
            return ""
        return self._client.build_cards(
            author,
            str(self._config.get("user_entity") or "") or None,
            str(self._config.get("namespace") or "") or None,
        )

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id
        self._reset_outage()
        if self._client:
            self._cards = self._build_cards()

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""
        session = _SESSION_LINE.format(session_id=self._session_id)
        return "\n\n".join(part for part in (self._cards, session) if part)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._client:
            return ""
        try:
            return self._client.build_prefetch(query)
        except Exception:
            return ""

    def _notify(self, message: str) -> None:
        if self._warn:
            self._warn(message)
        else:
            logging.getLogger("memory_base").warning(message)

    def _reachable(self, session_id: str, role: str, content: str, reasoning: str | None) -> bool:
        """Post one turn; whether the server answered, even with a refusal."""
        try:
            self._client.post_turn(
                _ORIGIN,
                session_id,
                role,
                content,
                reasoning,
                str(self._config.get("namespace") or "") or None,
            )
        except client.Unreachable:
            self._retry_at = time.monotonic() + _RETRY_AFTER_SECONDS
            if not self._unreachable:
                self._unreachable = True
                self._notify("memory-base unreachable; turns are not being saved")
            return False
        except client.Refused as exc:
            if exc.status not in self._refused:
                self._refused.add(exc.status)
                self._notify(
                    f"memory-base refused a turn ({exc.status}: {exc.message[:_MESSAGE_CHARS]})"
                )
        if self._unreachable:
            self._unreachable = False
            self._notify("memory-base is reachable again; turns are being saved")
        return True

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self._primary or not self._client or time.monotonic() < self._retry_at:
            return
        session = session_id or self._session_id
        try:
            if self._reachable(session, "user", user_content, None):
                reasoning = _turn_reasoning(messages)
                self._reachable(session, "assistant", assistant_content, reasoning)
        except Exception:
            return

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "url",
                "description": "memory-base API URL",
                "required": True,
                "default": "http://127.0.0.1:8010",
            },
            {
                "key": "api_key",
                "description": "memory-base API key",
                "secret": True,
                "required": True,
                "env_var": client.DEFAULT_API_KEY_ENV,
            },
            {
                "key": "author",
                "description": (
                    "The author slug this profile writes under and reads by default; "
                    "sent as the X-Author header"
                ),
                "required": True,
            },
            {
                "key": "user_entity",
                "description": "Slug of the user's entity; its card is delivered at session start",
            },
            {
                "key": "namespace",
                "description": "Namespace this agent's turns land in (empty: `default`)",
            },
            {
                "key": "top_k",
                "description": "Max prefetch search results",
                "type": "integer",
                "default": _DEFAULT_TOP_K,
            },
            {
                "key": "min_score",
                "description": "Relevance floor for prefetch search",
                "type": "number",
                "default": _DEFAULT_MIN_SCORE,
            },
            {
                "key": "timeout",
                "description": "Request timeout in seconds",
                "type": "number",
                "default": _DEFAULT_TIMEOUT,
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        from hermes_cli.config import load_config, save_config

        config = load_config()
        memory = config.get("memory")
        if not isinstance(memory, dict):
            memory = config["memory"] = {}
        current = memory.get("memory_base")
        saved = {k: v for k, v in values.items() if k != "api_key"}
        memory["memory_base"] = {**(current if isinstance(current, dict) else {}), **saved}
        save_config(config)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        pass


def register(ctx) -> None:
    """Register memory_base as a memory provider plugin."""
    ctx.register_memory_provider(MemoryBaseProvider())
