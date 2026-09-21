"""Pure REST client for the memory-base API's ``/search`` route.

No Hermes imports — importable and testable standalone (stdlib + httpx only).
Every network-facing call swallows errors and returns an empty result instead
of raising, since callers run inside a Hermes turn and must never block or
crash a conversation on a memory-base outage.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

PREFETCH_CHAR_BUDGET = 2000
DEFAULT_API_KEY_ENV = "MEMORY_BASE_API_KEY"

_CLIENT_CONTEXT_BLOCK = re.compile(r"<client_context>\n?.*?</client_context>\s*", re.DOTALL)
_DESIRE_TICK_MARKERS = ("MONITOR CHANGE DETECTED", "DESIRE_STATE_DIR")


@dataclass
class MemoryBaseClient:
    """Talks to a memory-base deployment's REST API over X-API-Key auth."""

    url: str
    api_key: str
    timeout: float = 5.0
    top_k: int = 5
    min_score: float = 0.6
    transport: httpx.BaseTransport | None = None

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(f"{self.url}{path}", json=body, headers=self._headers())
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def search(self, query: str) -> list[dict[str, Any]]:
        """Semantic search over memory notes. [] on any error."""
        data = self._post(
            "/search",
            {"query": query, "source": "memory", "top_k": self.top_k, "min_score": self.min_score},
        )
        return data if isinstance(data, list) else []

    def build_prefetch(self, query: str) -> str:
        """Search hits for the cleaned query, one line per hit, truncated to budget."""
        cleaned = clean_prefetch_query(query)
        if not cleaned:
            return ""
        lines = [
            f"- [{hit.get('date', '')}] {hit['text']}"
            for hit in self.search(cleaned)
            if hit.get("text")
        ]
        if not lines:
            return ""
        return _truncate_at_line_boundary("\n".join(lines), PREFETCH_CHAR_BUDGET)


def clean_prefetch_query(text: str) -> str:
    """Drop the client-injected context block, searching on what the turn itself says.

    A client prepends a ``<client_context>`` block it labels as not typed by
    the user: timestamp, frontmost app, posture, trigger, event payloads. Fed
    to search verbatim it self-matches notes about that machinery rather than
    the turn's subject, so none of it belongs in the query. Which fields the
    block holds is the client's business and changes without notice, so the
    whole block goes; a turn left with nothing skips the search.

    A desire tick is dropped whole. It names the prompt file that already tells
    it how to act, so a search over it returns a copy of that file at best.
    """
    if any(marker in text for marker in _DESIRE_TICK_MARKERS):
        return ""
    return _CLIENT_CONTEXT_BLOCK.sub("", text).strip()


def resolve_api_key(config: Mapping[str, Any], environ: Mapping[str, str]) -> str:
    """Resolve the memory-base API key: a configured value beats the ambient env var."""
    configured = config.get("api_key")
    if configured:
        return str(configured)
    env_var = str(config.get("api_key_env") or DEFAULT_API_KEY_ENV)
    return environ.get(env_var, "")


def _truncate_at_line_boundary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    cut = truncated.rfind("\n")
    return truncated[:cut] if cut > 0 else ""
