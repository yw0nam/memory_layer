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

_CLIENT_CONTEXT_BLOCK = re.compile(r"<client_context>\n?(.*?)</client_context>\s*", re.DOTALL)
_CONTEXT_BOILERPLATE_LINE = re.compile(r"^(time|frontmost|trigger|screenshot|Client-injected)\b")


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
    """Strip client_context boilerplate from a prefetch query, keeping semantic lines.

    YUI prepends a ``<client_context>`` block (timestamp, frontmost app,
    trigger) to every user message. Fed to search verbatim, that boilerplate
    self-matches notes describing the client_context format instead of what
    the turn is about. Event payload lines (agent event/detail, signal, body,
    cue note) stay — on autonomous turns they are the turn's only content.
    """

    def keep_semantic_lines(match: re.Match[str]) -> str:
        lines = [
            line
            for line in match.group(1).splitlines()
            if line.strip() and not _CONTEXT_BOILERPLATE_LINE.match(line.strip())
        ]
        return "\n".join(lines) + "\n" if lines else ""

    return _CLIENT_CONTEXT_BLOCK.sub(keep_semantic_lines, text).strip()


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
