"""Pure REST client for the memory-base API.

No Hermes imports — importable and testable standalone (stdlib + httpx only).
Every network-facing call swallows errors and returns an empty result instead
of raising, since callers run inside a Hermes turn and must never block or
crash a conversation on a memory-base outage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

DIGEST_TITLE = "Recent events (episodic memory):"
PREFETCH_CHAR_BUDGET = 2000


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

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.get(f"{self.url}{path}", params=params, headers=self._headers())
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(f"{self.url}{path}", json=body, headers=self._headers())
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def recent_episodes(self, limit: int = 5) -> list[dict[str, Any]]:
        """Newest-first episode notes, as returned by the server. [] on any error."""
        data = self._get("/notes", {"kind": "episode", "limit": limit})
        return data if isinstance(data, list) else []

    def search(self, query: str) -> list[dict[str, Any]]:
        """Semantic search over memory notes. [] on any error."""
        data = self._post(
            "/search",
            {"query": query, "source": "memory", "top_k": self.top_k, "min_score": self.min_score},
        )
        return data if isinstance(data, list) else []

    def build_prefetch(self, query: str, digest_ids: set[str]) -> str:
        """Search hits not already shown in the digest, one line per hit, truncated to budget.

        Search results carry no note id (see hit_to_dict in serve/api.py), so
        dedup compares each hit's exact ``text`` against the digest episodes'
        text instead.
        """
        lines = [
            f"- [{hit.get('date', '')}] {hit['text']}"
            for hit in self.search(query)
            if hit.get("text") and hit["text"] not in digest_ids
        ]
        if not lines:
            return ""
        return _truncate_at_line_boundary("\n".join(lines), PREFETCH_CHAR_BUDGET)


def format_digest(episodes: list[dict[str, Any]]) -> str:
    """Render episodes (newest-first input) as an oldest-first digest block."""
    if not episodes:
        return ""
    lines = [f"- [{e.get('date', '')}] {e.get('text', '')}" for e in reversed(episodes)]
    return DIGEST_TITLE + "\n" + "\n".join(lines)


def digest_identities(episodes: list[dict[str, Any]]) -> set[str]:
    """Exact-text identity set for digest episodes, used to dedup prefetch hits."""
    return {e["text"] for e in episodes if e.get("text")}


def _truncate_at_line_boundary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    cut = truncated.rfind("\n")
    return truncated[:cut] if cut > 0 else ""
