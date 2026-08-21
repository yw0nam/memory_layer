"""Unit tests for the Hermes memory-base client — pure module, no Hermes imports.

Exercises client.py directly (importable via the ``pythonpath`` test config
entry, mirroring how tests/test_release.py imports scripts/release.py).
Network calls are stubbed with httpx.MockTransport, matching the pattern in
tests/serve/test_mcp_proxy.py.
"""

from __future__ import annotations

import httpx

from client import MemoryBaseClient
from client import clean_prefetch_query
from client import digest_identities
from client import format_digest
from client import resolve_api_key


def _client(handler, **kwargs):
    return MemoryBaseClient(
        url="http://memory-base.local",
        api_key="secret-key",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


# ---- format_digest ----------------------------------------------------------


def test_format_digest_orders_oldest_first_with_dates():
    episodes = [
        {"date": "2026-08-20", "text": "newest event"},
        {"date": "2026-08-10", "text": "oldest event"},
    ]
    digest = format_digest(episodes)
    assert digest == (
        "Recent events (episodic memory):\n- [2026-08-10] oldest event\n- [2026-08-20] newest event"
    )


def test_format_digest_empty_returns_empty_string():
    assert format_digest([]) == ""


# ---- recent_episodes ---------------------------------------------------------


def test_recent_episodes_returns_empty_list_on_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _client(handler)
    assert client.recent_episodes(limit=5) == []


def test_recent_episodes_returns_empty_list_on_http_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = _client(handler)
    assert client.recent_episodes(limit=5) == []


def test_recent_episodes_returns_empty_list_on_invalid_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    client = _client(handler)
    assert client.recent_episodes(limit=5) == []


def test_recent_episodes_requests_notes_with_kind_episode():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=[{"id": "note:1", "date": "2026-08-01", "text": "x"}])

    client = _client(handler)
    result = client.recent_episodes(limit=3)
    assert captured["path"] == "/notes"
    assert captured["params"] == {"kind": "episode", "limit": "3"}
    assert result == [{"id": "note:1", "date": "2026-08-01", "text": "x"}]


# ---- search -------------------------------------------------------------------


def test_search_returns_empty_list_on_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    client = _client(handler)
    assert client.search("what happened") == []


def test_search_returns_empty_list_on_http_500():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = _client(handler)
    assert client.search("what happened") == []


def test_search_returns_empty_list_on_invalid_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not json")

    client = _client(handler)
    assert client.search("what happened") == []


def test_search_posts_query_source_top_k_min_score():
    import json as jsonlib

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = jsonlib.loads(request.content)
        return httpx.Response(200, json=[])

    client = _client(handler, top_k=7, min_score=0.42)
    client.search("deploy failure")
    assert captured["path"] == "/search"
    assert captured["body"] == {
        "query": "deploy failure",
        "source": "memory",
        "top_k": 7,
        "min_score": 0.42,
    }


# ---- auth header --------------------------------------------------------------


def test_auth_header_sent_on_every_request():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-api-key"))
        return httpx.Response(200, json=[])

    client = _client(handler)
    client.recent_episodes()
    client.search("q")
    assert seen == ["secret-key", "secret-key"]


# ---- digest_identities --------------------------------------------------------


def test_digest_identities_collects_episode_text():
    episodes = [{"text": "a"}, {"text": "b"}, {"text": ""}]
    assert digest_identities(episodes) == {"a", "b"}


# ---- build_prefetch ------------------------------------------------------------


def test_build_prefetch_drops_digest_duplicate_hits():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"date": "2026-08-19", "text": "already in digest"},
                {"date": "2026-08-18", "text": "fresh hit"},
            ],
        )

    client = _client(handler)
    result = client.build_prefetch("q", {"already in digest"})
    assert result == "- [2026-08-18] fresh hit"


def test_build_prefetch_returns_empty_when_all_filtered():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"date": "2026-08-19", "text": "dup"}])

    client = _client(handler)
    assert client.build_prefetch("q", {"dup"}) == ""


def test_build_prefetch_truncates_to_2000_chars_at_line_boundary():
    long_text = "x" * 300
    hits = [{"date": "2026-08-01", "text": f"{long_text}-{i}"} for i in range(10)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=hits)

    client = _client(handler)
    result = client.build_prefetch("q", set())
    assert len(result) <= 2000
    assert result != ""
    for line in result.splitlines():
        assert line.startswith("- [2026-08-01] ")
    # Every surviving line is complete — no line was cut mid-way.
    assert result.splitlines()[-1].endswith(tuple(f"-{i}" for i in range(10)))


def test_build_prefetch_returns_empty_when_first_line_alone_exceeds_limit():
    hits = [{"date": "2026-08-01", "text": "y" * 3000}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=hits)

    client = _client(handler)
    assert client.build_prefetch("q", set()) == ""


# ---- resolve_api_key -----------------------------------------------------------


def test_resolve_api_key_prefers_configured_key_over_env():
    config = {"api_key": "from-config"}
    environ = {"MEMORY_BASE_API_KEY": "from-env"}
    assert resolve_api_key(config, environ) == "from-config"


def test_resolve_api_key_falls_back_to_default_env_var():
    config = {}
    environ = {"MEMORY_BASE_API_KEY": "from-env"}
    assert resolve_api_key(config, environ) == "from-env"


def test_resolve_api_key_falls_back_to_configured_env_var_name():
    config = {"api_key_env": "CUSTOM_KEY_VAR"}
    environ = {"CUSTOM_KEY_VAR": "from-custom-env"}
    assert resolve_api_key(config, environ) == "from-custom-env"


def test_resolve_api_key_empty_when_neither_configured_nor_in_env():
    assert resolve_api_key({}, {}) == ""


# ---- clean_prefetch_query -------------------------------------------------------


RAW_USER_TURN = (
    "<client_context>\n"
    "Client-injected context; not typed by the user.\n"
    "time: 2026-08-21T21:38:11+09:00 (Asia/Seoul)\n"
    "frontmost: Orca (for 1min)\n"
    "trigger: user message\n"
    "</client_context>\n\n"
    "それ、多分できるよ。"
)

RAW_AGENT_TURN = (
    "<client_context>\n"
    "Client-injected context; not typed by the user.\n"
    "time: 2026-08-21T21:38:07+09:00 (Asia/Seoul)\n"
    "frontmost: Orca (for 1min)\n"
    "trigger: agent catchup (1 events)\n"
    'agent event: claude-code done, project "YUI" - "Polled v0.3.2 build completion." (3min ago)\n'
    "agent detail: 빌드가 아직 도는 중.\n"
    "</client_context>\n\n"
    "(my claude-code tasks piled up while I was away)"
)


def test_clean_prefetch_query_drops_boilerplate_keeps_user_text():
    assert clean_prefetch_query(RAW_USER_TURN) == "それ、多分できるよ。"


def test_clean_prefetch_query_keeps_semantic_context_lines():
    cleaned = clean_prefetch_query(RAW_AGENT_TURN)
    assert cleaned == (
        'agent event: claude-code done, project "YUI" - "Polled v0.3.2 build completion." (3min ago)\n'
        "agent detail: 빌드가 아직 도는 중.\n"
        "(my claude-code tasks piled up while I was away)"
    )
    assert "<client_context>" not in cleaned
    assert "frontmost" not in cleaned


def test_clean_prefetch_query_passes_plain_text_through():
    assert clean_prefetch_query("no context block here") == "no context block here"


def test_clean_prefetch_query_empty_when_only_boilerplate():
    raw = "<client_context>\ntime: now\ntrigger: screen app_switched\n</client_context>"
    assert clean_prefetch_query(raw) == ""


def test_clean_prefetch_query_handles_multiple_blocks():
    raw = (
        "<client_context>\ntime: t1\nbody: first payload\n</client_context>\n"
        "hello\n"
        "<client_context>\ntrigger: x\ncue note: second payload\n</client_context>"
    )
    cleaned = clean_prefetch_query(raw)
    assert "body: first payload" in cleaned
    assert "cue note: second payload" in cleaned
    assert "time: t1" not in cleaned


def test_build_prefetch_searches_with_cleaned_query():
    import json as jsonlib

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = jsonlib.loads(request.content)
        return httpx.Response(200, json=[])

    client = _client(handler)
    client.build_prefetch(RAW_USER_TURN, set())
    assert captured["body"]["query"] == "それ、多分できるよ。"


def test_build_prefetch_skips_search_when_query_cleans_to_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no search request expected")

    client = _client(handler)
    raw = "<client_context>\ntime: now\ntrigger: screen\n</client_context>"
    assert client.build_prefetch(raw, set()) == ""
