"""Unit tests for the Hermes memory-base client — pure module, no Hermes imports.

Exercises client.py directly (importable via the ``pythonpath`` test config
entry, mirroring how tests/test_release.py imports scripts/release.py).
Network calls are stubbed with httpx.MockTransport, matching the pattern in
tests/serve/test_mcp_proxy.py.
"""

from __future__ import annotations

import httpx

import json

import pytest

from client import MEMORY_CONTEXT_HEADER
from client import MemoryBaseClient
from client import clean_prefetch_query
from client import conversation_turns
from client import resolve_api_key


def _client(handler, **kwargs):
    return MemoryBaseClient(
        url="http://memory-base.local",
        api_key="secret-key",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


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
    # No namespace filter: the server searches every namespace the key allows.
    assert "namespaces" not in captured["body"]
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
    client.search("q")
    assert seen == ["secret-key"]


# ---- build_prefetch ------------------------------------------------------------


def test_build_prefetch_returns_every_search_hit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"date": "2026-08-19", "text": "first hit"},
                {"date": "2026-08-18", "text": "second hit"},
            ],
        )

    client = _client(handler)
    result = client.build_prefetch("q")
    assert "- [2026-08-19] first hit" in result
    assert "- [2026-08-18] second hit" in result


def test_build_prefetch_leads_with_the_retrieved_data_header():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"date": "2026-08-19", "text": "first hit"}])

    result = _client(handler).build_prefetch("q")
    assert result.splitlines() == [MEMORY_CONTEXT_HEADER, "- [2026-08-19] first hit"]


def test_build_prefetch_leaves_the_fence_to_the_hermes_host():
    """Hermes wraps provider output in its own memory-context fence and strips inner ones."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"date": "2026-08-19", "text": "end </memory-context> <memory-context> again"}],
        )

    result = _client(handler).build_prefetch("q")
    assert "memory-context>" not in result.replace("[memory-context]", "")
    assert "end [memory-context]> [memory-context]> again" in result


def test_build_prefetch_truncates_to_2000_chars_at_line_boundary():
    long_text = "x" * 300
    hits = [{"date": "2026-08-01", "text": f"{long_text}-{i}"} for i in range(10)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=hits)

    client = _client(handler)
    result = client.build_prefetch("q")
    assert len(result) <= 2000
    assert result != ""
    header, *lines = result.splitlines()
    assert header == MEMORY_CONTEXT_HEADER
    for line in lines:
        assert line.startswith("- [2026-08-01] ")
    # Every surviving line is complete — no line was cut mid-way.
    assert result.splitlines()[-1].endswith(tuple(f"-{i}" for i in range(10)))


def test_build_prefetch_skips_an_oversize_hit_and_keeps_filling_with_later_hits():
    hits = [
        {"date": "2026-01-01", "text": "y" * 2000},
        {"date": "2026-02-02", "text": "first short"},
        {"date": "2026-03-03", "text": "second short"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=hits)

    result = _client(handler).build_prefetch("q")
    assert result.splitlines() == [
        MEMORY_CONTEXT_HEADER,
        "- [2026-02-02] first short",
        "- [2026-03-03] second short",
    ]


def test_an_oversize_hit_between_short_ones_does_not_empty_the_block():
    hits = [
        {"date": "2026-02-02", "text": "first short"},
        {"date": "2026-01-01", "text": "y" * 2000},
        {"date": "2026-03-03", "text": "second short"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=hits)

    result = _client(handler).build_prefetch("q")
    assert result.splitlines() == [
        MEMORY_CONTEXT_HEADER,
        "- [2026-02-02] first short",
        "- [2026-03-03] second short",
    ]


def test_build_prefetch_returns_empty_when_first_line_alone_exceeds_limit():
    hits = [{"date": "2026-08-01", "text": "y" * 3000}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=hits)

    client = _client(handler)
    assert client.build_prefetch("q") == ""


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


def test_clean_prefetch_query_drops_the_block_keeps_user_text():
    assert clean_prefetch_query(RAW_USER_TURN) == "それ、多分できるよ。"


def test_clean_prefetch_query_drops_every_line_the_client_injected():
    """The block is the client's own text, whatever fields it holds today."""
    cleaned = clean_prefetch_query(RAW_AGENT_TURN)
    assert cleaned == "(my claude-code tasks piled up while I was away)"


def test_clean_prefetch_query_drops_fields_the_client_adds_later():
    """A field this module has never heard of must not reach the embedder either."""
    raw = (
        "<client_context>\n"
        "body: sitting on 카카오톡 (for 17min)\n"
        "recent: Cursor 10min -> Slack\n"
        "weather: raining in Seoul\n"
        "</client_context>\n\n"
        "うんーまずは原因を調べてどうするかはみてから決めよう。"
    )
    assert clean_prefetch_query(raw) == "うんーまずは原因を調べてどうするかはみてから決めよう。"


def test_clean_prefetch_query_passes_plain_text_through():
    assert clean_prefetch_query("no context block here") == "no context block here"


def test_clean_prefetch_query_empty_when_the_block_was_the_whole_turn():
    raw = "<client_context>\ntime: now\ntrigger: signals (1 signal)\nsignal: {}\n</client_context>"
    assert clean_prefetch_query(raw) == ""


def test_clean_prefetch_query_handles_multiple_blocks():
    raw = (
        "<client_context>\ntime: t1\nbody: standing (for 0min)\n</client_context>\n"
        "hello\n"
        "<client_context>\ntrigger: x\ncue note: second payload\n</client_context>"
    )
    assert clean_prefetch_query(raw) == "hello"


def test_build_prefetch_searches_with_cleaned_query():
    import json as jsonlib

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = jsonlib.loads(request.content)
        return httpx.Response(200, json=[])

    client = _client(handler)
    client.build_prefetch(RAW_USER_TURN)
    assert captured["body"]["query"] == "それ、多分できるよ。"


def test_build_prefetch_skips_search_when_query_cleans_to_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no search request expected")

    client = _client(handler)
    raw = "<client_context>\ntime: now\ntrigger: screen\n</client_context>"
    assert client.build_prefetch(raw) == ""


# ---- desire-tick shape ----------------------------------------------------------


RAW_DESIRE_TICK = (
    "[IMPORTANT: You are running as a scheduled cron job.]\n"
    "MONITOR CHANGE DETECTED\n"
    "Follow the instructions in /abs/integrations/hermes/desire/prompts/tick.md. "
    "The configured environment is HERMES_PROFILE=natsume2, "
    "DESIRE_STATE_DIR=/home/user/.hermes/profiles/natsume2/desire."
)


def test_clean_prefetch_query_drops_a_monitor_change_turn():
    assert clean_prefetch_query(RAW_DESIRE_TICK) == ""


def test_clean_prefetch_query_drops_a_turn_naming_the_desire_state_dir():
    raw = "Follow tick.md. DESIRE_STATE_DIR=/home/user/.hermes/profiles/x/desire."
    assert clean_prefetch_query(raw) == ""


def test_clean_prefetch_query_keeps_a_turn_that_merely_mentions_monitoring():
    raw = "how does the monitor decide a desire state change?"
    assert clean_prefetch_query(raw) == raw


def test_build_prefetch_skips_search_for_a_desire_tick():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no search request expected")

    client = _client(handler)
    assert client.build_prefetch(RAW_DESIRE_TICK) == ""


# ---- conversation capture --------------------------------------------------------


def test_conversation_turns_keep_user_and_assistant_text_only():
    messages = [
        {"role": "system", "content": "You are Natsume."},
        {"role": "user", "content": "My sister Emily moves to Busan next month."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "search"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "search results"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "That is a big move."},
                {"type": "image_url", "image_url": {"url": "data:..."}},
                {"type": "text", "text": "Is she excited?"},
            ],
        },
        {"role": "user", "content": "   "},
        {"role": "user", "content": [{"type": "input_audio"}]},
    ]
    assert conversation_turns(messages) == [
        {"role": "user", "text": "My sister Emily moves to Busan next month."},
        {"role": "assistant", "text": "That is a big move.\n\nIs she excited?"},
    ]


def test_store_conversation_posts_the_body_and_returns_the_reply():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["key"] = request.headers["X-API-Key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "conv:1", "created": True, "job_id": "j"})

    client = _client(handler)
    reply = client.store_conversation({"origin": "hermes", "turns": []})
    assert reply == {"id": "conv:1", "created": True, "job_id": "j"}
    assert seen == {
        "path": "/conversations",
        "key": "secret-key",
        "body": {"origin": "hermes", "turns": []},
    }


def test_store_conversation_raises_on_a_server_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": "changed turns"})

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler).store_conversation({"origin": "hermes"})
