"""The entity tools' consent step on a client with elicitation and on one without.

No DB/network: rest_client.client is mocked via httpx.MockTransport; the client is the
in-process MCP session, with or without an elicitation callback.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ElicitResult

from memory_base.serve import mcp_server
from memory_base.serve.common import rest_client

REGISTRY = [
    {
        "slug": "natsume", "kind": "person", "name": "Natsume", "aliases": ["Nat"],
        "description": "A friend", "archived_at": None,
    }
]  # fmt: skip
NEW = {"slug": "yui", "kind": "person", "name": "Yui", "aliases": ["ゆい"]}


@pytest.fixture()
def backend(monkeypatch):
    posts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"entities": REGISTRY})
        posts.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"entities": [], "entity": {}})

    monkeypatch.setattr(
        rest_client,
        "client",
        lambda: httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        ),
    )
    return posts


def _call(arguments, answer=None):
    """Run register_entity; `answer` is the elicitation result, None for a client without."""
    messages = []

    async def elicit(context, params):
        messages.append(params.message)
        return answer

    async def run():
        callback = elicit if answer is not None else None
        async with create_connected_server_and_client_session(
            mcp_server.mcp._mcp_server, elicitation_callback=callback
        ) as client:
            return await client.call_tool("register_entity", arguments)

    return asyncio.run(run()), messages


def _text(result):
    return result.content[0].text


def _accept(approve=True):
    return ElicitResult(action="accept", content={"approve": approve})


def test_an_accepted_dialog_writes_with_the_elicitation_method(backend):
    result, messages = _call({"entities": [NEW], "author": "claude-code"}, _accept())
    assert not result.isError
    assert backend == [
        (
            "/entities",
            {"entities": [NEW], "author": "claude-code", "consent_method": "elicitation"},
        )
    ]
    assert "create yui (person): Yui; aliases: ゆい" in messages[0]


@pytest.mark.parametrize(
    "answer",
    [_accept(False), ElicitResult(action="decline"), ElicitResult(action="cancel")],
    ids=["approve-false", "decline", "cancel"],
)
def test_a_refused_dialog_writes_nothing(backend, answer):
    result, _ = _call({"entities": [NEW], "author": "claude-code"}, answer)
    assert result.isError
    assert "consent" in _text(result)
    assert backend == []


def test_confirmed_on_an_eliciting_client_is_refused(backend):
    result, messages = _call(
        {"entities": [NEW], "author": "claude-code", "confirmed": True}, _accept()
    )
    assert result.isError
    assert "without confirmed" in _text(result)
    assert backend == [] and messages == []


def test_a_client_without_elicitation_gets_needs_consent_then_confirms(backend):
    arguments = {"entities": [NEW], "author": "claude-code"}
    result, _ = _call(arguments)
    payload = json.loads(_text(result))
    assert payload["status"] == "needs_consent"
    assert "create yui" in payload["summary"]
    assert backend == []
    _call({**arguments, "confirmed": True})
    assert backend[0][1]["consent_method"] == "conversation"


def test_the_summary_of_an_edit_shows_old_and_new(backend):
    edit = {"slug": "natsume", "name": "Natsume K", "aliases": ["Nat", "Natchan"]}
    result, _ = _call({"entities": [edit], "author": "claude-code"})
    summary = json.loads(_text(result))["summary"]
    assert "edit natsume: name: Natsume -> Natsume K" in summary
    assert "aliases: Nat -> Nat, Natchan" in summary


def test_archive_takes_one_slug_and_posts_to_the_archive_route(backend):
    result, _ = _call(
        {"entities": [{"slug": "natsume"}], "author": "claude-code", "archive": True}, _accept()
    )
    assert not result.isError
    assert backend == [
        ("/entities/natsume/archive", {"author": "claude-code", "consent_method": "elicitation"})
    ]


def test_archive_with_two_items_is_refused(backend):
    items = [{"slug": "natsume"}, {"slug": "yui"}]
    result, _ = _call({"entities": items, "author": "claude-code", "archive": True}, _accept())
    assert result.isError
    assert backend == []
