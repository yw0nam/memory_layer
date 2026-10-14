"""The entity tools' consent step on a client with elicitation and on one without.

No DB/network: rest_client.client is mocked via httpx.MockTransport; the client is the
in-process MCP session, with or without an elicitation callback.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ElicitResult

from memory_base.serve import mcp_server
from memory_base.serve.common import rest_client

REGISTRY = [
    {
        "slug": "natsume", "kind": "person", "name": "Natsume", "aliases": ["Nat"],
        "description": "A friend", "updated_at": "2023-11-14T22:13:20+00:00",
    }
]  # fmt: skip
PLAN = {"note_ids": ["note:a", "note:b"], "facts": 3, "incoming": 1}
EXPECT = {"notes": 2, "facts": 3, "incoming": 1}
NEW = {"slug": "yui", "kind": "person", "name": "Yui", "aliases": ["ゆい"]}


@pytest.fixture()
def backend(monkeypatch):
    """The fake REST backend: `posts` are the real writes, `dry_runs` the validations."""
    state = SimpleNamespace(posts=[], dry_runs=[], refusal=None)

    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"entities": REGISTRY})
        body = json.loads(request.content)
        if body.get("dry_run"):
            state.dry_runs.append(body)
            if state.refusal is not None:
                return httpx.Response(409, json={"error": state.refusal})
        else:
            state.posts.append((request.url.path, body))
        if request.url.path.endswith("/delete"):
            return httpx.Response(200, json=PLAN)
        return httpx.Response(200, json={"entities": [], "entity": {}})

    monkeypatch.setattr(
        rest_client,
        "client",
        lambda: httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        ),
    )
    return state


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
    assert backend.posts == [
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
    assert backend.posts == []


def test_confirmed_on_an_eliciting_client_is_refused(backend):
    result, messages = _call(
        {"entities": [NEW], "author": "claude-code", "confirmed": True}, _accept()
    )
    assert result.isError
    assert "without confirmed" in _text(result)
    assert backend.posts == [] and messages == []


def test_a_client_without_elicitation_gets_needs_consent_then_confirms(backend):
    arguments = {"entities": [NEW], "author": "claude-code"}
    result, _ = _call(arguments)
    payload = json.loads(_text(result))
    assert payload["status"] == "needs_consent"
    assert "create yui" in payload["summary"]
    assert backend.posts == []
    _call({**arguments, "confirmed": True})
    assert backend.posts[0][1]["consent_method"] == "conversation"


def test_the_summary_of_an_edit_shows_old_and_new(backend):
    edit = {"slug": "natsume", "name": "Natsume K", "aliases": ["Nat", "Natchan"]}
    result, _ = _call({"entities": [edit], "author": "claude-code"})
    summary = json.loads(_text(result))["summary"]
    assert "edit natsume: name: Natsume -> Natsume K" in summary
    assert "aliases: Nat -> Nat, Natchan" in summary


def test_an_accepted_delete_dialog_shows_the_plan_and_posts_it_back_as_expect(backend):
    result, messages = _call(
        {"entities": [{"slug": "natsume"}], "author": "claude-code", "delete": True}, _accept()
    )
    assert not result.isError
    assert backend.posts == [
        (
            "/entities/natsume/delete",
            {"author": "claude-code", "consent_method": "elicitation", "expect": EXPECT},
        )
    ]
    assert "delete natsume (Natsume)" in messages[0]
    assert "note:a" in messages[0] and "3 facts" in messages[0] and "1 fact" in messages[0]


def test_a_delete_without_elicitation_needs_consent_then_confirms_with_expect(backend):
    arguments = {"entities": [{"slug": "natsume"}], "author": "claude-code", "delete": True}
    result, _ = _call(arguments)
    payload = json.loads(_text(result))
    assert payload["status"] == "needs_consent"
    assert "note:b" in payload["summary"]
    assert backend.posts == []
    bare, _ = _call({**arguments, "confirmed": True})
    assert bare.isError and backend.posts == []
    _call({**arguments, "confirmed": True, "expect": EXPECT})
    assert backend.posts == [
        (
            "/entities/natsume/delete",
            {"author": "claude-code", "consent_method": "conversation", "expect": EXPECT},
        )
    ]


@pytest.mark.parametrize(
    "entities",
    [[{"slug": "natsume"}, {"slug": "yui"}], [{"slug": "natsume", "name": "N"}]],
    ids=["two-items", "edit-fields"],
)
def test_delete_takes_exactly_one_slug(backend, entities):
    result, _ = _call({"entities": entities, "author": "claude-code", "delete": True}, _accept())
    assert result.isError
    assert backend.posts == []


def test_a_refused_create_never_reaches_the_user(backend):
    backend.refusal = "'yui' is already held by entity 'yui-1'; reuse it"
    result, messages = _call({"entities": [NEW], "author": "claude-code"}, _accept())
    assert result.isError
    assert "already held" in _text(result)
    assert messages == [] and backend.posts == []


def test_an_edit_that_changes_nothing_asks_no_consent(backend):
    same = {"slug": "natsume", "name": "Natsume"}
    result, messages = _call({"entities": [same], "author": "claude-code"}, _accept())
    assert json.loads(_text(result)) == {"status": "unchanged"}
    assert messages == [] and backend.posts == [] and backend.dry_runs == []


def test_a_description_edit_carries_the_updated_at_of_the_registry_it_read(backend):
    edit = {"slug": "natsume", "description": "A close friend"}
    _call({"entities": [edit], "author": "claude-code"}, _accept())
    stamped = {**edit, "base_updated_at": "2023-11-14T22:13:20+00:00"}
    assert backend.dry_runs[0]["entities"] == [stamped]
    assert backend.posts[0][1]["entities"] == [stamped]


def test_read_entity_composes_the_card_and_the_notes_naming_the_entity(monkeypatch):
    calls = []

    def handler(request):
        calls.append((request.url.path, sorted(request.url.params.multi_items())))
        if request.url.path == "/notes":
            return httpx.Response(200, json=[{"id": "note:a"}])
        return httpx.Response(
            200, json={"entity": {"slug": "natsume"}, "facts": [], "incoming": []}
        )

    monkeypatch.setattr(
        rest_client,
        "client",
        lambda: httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        ),
    )

    async def run():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as c:
            args = {"slug": "natsume", "include_past": True, "namespace": "team-a"}
            return await c.call_tool("read_entity", args)

    result = asyncio.run(run())
    assert json.loads(_text(result)) == {
        "entity": {"slug": "natsume"}, "facts": [], "incoming": [], "notes": [{"id": "note:a"}]
    }  # fmt: skip
    assert (
        "/entities/natsume",
        [("include_past", "true"), ("namespace", "team-a")],
    ) in calls
    assert (
        "/notes",
        [("entity", "natsume"), ("limit", "20"), ("namespace", "team-a")],
    ) in calls
