"""Conversation sources and the note-to-source link, against fake connections.

No DB and no network: ``conversations`` and ``notes`` see a fake connection that
answers the lookups they make; the MCP proxy sees an httpx mock transport. The
fixed ``test-key`` header resolves to an admin identity (tests/serve/conftest.py).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from starlette.testclient import TestClient

from memory_base.core import config
from memory_base.retrieval import search as search_module
from memory_base.retrieval.search import Hit
from memory_base.serve import api, auth, conversations, mcp_server, notes
from memory_base.serve.notes import build_note_row

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

NOW = 1_700_000_000.0
TURNS = [
    {"role": "user", "text": "Which reranker should the search use?"},
    {"role": "assistant", "text": "Qwen3 reranker, templated as vLLM documents it."},
    {"role": "user", "text": "Keep the floor at 0.25."},
]
BODY = {
    "namespace": "default",
    "origin": "claude_code",
    "external_session_id": "sess-1",
    "started_at": NOW,
    "ended_at": NOW + 60,
    "turns": TURNS,
}
SOURCE_ID = conversations.conversation_source_id("default", "claude_code", "sess-1")


class _Tx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        return None


class FakeConn:
    """One conversation_sources row at most, plus whether a note references it."""

    def __init__(self, source=None, *, referenced=False, registered=True):
        self.source = source
        self.referenced = referenced
        self.registered = registered
        self.statements: list[tuple[str, tuple]] = []

    def transaction(self):
        return _Tx()

    def _record(self, query, args):
        self.statements.append((" ".join(query.split()), args))

    def sql(self, needle):
        return [s for s in self.statements if needle in s[0]]

    async def fetchval(self, query, *args):
        self._record(query, args)
        if "namespaces" in query and "FOR SHARE" in query:
            return 1 if self.registered else None
        if "memory_chunks" in query and "conversation_id" in query:
            return self.referenced
        return True

    async def fetchrow(self, query, *args):
        self._record(query, args)
        if self.source is None:
            return None
        if "FOR UPDATE" in query:
            return {
                "created_by": self.source["created_by"],
                "turns": json.dumps(self.source["turns"]),
                "distilled_through": self.source.get("distilled_through", 0),
            }
        if "jsonb_array_length" in query:
            return {
                "namespace": self.source["namespace"],
                "turn_count": len(self.source["turns"]),
            }
        return {
            "id": self.source["id"],
            "namespace": self.source["namespace"],
            "origin": self.source["origin"],
            "external_session_id": self.source["external_session_id"],
            "started_at": self.source["started_at"],
            "ended_at": self.source["ended_at"],
            "turns": json.dumps(self.source["turns"]),
        }

    async def fetch(self, query, *args):
        self._record(query, args)
        return []

    async def execute(self, query, *args):
        self._record(query, args)
        if "INSERT INTO" in query and "conversation_sources" in query:
            return "INSERT 0 0" if self.source is not None else "INSERT 0 1"
        if "INSERT INTO" in query:
            return "INSERT 0 1"
        return "UPDATE 1"


def _source(**overrides):
    source = {
        "id": SOURCE_ID,
        "namespace": "default",
        "origin": "claude_code",
        "external_session_id": "sess-1",
        "started_at": NOW,
        "ended_at": NOW + 60,
        "turns": TURNS,
        "created_by": "test",
    }
    source.update(overrides)
    return source


async def _noop(conn):
    return None


class FakeAdmission:
    """Records each distill job admission and the connection it ran on."""

    def __init__(self):
        self.calls: list[dict] = []
        self.error: BaseException | None = None

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(job_id=f"job-{len(self.calls)}")


@pytest.fixture
def admission(monkeypatch):
    fake = FakeAdmission()
    monkeypatch.setattr(conversations.job_store, "admit_conversation", fake)
    return fake


@pytest.fixture
def use(monkeypatch, admission):
    def _use(conn):
        @asynccontextmanager
        async def acquire(timeout=None):
            yield conn

        monkeypatch.setattr(conversations.db, "acquire", acquire)
        monkeypatch.setattr(conversations, "ensure_schema_once", _noop)
        monkeypatch.setattr(notes.db, "acquire", acquire)
        monkeypatch.setattr(notes, "ensure_schema_once", _noop)
        return conn

    return _use


def _member_client(monkeypatch, label="member", allowed=frozenset({"default"})):
    identity = auth.KeyIdentity(
        key_id=f"{label}-hash",
        label=label,
        home="default",
        is_admin=False,
        allowed=frozenset(allowed),
        authors=frozenset({"claude-code"}),
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "member-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)
    return TestClient(api.app, headers={"X-API-Key": "member-key"})


# ---- source id ------------------------------------------------------------------


def test_source_id_hashes_namespace_origin_and_external_session_id():
    digest = hashlib.sha256(b"default\nclaude_code\nsess-1").hexdigest()[:16]
    assert SOURCE_ID == f"conv:{digest}"


def test_source_id_differs_per_namespace_and_origin():
    ids = {
        conversations.conversation_source_id("default", "claude_code", "sess-1"),
        conversations.conversation_source_id("team-a", "claude_code", "sess-1"),
        conversations.conversation_source_id("default", "hermes", "sess-1"),
    }
    assert len(ids) == 3


# ---- POST /conversations: validation -------------------------------------------


@pytest.mark.parametrize(
    "turns",
    [
        [],
        "user: hi",
        [{"role": "tool", "text": "ls output"}],
        [{"role": "system", "text": "be terse"}],
        [{"role": "user", "text": ""}],
        [{"role": "user", "text": "   "}],
        [{"role": "user", "text": 3}],
        [{"role": "user"}],
        [{"text": "hi"}],
        [{"role": "user", "text": "hi", "tool_output": "secret listing"}],
        ["hi"],
    ],
)
def test_post_conversation_rejects_malformed_turns(use, turns):
    conn = use(FakeConn())
    response = client.post("/conversations", json={**BODY, "turns": turns})
    assert response.status_code == 400
    assert "turn" in response.json()["error"]
    assert conn.sql("INSERT") == []


@pytest.mark.parametrize("field", ["origin", "external_session_id"])
@pytest.mark.parametrize("value", [None, "", "  ", 7])
def test_post_conversation_requires_origin_and_session_id(use, field, value):
    use(FakeConn())
    response = client.post("/conversations", json={**BODY, field: value})
    assert response.status_code == 400
    assert field in response.json()["error"]


@pytest.mark.parametrize("field", ["started_at", "ended_at"])
@pytest.mark.parametrize("value", [None, "2026-01-01", True])
def test_post_conversation_requires_epoch_bounds(use, field, value):
    use(FakeConn())
    response = client.post("/conversations", json={**BODY, field: value})
    assert response.status_code == 400
    assert field in response.json()["error"]


def test_post_conversation_rejects_an_end_before_the_start(use):
    use(FakeConn())
    response = client.post("/conversations", json={**BODY, "ended_at": NOW - 1})
    assert response.status_code == 400
    assert "ended_at" in response.json()["error"]


def test_post_conversation_rejects_unknown_fields(use):
    use(FakeConn())
    response = client.post("/conversations", json={**BODY, "tool_calls": []})
    assert response.status_code == 400
    assert "tool_calls" in response.json()["error"]


def test_post_conversation_over_the_text_cap_is_413(use):
    conn = use(FakeConn())
    half = conversations.CONVERSATION_MAX_CHARS // 2
    turns = [
        {"role": "user", "text": "u" * half},
        {"role": "assistant", "text": "a" * (half + 1)},
    ]
    response = client.post("/conversations", json={**BODY, "turns": turns})
    assert response.status_code == 413
    assert conn.sql("INSERT") == []


def test_post_conversation_at_the_text_cap_is_accepted(use):
    use(FakeConn())
    turns = [{"role": "user", "text": "u" * conversations.CONVERSATION_MAX_CHARS}]
    response = client.post("/conversations", json={**BODY, "turns": turns})
    assert response.status_code == 201


# Uppercase on purpose: the AWS detector matches the canonical key shape only.
AWS_KEY = "AKIA" + "Q" * 16


def test_post_conversation_with_a_credential_in_a_turn_is_refused_whole(use):
    conn = use(FakeConn())
    turns = [*TURNS, {"role": "user", "text": f"the deploy key is {AWS_KEY}, use it"}]
    response = client.post("/conversations", json={**BODY, "turns": turns})
    assert response.status_code == 400
    error = response.json()["error"]
    assert "turn 3" in error
    assert "AWS" in error
    assert AWS_KEY not in error
    assert conn.statements == []


# ---- POST /conversations: namespace access -------------------------------------


def test_post_conversation_outside_the_allowed_set_is_403(use, monkeypatch):
    conn = use(FakeConn())
    member = _member_client(monkeypatch, allowed={"default"})
    response = member.post("/conversations", json={**BODY, "namespace": "team-b"})
    assert response.status_code == 403
    assert conn.statements == []


def test_post_conversation_omitted_namespace_lands_in_the_key_home(use):
    conn = use(FakeConn())
    body = {k: v for k, v in BODY.items() if k != "namespace"}
    response = client.post("/conversations", json=body)
    assert response.status_code == 201
    insert = conn.sql("INSERT INTO")[0]
    assert insert[1][1] == "default"


def test_post_conversation_into_an_unregistered_namespace_is_400(use):
    conn = use(FakeConn(registered=False))
    response = client.post("/conversations", json=BODY)
    assert response.status_code == 400
    assert "unregistered namespace" in response.json()["error"]
    assert conn.sql("INSERT") == []


def test_post_conversation_by_another_key_over_an_existing_source_is_403(use, monkeypatch):
    conn = use(FakeConn(_source(created_by="someone-else")))
    member = _member_client(monkeypatch, label="member")
    changed = [*TURNS, {"role": "assistant", "text": "replaced evidence"}]
    response = member.post("/conversations", json={**BODY, "turns": changed})
    assert response.status_code == 403
    assert conn.sql("SET turns") == []


def test_post_conversation_by_its_creator_replaces_an_unreferenced_source(use, monkeypatch):
    conn = use(FakeConn(_source(created_by="member")))
    member = _member_client(monkeypatch, label="member")
    changed = [*TURNS, {"role": "assistant", "text": "one more turn"}]
    response = member.post("/conversations", json={**BODY, "turns": changed})
    assert response.status_code == 200
    assert response.json() == {"id": SOURCE_ID, "created": False, "turns": 4, "job_id": "job-1"}
    assert len(conn.sql("SET turns")) == 1


# ---- POST /conversations: upsert ----------------------------------------------


def test_post_conversation_creates_a_new_source(use):
    conn = use(FakeConn())
    response = client.post("/conversations", json=BODY)
    assert response.status_code == 201
    assert response.json() == {"id": SOURCE_ID, "created": True, "turns": 3, "job_id": "job-1"}
    (insert,) = conn.sql("INSERT INTO")
    assert "conversation_sources" in insert[0]
    assert "ON CONFLICT (namespace, origin, external_session_id)" in insert[0]
    assert json.loads(insert[1][6]) == TURNS
    assert json.loads(insert[1][7]) == {}
    assert insert[1][9] == "test"


def test_post_conversation_replaces_an_unreferenced_source(use):
    conn = use(FakeConn(_source()))
    changed = [*TURNS, {"role": "assistant", "text": "a fourth turn"}]
    response = client.post("/conversations", json={**BODY, "turns": changed, "ended_at": NOW + 120})
    assert response.status_code == 200
    assert response.json() == {"id": SOURCE_ID, "created": False, "turns": 4, "job_id": "job-1"}
    (update,) = conn.sql("SET turns")
    assert update[1][0] == SOURCE_ID
    assert json.loads(update[1][1]) == changed
    assert update[1][3] == NOW + 120


def test_post_conversation_refuses_changed_turns_once_a_note_references_it(use, admission):
    conn = use(FakeConn(_source(), referenced=True))
    changed = [TURNS[0], {"role": "assistant", "text": "rewritten evidence"}, TURNS[2]]
    response = client.post("/conversations", json={**BODY, "turns": changed})
    assert response.status_code == 409
    assert "referenced" in response.json()["error"]
    assert conn.sql("SET turns") == []
    assert admission.calls == []


def test_post_conversation_refuses_a_shorter_upload_of_a_referenced_source(use):
    conn = use(FakeConn(_source(), referenced=True))
    response = client.post("/conversations", json={**BODY, "turns": TURNS[:2]})
    assert response.status_code == 409
    assert conn.sql("SET turns") == []


def test_post_conversation_refuses_changed_turns_once_turns_are_distilled(use):
    conn = use(FakeConn(_source(distilled_through=2)))
    changed = [TURNS[0], {"role": "assistant", "text": "rewritten evidence"}, TURNS[2]]
    response = client.post("/conversations", json={**BODY, "turns": changed})
    assert response.status_code == 409
    assert "distilled" in response.json()["error"]
    assert conn.sql("SET turns") == []


def test_post_conversation_extends_a_referenced_source_by_appended_turns(use, admission):
    conn = use(FakeConn(_source(distilled_through=3), referenced=True))
    extended = [*TURNS, {"role": "assistant", "text": "a fourth turn"}]
    response = client.post(
        "/conversations", json={**BODY, "turns": extended, "ended_at": NOW + 120}
    )
    assert response.status_code == 200
    assert response.json() == {"id": SOURCE_ID, "created": False, "turns": 4, "job_id": "job-1"}
    (update,) = conn.sql("SET turns")
    assert json.loads(update[1][1]) == extended
    assert update[1][3] == NOW + 120
    assert len(admission.calls) == 1


def test_an_upload_never_writes_the_distill_cursor(use):
    conn = use(FakeConn(_source(distilled_through=3), referenced=True))
    extended = [*TURNS, {"role": "assistant", "text": "a fourth turn"}]
    assert client.post("/conversations", json={**BODY, "turns": extended}).status_code == 200
    use(FakeConn())
    assert client.post("/conversations", json=BODY).status_code == 201
    written = [s for s in conn.statements if s[0].startswith(("INSERT", "UPDATE"))]
    assert written
    assert all("distilled_through" not in statement for statement, _ in written)


def test_post_conversation_identical_turns_on_a_referenced_source_is_a_no_op(use):
    conn = use(FakeConn(_source(), referenced=True))
    response = client.post("/conversations", json=BODY)
    assert response.status_code == 200
    assert response.json() == {"id": SOURCE_ID, "created": False, "turns": 3, "job_id": "job-1"}
    assert conn.sql("SET turns") == []


# ---- POST /conversations: metadata and the distill job ---------------------------


def test_post_conversation_stores_the_client_metadata(use):
    conn = use(FakeConn())
    metadata = {"repo": "memory_base", "cwd": "/home/me/memory_base"}
    response = client.post("/conversations", json={**BODY, "metadata": metadata})
    assert response.status_code == 201
    (insert,) = conn.sql("INSERT INTO")
    assert json.loads(insert[1][7]) == metadata


@pytest.mark.parametrize("metadata", [[], "repo", 3, {"cwd": "x" * 2048}])
def test_post_conversation_rejects_bad_metadata(use, metadata):
    conn = use(FakeConn())
    response = client.post("/conversations", json={**BODY, "metadata": metadata})
    assert response.status_code == 400
    assert "metadata" in response.json()["error"]
    assert conn.statements == []


def test_post_conversation_admits_a_distill_job_on_the_upload_connection(use, admission):
    conn = use(FakeConn())
    response = client.post("/conversations", json=BODY)
    assert response.json()["job_id"] == "job-1"
    (call,) = admission.calls
    assert call["connection"] is conn
    assert call["conversation_id"] == SOURCE_ID
    assert call["namespace"] == "default"
    assert (call["key_id"], call["key_label"]) == ("test-key-hash", "test")


def test_a_failed_admission_fails_the_upload(use, admission):
    use(FakeConn())
    admission.error = RuntimeError("jobs table unavailable")
    failing = TestClient(api.app, headers={"X-API-Key": "test-key"}, raise_server_exceptions=False)
    response = failing.post("/conversations", json=BODY)
    assert response.status_code == 500


def test_an_origin_without_an_extraction_prompt_admits_no_job(use, admission):
    use(FakeConn())
    response = client.post("/conversations", json={**BODY, "origin": "codex"})
    assert response.status_code == 201
    assert response.json()["job_id"] is None
    assert admission.calls == []


def test_get_conversation_job_returns_its_state(monkeypatch):
    from memory_base.serve import distill, job_store

    job = distill.ConversationJob(
        job_id="job-9",
        conversation_id=SOURCE_ID,
        namespace="default",
        key_id="k",
        key_label="test",
        status="succeeded",
        result={"units": 2, "stored": 1, "refused": 1, "similar": 0},
    )

    async def get_job(job_id, *, kind):
        assert kind == "conversation"
        return job if job_id == "job-9" else None

    monkeypatch.setattr(job_store, "get_job", get_job)
    response = client.get("/conversations/jobs/job-9")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["result"] == {"units": 2, "stored": 1, "refused": 1, "similar": 0}
    assert body["conversation_id"] == SOURCE_ID
    assert "key_id" not in body
    assert client.get("/conversations/jobs/nope").status_code == 404
    member = _member_client(monkeypatch, allowed={"team-b"})
    assert member.get("/conversations/jobs/job-9").status_code == 404


def test_storing_a_source_never_calls_the_embedding_client(use, monkeypatch):
    calls = []

    async def recording_embed(self, text, query=False):
        calls.append(text)
        return [0.0]

    async def recording_embed_text(embedder, text):
        calls.append(text)
        return [0.0]

    monkeypatch.setattr(config.VllmEmbedder, "embed", recording_embed)
    monkeypatch.setattr(config, "embed_text", recording_embed_text)
    monkeypatch.setattr(notes, "embed_text", recording_embed_text)
    use(FakeConn())
    assert client.post("/conversations", json=BODY).status_code == 201
    use(FakeConn(_source()))
    changed = [*TURNS, {"role": "assistant", "text": "another turn"}]
    assert client.post("/conversations", json={**BODY, "turns": changed}).status_code == 200
    assert calls == []


# ---- GET /conversations/{id} ---------------------------------------------------


def test_get_conversation_returns_every_turn_with_its_index(use):
    use(FakeConn(_source()))
    response = client.get(f"/conversations/{SOURCE_ID}")
    assert response.status_code == 200
    assert response.json() == {
        "id": SOURCE_ID,
        "namespace": "default",
        "origin": "claude_code",
        "external_session_id": "sess-1",
        "started_at": NOW,
        "ended_at": NOW + 60,
        "turns": [{"index": i, **turn} for i, turn in enumerate(TURNS)],
    }


def test_get_conversation_slices_inclusively(use):
    use(FakeConn(_source()))
    response = client.get(f"/conversations/{SOURCE_ID}?turn_start=1&turn_end=2")
    assert response.status_code == 200
    assert response.json()["turns"] == [
        {"index": 1, **TURNS[1]},
        {"index": 2, **TURNS[2]},
    ]


def test_get_conversation_single_turn(use):
    use(FakeConn(_source()))
    response = client.get(f"/conversations/{SOURCE_ID}?turn_start=2&turn_end=2")
    assert response.json()["turns"] == [{"index": 2, **TURNS[2]}]


def test_get_conversation_open_ended_slice(use):
    use(FakeConn(_source()))
    response = client.get(f"/conversations/{SOURCE_ID}?turn_start=1")
    assert [turn["index"] for turn in response.json()["turns"]] == [1, 2]


@pytest.mark.parametrize(
    "query",
    ["turn_start=two", "turn_start=-1", "turn_start=2&turn_end=1", "turn_end=3"],
)
def test_get_conversation_bad_range_is_400(use, query):
    use(FakeConn(_source()))
    response = client.get(f"/conversations/{SOURCE_ID}?{query}")
    assert response.status_code == 400
    assert "turn" in response.json()["error"]


def test_get_conversation_missing_is_404(use):
    use(FakeConn())
    response = client.get(f"/conversations/{SOURCE_ID}")
    assert response.status_code == 404


def test_get_conversation_outside_the_readable_set_is_404(use, monkeypatch):
    use(FakeConn(_source(namespace="team-b")))
    member = _member_client(monkeypatch, allowed={"default"})
    response = member.get(f"/conversations/{SOURCE_ID}?turn_end=9")
    assert response.status_code == 404


# ---- build_note_row: conversation identity -------------------------------------


CID = SOURCE_ID


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def test_unlinked_note_id_hashes_an_empty_conversation_id():
    row = build_note_row("prefer ruff for linting", "note", ["test"], NOW)
    assert row["id"] == f"note:default:{_digest(chr(10) + 'prefer ruff for linting')}"
    assert row["session_id"] == row["id"]
    assert row["conversation_id"] is None


def test_linked_note_id_hashes_the_conversation_id_and_content():
    row = build_note_row(
        "prefer ruff for linting", "note", ["test"], NOW, "team-a", conversation_id=CID
    )
    assert row["id"] == f"note:team-a:{_digest(CID + chr(10) + 'prefer ruff for linting')}"
    assert row["conversation_id"] == CID


def test_linked_note_keeps_its_own_id_as_session_id():
    row = build_note_row("fact", "note", ["test"], NOW, conversation_id=CID)
    assert row["session_id"] == row["id"]


def test_identical_content_from_two_conversations_yields_two_ids():
    a = build_note_row("same fact", "note", ["test"], NOW, conversation_id="conv:aaaa")
    b = build_note_row("same fact", "note", ["test"], NOW, conversation_id="conv:bbbb")
    assert a["id"] != b["id"]


def test_turn_range_lands_on_the_row():
    row = build_note_row(
        "fact", "note", ["test"], NOW, conversation_id=CID, turn_start=0, turn_end=1
    )
    assert (row["turn_start"], row["turn_end"]) == (0, 1)


@pytest.mark.parametrize(
    ("conversation_id", "turn_start", "turn_end", "message"),
    [
        (None, 0, 1, "conversation_id"),
        (CID, 0, None, "together"),
        (CID, None, 1, "together"),
        (CID, -1, 1, "turn_start"),
        (CID, 2, 1, "turn_start"),
        (CID, "0", 1, "integer"),
        (CID, True, 1, "integer"),
        ("", None, None, "conversation_id"),
        (7, None, None, "conversation_id"),
    ],
)
def test_turn_range_validation(conversation_id, turn_start, turn_end, message):
    with pytest.raises(ValueError, match=message):
        build_note_row(
            "fact",
            "note",
            ["test"],
            NOW,
            conversation_id=conversation_id,
            turn_start=turn_start,
            turn_end=turn_end,
        )


def test_occurred_at_has_its_own_field_and_timestamp_stays_now():
    row = build_note_row("fact", "episode", ["test"], NOW, occurred_at=NOW - 86400)
    assert row["occurred_at"] == NOW - 86400
    assert row["timestamp"] == NOW


# ---- save_note: the linked source must exist in the note's namespace -----------


async def _gate(row):
    return None


async def _embed(embedder, text):
    return [0.0] * 4


@pytest.fixture
def saving(use, monkeypatch):
    monkeypatch.setattr(notes, "_content_gate", _gate)
    monkeypatch.setattr(notes, "embed_text", _embed)
    monkeypatch.setattr(notes, "VllmEmbedder", lambda: None)
    return use


def test_save_note_links_to_a_stored_source(saving):
    conn = saving(FakeConn(_source()))
    result = asyncio.run(
        notes.save_note("fact", tags=["test"], conversation_id=CID, turn_start=0, turn_end=1)
    )
    assert result["stored"] is True
    (lookup,) = conn.sql("jsonb_array_length")
    assert "FOR SHARE" in lookup[0]
    assert lookup[1] == (CID,)
    (insert,) = conn.sql("INSERT INTO")
    assert "conversation_id" in insert[0]
    assert CID in insert[1]


def test_save_note_to_a_missing_source_writes_nothing(saving):
    conn = saving(FakeConn())
    with pytest.raises(ValueError, match="unknown conversation_id"):
        asyncio.run(notes.save_note("fact", tags=["test"], conversation_id=CID))
    assert conn.sql("INSERT") == []


def test_save_note_to_a_source_in_another_namespace_writes_nothing(saving):
    conn = saving(FakeConn(_source(namespace="team-b")))
    with pytest.raises(ValueError, match="unknown conversation_id"):
        asyncio.run(notes.save_note("fact", tags=["test"], conversation_id=CID))
    assert conn.sql("INSERT") == []


def test_save_note_turn_end_past_the_source_is_refused(saving):
    conn = saving(FakeConn(_source()))
    with pytest.raises(ValueError, match="3 turns"):
        asyncio.run(
            notes.save_note("fact", tags=["test"], conversation_id=CID, turn_start=2, turn_end=3)
        )
    assert conn.sql("INSERT") == []


def test_save_note_occurred_at_is_stored_apart_from_the_save_time(saving, monkeypatch):
    conn = saving(FakeConn())
    monkeypatch.setattr(notes.time, "time", lambda: NOW)
    asyncio.run(notes.save_note("fact", tags=["test"], occurred_at="2020-01-01"))
    (insert,) = conn.sql("INSERT INTO")
    assert insert[1][8] == NOW
    assert 1577836800.0 in insert[1][9:]


# ---- POST /save_memory: link fields -------------------------------------------


def test_save_memory_forwards_the_link_fields(monkeypatch):
    captured = {}

    async def fake_save_note(content, **kwargs):
        captured.update(kwargs)
        return {"id": "note:x", "kind": "note", "stored": True, "superseded": None, "similar": []}

    monkeypatch.setattr(api, "save_note", fake_save_note)
    response = client.post(
        "/save_memory",
        json={
            "author": "natsume",
            "content": "fact",
            "tags": ["test"],
            "conversation_id": CID,
            "turn_start": 0,
            "turn_end": 1,
        },
    )
    assert response.status_code == 200
    assert (captured["conversation_id"], captured["turn_start"], captured["turn_end"]) == (
        CID,
        0,
        1,
    )


def test_save_memory_linked_to_a_missing_source_is_400(saving):
    conn = saving(FakeConn())
    response = client.post(
        "/save_memory",
        json={
            "author": "natsume",
            "content": "fact",
            "tags": ["test"],
            "conversation_id": CID,
            "turn_start": 0,
            "turn_end": 0,
        },
    )
    assert response.status_code == 400
    assert "unknown conversation_id" in response.json()["error"]
    assert conn.sql("INSERT") == []


# ---- search hits and note listings carry the link -------------------------------


def _memory_hit(**meta):
    base = {
        "id": "note:default:abcdabcdabcdabcd",
        "kind": "note",
        "tags": ["search"],
        "author": None,
        "namespace": "default",
        "session_id": "note:default:abcdabcdabcdabcd",
        "archived": False,
        "conversation_id": None,
        "turn_start": None,
        "turn_end": None,
        "occurred_at": None,
    }
    base.update(meta)
    return Hit(source="memory", ref="save_memory", text="fact", ts=NOW, rrf=0.5, meta=base)


def test_memory_hit_dict_carries_the_note_id_kind_and_tags():
    out = api.hit_to_dict(_memory_hit())
    assert out["id"] == "note:default:abcdabcdabcdabcd"
    assert out["kind"] == "note"
    assert out["tags"] == ["search"]
    assert "conversation_id" not in out
    assert "turn_start" not in out


def test_memory_hit_dict_carries_the_conversation_link():
    out = api.hit_to_dict(_memory_hit(conversation_id=CID, turn_start=2, turn_end=2))
    assert out["conversation_id"] == CID
    assert (out["turn_start"], out["turn_end"]) == (2, 2)


def test_memory_hit_dict_dates_by_occurred_at_when_set():
    out = api.hit_to_dict(_memory_hit(occurred_at=1577836800.0))
    assert out["date"] == "2020-01-01"
    assert api.hit_to_dict(_memory_hit())["date"] == "2023-11-14"


def test_code_hit_dict_is_unchanged():
    hit = Hit(
        source="code",
        ref="a.py:L1-L2",
        text="x",
        ts=NOW,
        meta={"id": "c1", "repo": "r", "filename": "a.py", "start_line": 1},
    )
    assert set(api.hit_to_dict(hit)) == {"source", "ref", "date", "score", "text", "repo"}


def _memory_row(cid, **overrides):
    row = {
        "id": cid,
        "source_ref": "save_memory",
        "chunk_kind": "note",
        "metadata": {"tags": ["search"]},
        "distilled": cid,
        "content_raw": cid,
        "ts_last_active": NOW,
        "archived_at": None,
        "namespace": "default",
        "session_id": cid,
        "conversation_id": None,
        "source_turn_start": None,
        "source_turn_end": None,
        "occurred_at": None,
    }
    row.update(overrides)
    return row


class FakeSearchConnection:
    def __init__(self, fetch_results):
        self.fetch_results = list(fetch_results)
        self.queries = []

    async def fetch(self, query, *args):
        self.queries.append(query)
        return self.fetch_results.pop(0) if self.fetch_results else []


def test_search_memory_reads_the_link_columns_into_the_hit():
    row = _memory_row(
        "note:default:1",
        conversation_id=CID,
        source_turn_start=0,
        source_turn_end=1,
        occurred_at=1577836800.0,
    )
    conn = FakeSearchConnection([[row], []])
    (hit,) = asyncio.run(search_module._search_memory(conn, "q", "[1]"))
    assert "conversation_id" in conn.queries[0]
    assert hit.meta["conversation_id"] == CID
    assert (hit.meta["turn_start"], hit.meta["turn_end"]) == (0, 1)
    assert hit.meta["occurred_at"] == 1577836800.0
    assert hit.ts == NOW


def test_four_linked_notes_from_one_conversation_all_reach_the_reranker(monkeypatch):
    rows = []
    for index in range(4):
        note = build_note_row(
            f"linked fact number {index}",
            "note",
            ["search"],
            NOW,
            conversation_id=CID,
            turn_start=index,
            turn_end=index,
        )
        rows.append(
            _memory_row(
                note["id"],
                session_id=note["session_id"],
                conversation_id=CID,
                source_turn_start=index,
                source_turn_end=index,
            )
        )
    conn = FakeSearchConnection([rows, []])

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    async def fake_embed_query(query):
        return "[1]"

    reranked: list[Hit] = []

    async def fake_rerank(query, hits):
        reranked.extend(hits)
        return hits

    monkeypatch.setattr(search_module.db, "acquire", acquire)
    monkeypatch.setattr(search_module, "_embed_query", fake_embed_query)
    monkeypatch.setattr(search_module, "_rerank", fake_rerank)
    asyncio.run(search_module.search("linked fact", source="memory"))
    assert len(reranked) == 4
    assert {h.meta["id"] for h in reranked} == {row["id"] for row in rows}


def test_list_notes_bounds_time_by_the_event_falling_back_to_the_save(monkeypatch):
    class ListConn:
        async def fetch(self, query, *args):
            self.query = query
            return []

    conn = ListConn()

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(notes.db, "acquire", acquire)
    asyncio.run(notes.list_notes(since="2020-01-01", until="2020-01-01"))
    assert "COALESCE(occurred_at, ts_last_active) >= $" in conn.query
    assert "COALESCE(occurred_at, ts_last_active) < $" in conn.query
    assert "ORDER BY ts_last_active DESC" in conn.query


def test_list_notes_carries_the_link_fields(monkeypatch):
    class ListConn:
        async def fetch(self, query, *args):
            self.query = query
            return [
                {
                    "id": "note:default:1",
                    "kind": "note",
                    "text": "fact",
                    "metadata": {"tags": ["t"]},
                    "ts_last_active": NOW,
                    "namespace": "default",
                    "archived_at": None,
                    "conversation_id": CID,
                    "source_turn_start": 0,
                    "source_turn_end": 1,
                    "occurred_at": 1577836800.0,
                }
            ]

    conn = ListConn()

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(notes.db, "acquire", acquire)
    (note,) = asyncio.run(notes.list_notes())
    assert "conversation_id" in conn.query
    assert note["conversation_id"] == CID
    assert (note["turn_start"], note["turn_end"]) == (0, 1)
    assert note["date"] == "2020-01-01"


# ---- MCP: save_memory link fields and expand_source ------------------------------


def _patch_mcp(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=mcp_server.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(mcp_server, "_client", fake_client)


def test_mcp_save_memory_posts_the_link_fields(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "note:x", "stored": True})

    _patch_mcp(monkeypatch, handler)
    asyncio.run(
        mcp_server.save_memory(
            "fact", "natsume", tags=["t"], conversation_id=CID, turn_start=0, turn_end=1
        )
    )
    assert captured["json"]["conversation_id"] == CID
    assert (captured["json"]["turn_start"], captured["json"]["turn_end"]) == (0, 1)


def test_mcp_save_memory_omits_unset_link_fields(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "note:x", "stored": True})

    _patch_mcp(monkeypatch, handler)
    asyncio.run(mcp_server.save_memory("fact", "natsume", tags=["t"]))
    assert not {"conversation_id", "turn_start", "turn_end"} & set(captured["json"])


def test_mcp_expand_source_gets_the_turn_range(monkeypatch):
    captured = {}
    payload = {"id": CID, "turns": [{"index": 2, **TURNS[2]}]}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=payload)

    _patch_mcp(monkeypatch, handler)
    result = asyncio.run(mcp_server.expand_source(CID, turn_start=2, turn_end=2))
    assert captured["method"] == "GET"
    assert captured["path"] == f"/conversations/{CID}"
    assert captured["params"] == {"turn_start": "2", "turn_end": "2"}
    assert result == payload


def test_mcp_expand_source_without_a_range_sends_no_params(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"id": CID, "turns": []})

    _patch_mcp(monkeypatch, handler)
    asyncio.run(mcp_server.expand_source(CID))
    assert captured["params"] == {}


def test_mcp_expand_source_surfaces_a_404(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": f"unknown conversation source: {CID}"})

    _patch_mcp(monkeypatch, handler)
    with pytest.raises(ValueError, match="unknown conversation source"):
        asyncio.run(mcp_server.expand_source(CID))


@pytest.mark.parametrize("bad", ["../admin/notes", "..", "conv:XYZ", "note:default:1", ""])
def test_mcp_expand_source_rejects_a_malformed_id_without_a_call(monkeypatch, bad):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a malformed id must not reach the REST backend")

    _patch_mcp(monkeypatch, handler)
    with pytest.raises(ValueError, match="conversation_id"):
        asyncio.run(mcp_server.expand_source(bad))
