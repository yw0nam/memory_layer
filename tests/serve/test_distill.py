"""The conversation distill job against a fake source row, a fake provider, and a fake save.

No DB and no network: ``distill`` sees a fake connection holding one
conversation_sources row, ``chat_json`` is a scripted fake, and ``save_note``
records its calls instead of embedding or storing anything.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest

from memory_base.serve import distill, notes
from memory_base.serve.notes import CredentialNoteError, LowSignalNoteError, SimilarNotesError

CID = "conv:0123456789abcdef"
STARTED = 1_790_000_000.0  # 2026-09-21 UTC


def _turns(count: int, text: str = "turn text") -> list[dict[str, str]]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "text": f"{text} {i}"} for i in range(count)
    ]


class FakeConn:
    """One conversation_sources row; the cursor moves only on a matching compare-and-swap."""

    def __init__(self, turns, *, distilled_through=0, origin="claude_code", metadata=None):
        self.source = {
            "namespace": "dev",
            "origin": origin,
            "started_at": STARTED,
            "turns": turns,
            "distilled_through": distilled_through,
            "metadata": {"repo": "memory_base", "cwd": "/repo"} if metadata is None else metadata,
        }
        self.cursor_updates: list[tuple[int, int]] = []
        self.moved_by_another_job = False

    async def fetchrow(self, query, *args):
        assert "conversation_sources" in query
        assert args == (CID,)
        return {
            **self.source,
            "turns": json.dumps(self.source["turns"]),
            "metadata": json.dumps(self.source["metadata"]),
        }

    async def execute(self, query, *args):
        assert "SET distilled_through" in query
        conversation_id, read, new = args
        assert conversation_id == CID
        if self.moved_by_another_job:
            self.source["distilled_through"] += 1
        if self.source["distilled_through"] != read:
            return "UPDATE 0"
        self.source["distilled_through"] = new
        self.cursor_updates.append((read, new))
        return "UPDATE 1"


class FakeProvider:
    """Scripted chat_json: each call pops the next reply; an exception reply is raised."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.inputs: list[str] = []
        self.systems: list[str] = []

    async def __call__(self, messages, schema, *, timeout):
        self.systems.append(messages[0]["content"])
        self.inputs.append(messages[-1]["content"])
        reply = self.replies.pop(0) if self.replies else {"notes": []}
        if isinstance(reply, BaseException):
            raise reply
        return reply


class FakeSave:
    """Records every save_note call; an outcome per content decides stored or raised."""

    def __init__(self, outcomes=None):
        self.calls: list[dict] = []
        self.outcomes = outcomes or {}

    async def __call__(self, content, **kwargs):
        self.calls.append({"content": content, **kwargs})
        outcome = self.outcomes.get(content)
        if isinstance(outcome, BaseException):
            raise outcome
        return {"id": f"note:dev:{len(self.calls)}", "stored": outcome != "duplicate"}


async def _noop(conn):
    return None


@pytest.fixture
def world(monkeypatch):
    def _world(conn, provider, save=None):
        @asynccontextmanager
        async def acquire(timeout=None):
            yield conn

        save = save or FakeSave()
        monkeypatch.setattr(distill.db, "acquire", acquire)
        monkeypatch.setattr(distill, "ensure_schema_once", _noop)
        monkeypatch.setattr(distill, "chat_json", provider)
        monkeypatch.setattr(notes, "save_note", save)
        return save

    return _world


def _job():
    return distill.ConversationJob(
        job_id="job-1", conversation_id=CID, namespace="dev", key_id="k", key_label="test"
    )


def _run(job=None):
    job = job or _job()
    asyncio.run(distill.run_conversation_job(job))
    return job


def _unit(content, start, end, *, kind="note", tags=("staging",), **extra):
    return {
        "content": content,
        "kind": kind,
        "turn_start": start,
        "turn_end": end,
        "tags": list(tags),
        **extra,
    }


# ---- rendering ------------------------------------------------------------------


def test_new_turns_are_rendered_with_their_absolute_indices_and_the_session_date(world):
    turns = _turns(5)
    provider = FakeProvider()
    world(FakeConn(turns, distilled_through=2), provider)
    _run()
    (rendered,) = provider.inputs
    assert "[2] user: turn text 2" in rendered
    assert "[3] assistant: turn text 3" in rendered
    assert "[4] user: turn text 4" in rendered
    assert "[0]" not in rendered and "[1]" not in rendered
    assert "turn text 1" not in rendered
    assert "Session date: 2026-09-21" in rendered


def test_the_origin_chooses_the_extraction_prompt(world):
    coding = FakeProvider()
    world(FakeConn(_turns(2)), coding)
    _run()
    personal = FakeProvider()
    world(FakeConn(_turns(2), origin="hermes", metadata={}), personal)
    _run()
    assert "memory save policy" in coding.inputs[0]
    assert "Personal memory policy" in personal.inputs[0]
    assert coding.systems[0] == personal.systems[0] == distill.EXTRACTION_SYSTEM_PROMPT


def test_the_package_prompts_carry_the_turn_range_and_tags_contract():
    for name in ("digest", "personal"):
        prompt = distill.load_prompt(name)
        assert "turn_start" in prompt and "turn_end" in prompt and "tags" in prompt
        assert "{date}" in prompt and "{session}" in prompt
    assert "turn_start" in distill.EXTRACTION_SYSTEM_PROMPT


# ---- saving ---------------------------------------------------------------------


def test_each_unit_is_saved_linked_to_its_turns(world):
    provider = FakeProvider(
        {
            "notes": [
                _unit("Staging images are built multi-arch.", 0, 1, tags=["Staging", "arm"]),
                _unit(
                    "On 2026-09-20, the staging deploy dropped connections - draining fixed it.",
                    2,
                    3,
                    kind="episode",
                    date="2026-09-20",
                ),
            ]
        }
    )
    save = world(FakeConn(_turns(4)), provider)
    job = _run()
    first, second = save.calls
    assert first["conversation_id"] == CID
    assert (first["turn_start"], first["turn_end"]) == (0, 1)
    assert first["tags"] == ["Staging", "arm", "repo:memory_base"]
    assert first["source_ref"] == "distill"
    assert first["namespace"] == "dev"
    assert first["kind"] == "note"
    assert first["author"] == "claude_code"
    assert first["occurred_at"] is None
    assert (second["turn_start"], second["turn_end"]) == (2, 3)
    assert second["kind"] == "episode"
    assert second["occurred_at"] == "2026-09-20"
    assert job.status == "succeeded"
    assert job.result == {"units": 2, "stored": 2, "refused": 0, "similar": 0}


def test_the_cursor_ends_at_the_turn_count_processed(world):
    conn = FakeConn(_turns(4))
    world(conn, FakeProvider())
    _run()
    assert conn.source["distilled_through"] == 4
    assert conn.cursor_updates == [(0, 4)]


def test_a_source_without_a_repo_adds_no_repo_tag(world):
    provider = FakeProvider({"notes": [_unit("A fact.", 0, 1, tags=["prefs"])]})
    save = world(FakeConn(_turns(2), origin="hermes", metadata={}), provider)
    _run()
    assert save.calls[0]["tags"] == ["prefs"]
    assert save.calls[0]["author"] == "hermes"


def test_a_unit_without_tags_falls_back_to_the_origin(world):
    provider = FakeProvider({"notes": [{"content": "A fact.", "kind": "note"}]})
    save = world(FakeConn(_turns(2), origin="hermes", metadata={}), provider)
    _run()
    assert save.calls[0]["tags"] == ["hermes"]


@pytest.mark.parametrize(
    ("start", "end"),
    [(None, None), (0, 7), (3, 1), (-1, 0), ("0", "1"), (True, 1)],
)
def test_a_missing_or_out_of_batch_turn_range_links_the_whole_batch(world, start, end):
    unit = {"content": "A fact.", "kind": "note", "tags": ["x"]}
    if start is not None:
        unit.update(turn_start=start, turn_end=end)
    save = world(FakeConn(_turns(3), metadata={}), FakeProvider({"notes": [unit]}))
    _run()
    assert (save.calls[0]["turn_start"], save.calls[0]["turn_end"]) == (0, 2)


@pytest.mark.parametrize("date", ["2020-13-40", "yesterday", "2999-01-01", 20200101])
def test_an_unusable_date_is_dropped_not_refused(world, date):
    unit = _unit("On a day, something happened - it ended well.", 0, 1, kind="episode", date=date)
    save = world(FakeConn(_turns(2)), FakeProvider({"notes": [unit]}))
    job = _run()
    assert save.calls[0]["occurred_at"] is None
    assert job.result["stored"] == 1


def test_refused_units_are_counted_not_raised(world):
    provider = FakeProvider(
        {
            "notes": [
                _unit("kept", 0, 1),
                _unit("gated", 0, 1),
                _unit("secret", 0, 1),
                _unit("similar", 0, 1),
                _unit("invalid", 0, 1),
                _unit("again", 0, 1),
            ]
        }
    )
    save = FakeSave(
        {
            "gated": LowSignalNoteError("restates a PR"),
            "secret": CredentialNoteError("AWS Access Key"),
            "similar": SimilarNotesError([{"id": "note:dev:x", "score": 0.9, "text": "kept"}]),
            "invalid": ValueError("kind must be one of ..."),
            "again": "duplicate",
        }
    )
    world(FakeConn(_turns(2)), provider, save)
    job = _run()
    assert job.status == "succeeded"
    assert job.result == {"units": 6, "stored": 1, "refused": 3, "similar": 1}


# ---- no_op and idempotency ------------------------------------------------------


@pytest.mark.parametrize(("count", "cursor"), [(1, 0), (5, 4), (5, 5)])
def test_fewer_than_two_new_turns_is_a_no_op_without_a_provider_call(world, count, cursor):
    conn = FakeConn(_turns(count), distilled_through=cursor)
    provider = FakeProvider()
    world(conn, provider)
    job = _run()
    assert job.status == "no_op"
    assert provider.inputs == []
    assert conn.cursor_updates == []


def test_a_second_job_on_the_same_source_is_a_no_op(world):
    conn = FakeConn(_turns(4))
    provider = FakeProvider({"notes": [_unit("A fact.", 0, 1)]})
    save = world(conn, provider)
    assert _run().status == "succeeded"
    second = _run()
    assert second.status == "no_op"
    assert len(provider.inputs) == 1
    assert len(save.calls) == 1


def test_a_job_after_new_turns_renders_only_the_new_ones(world):
    conn = FakeConn(_turns(4))
    provider = FakeProvider()
    world(conn, provider)
    _run()
    conn.source["turns"] = _turns(6)
    _run()
    assert "[3]" not in provider.inputs[1]
    assert "[4] user:" in provider.inputs[1] and "[5] assistant:" in provider.inputs[1]
    assert conn.source["distilled_through"] == 6


def test_a_cursor_moved_by_another_job_fails_this_one_and_stores_nothing_further(world):
    turns = _turns(4, text="x" * 40_000)
    conn = FakeConn(turns)
    conn.moved_by_another_job = True
    provider = FakeProvider({"notes": [_unit("first", 0, 1)]}, {"notes": [_unit("second", 2, 3)]})
    save = world(conn, provider)
    with pytest.raises(distill.CursorMoved, match="another job"):
        _run()
    assert [call["content"] for call in save.calls] == ["first"]
    assert len(provider.inputs) == 1


def test_a_removed_source_fails_the_job(world, monkeypatch):
    conn = FakeConn(_turns(2))

    async def missing(query, *args):
        return None

    conn.fetchrow = missing
    world(conn, FakeProvider())
    with pytest.raises(RuntimeError, match=CID):
        _run()


# ---- provider output ------------------------------------------------------------


def test_unparseable_output_is_retried_once(world):
    provider = FakeProvider(
        json.JSONDecodeError("bad", "doc", 0), {"notes": [_unit("A fact.", 0, 1)]}
    )
    save = world(FakeConn(_turns(2)), provider)
    job = _run()
    assert len(provider.inputs) == 2
    assert job.result["stored"] == 1
    assert len(save.calls) == 1


def test_output_unparseable_twice_fails_the_job_and_keeps_the_cursor(world):
    conn = FakeConn(_turns(2))
    provider = FakeProvider({"facts": []}, {"notes": "none"})
    world(conn, provider)
    with pytest.raises(distill.ExtractionFailed):
        _run()
    assert len(provider.inputs) == 2
    assert conn.source["distilled_through"] == 0


def test_parse_units_keeps_well_typed_optional_fields_only():
    units = distill.parse_units(
        {
            "notes": [
                {
                    "content": "a",
                    "kind": "episode",
                    "turn_start": 1,
                    "turn_end": 2,
                    "tags": ["x"],
                    "date": "2026-01-02",
                },
                {"content": "b", "turn_start": "1", "tags": "x", "date": 3},
            ]
        }
    )
    assert units == [
        {
            "content": "a",
            "kind": "episode",
            "turn_start": 1,
            "turn_end": 2,
            "tags": ["x"],
            "date": "2026-01-02",
        },
        {"content": "b", "kind": "note"},
    ]


@pytest.mark.parametrize(
    "payload",
    [[], {"facts": []}, {"notes": {}}, {"notes": ["a"]}, {"notes": [{"content": 1}]}],
)
def test_parse_units_rejects_a_malformed_reply(payload):
    with pytest.raises(ValueError):
        distill.parse_units(payload)


def test_parse_extraction_reads_the_json_text():
    assert distill.parse_extraction('{"notes": [{"content": "a", "kind": "note"}]}') == [
        {"content": "a", "kind": "note"}
    ]
    with pytest.raises(ValueError):
        distill.parse_extraction("not json")


# ---- batching -------------------------------------------------------------------


def test_batches_never_split_a_turn_and_stay_under_the_limit():
    turns = [{"role": "user", "text": "x" * 25_000} for _ in range(5)]
    assert distill.batch_ranges(turns, 0) == [(0, 1), (2, 3), (4, 4)]
    assert distill.batch_ranges(turns, 3) == [(3, 4)]


def test_a_turn_over_the_limit_is_its_own_batch():
    turns = [
        {"role": "user", "text": "a" * 10},
        {"role": "assistant", "text": "b" * (distill.DISTILL_BATCH_CHARS + 5)},
        {"role": "user", "text": "c" * 10},
    ]
    assert distill.batch_ranges(turns, 0) == [(0, 0), (1, 1), (2, 2)]


def test_a_long_conversation_goes_to_the_provider_in_several_calls_advancing_per_batch(world):
    turns = _turns(6, text="y" * 25_000)
    conn = FakeConn(turns)
    provider = FakeProvider(
        {"notes": [_unit("one", 0, 1)]},
        {"notes": [_unit("two", 2, 3)]},
        {"notes": [_unit("three", 4, 5)]},
    )
    save = world(conn, provider)
    job = _run()
    assert len(provider.inputs) == 3
    assert "[0] user:" in provider.inputs[0] and "[2]" not in provider.inputs[0]
    assert "[2] user:" in provider.inputs[1] and "[4]" not in provider.inputs[1]
    assert conn.cursor_updates == [(0, 2), (2, 4), (4, 6)]
    assert [(c["turn_start"], c["turn_end"]) for c in save.calls] == [(0, 1), (2, 3), (4, 5)]
    assert job.result == {"units": 3, "stored": 3, "refused": 0, "similar": 0}


def test_a_provider_failure_on_the_second_batch_leaves_the_cursor_after_the_first(world):
    turns = _turns(4, text="z" * 25_000)
    conn = FakeConn(turns)
    provider = FakeProvider({"notes": [_unit("one", 0, 1)]}, TimeoutError("provider down"))
    save = world(conn, provider)
    with pytest.raises(TimeoutError):
        _run()
    assert conn.source["distilled_through"] == 2
    assert [call["content"] for call in save.calls] == ["one"]


def test_a_transient_save_failure_keeps_the_cursor_before_its_batch(world):
    conn = FakeConn(_turns(2))
    save = FakeSave({"one": ConnectionError("database went away")})
    world(conn, FakeProvider({"notes": [_unit("one", 0, 1)]}), save)
    with pytest.raises(ConnectionError):
        _run()
    assert conn.source["distilled_through"] == 0


def test_an_oversized_turn_is_truncated_for_the_model_input_only(world):
    huge = "w" * (distill.DISTILL_BATCH_CHARS + 1000)
    turns = [{"role": "user", "text": huge}, {"role": "assistant", "text": "short"}]
    conn = FakeConn(turns)
    provider = FakeProvider()
    world(conn, provider)
    _run()
    assert "w" * distill.DISTILL_BATCH_CHARS in provider.inputs[0]
    assert "w" * (distill.DISTILL_BATCH_CHARS + 1) not in provider.inputs[0]
    assert conn.source["turns"][0]["text"] == huge
    assert conn.source["distilled_through"] == 2


# ---- the job row ----------------------------------------------------------------


def test_a_conversation_job_row_loads_its_result():
    from memory_base.serve import job_store

    row = {
        "job_id": "job-1",
        "kind": "conversation",
        "status": "succeeded",
        "error": None,
        "created_at": 0.0,
        "updated_at": 0.0,
        "conversation_id": CID,
        "namespace": "dev",
        "key_id": "k",
        "key_label": "test",
        "result": json.dumps({"units": 1, "stored": 1, "refused": 0, "similar": 0}),
    }
    job = job_store._row_to_job(row)
    assert isinstance(job, distill.ConversationJob)
    assert job.kind == "conversation"
    assert job.result == {"units": 1, "stored": 1, "refused": 0, "similar": 0}
    response = job.response()
    assert "key_id" not in response and "key_label" not in response
    assert response["conversation_id"] == CID


def test_the_conversation_kind_is_claimable():
    from memory_base.serve import job_store

    assert "kind = 'conversation'" in job_store.CONVERSATION_CLAIM_SQL
    assert "active.conversation_id = queued.conversation_id" in job_store.CONVERSATION_CLAIM_SQL
