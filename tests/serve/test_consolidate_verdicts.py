"""Unit tests for POST /admin/consolidate/verdicts: body schema, plan, token check, apply.

No DB, no network: the plan is a pure function over a loaded state, the apply flow runs
on fake connections with its readers and writer stubbed, and the write itself is checked
against a recording connection.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import asyncpg
import pytest
from starlette.testclient import TestClient

from memory_base.serve import admin, api, auth, namespaces, notes, verdicts
from memory_base.serve.consolidate import Note, Pair, group_key

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

NS = "work"
A, B, C, D = "note:work:a", "note:work:b", "note:work:c", "note:work:d"
TEXTS = {
    A: "Deploys run at 04:00 on weekdays.",
    B: "Weekday deploys start at 04:00.",
    C: "On weekdays the deploy job starts at 04:00.",
    D: "The weekday deploy begins at 04:00.",
}
MERGED = "Weekday deploys run at 04:00."


def note(note_id, text=None, **fields):
    values = {
        "id": note_id,
        "kind": "work",
        "author": "claude-code",
        "saved": 1_700_000_000.0,
        "occurred_at": None,
        "tags": ("deploy",),
        "similar_ack": (),
        "supersedes": None,
        "text": TEXTS[note_id] if text is None else text,
    }
    values.update(fields)
    return Note(**values)


NOTES = {i: note(i) for i in (A, B, C)}
PAIRS = [Pair(A, B, 0.80), Pair(A, C, 0.78), Pair(B, C, 0.75)]
KEY = group_key(NS, NOTES.values())


def item(action="keep", **fields):
    out = {
        "namespace": NS,
        "group_key": KEY,
        "idempotency_key": "k-1",
        "member_ids": [A, B, C],
        "action": action,
        "reason": "the three notes state one rule",
    }
    out.update(fields)
    return out


def body(*items, **top):
    out = {"run_id": "run-1", "author": "consolidator", "model": "test-model"}
    out.update(top)
    out["verdicts"] = list(items) or [item()]
    return out


def parsed(action="keep", top=None, **fields):
    batch = verdicts.parse_batch(body(item(action, **fields), **(top or {})), {NS})
    return batch, batch.verdicts[0]


def state(**over):
    values = {
        "recorded": None,
        "cached": None,
        "pairs": PAIRS,
        "notes": NOTES,
        "used_actions": 0,
        "replacement": None,
    }
    values.update(over)
    return verdicts.State(**values)


def plan(action="keep", state_over=None, top=None, **fields):
    batch, verdict = parsed(action, top, **fields)
    return verdicts.plan_verdict(verdict, batch, state(**(state_over or {})))


def replacement_id(text=MERGED):
    return notes.note_id(NS, text)


# ---- body schema ------------------------------------------------------------


def test_parse_batch_resolves_defaults():
    batch, verdict = parsed()
    assert batch.run_id == "run-1"
    assert batch.author == "consolidator"
    assert batch.model == "test-model"
    assert batch.dry_run is False
    assert batch.params == verdicts.Params(0.72, 5, 6, 12000)
    assert batch.max_actions == 20
    assert verdict.member_ids == (A, B, C)
    assert verdict.retire_ids is None
    assert verdict.merged_text is None


def test_model_may_be_null_or_absent():
    assert verdicts.parse_batch(body(model=None), {NS}).model is None
    raw = body()
    del raw["model"]
    assert verdicts.parse_batch(raw, {NS}).model is None


def _without(mapping, key):
    out = dict(mapping)
    del out[key]
    return out


BAD_BODIES = [
    body(extra=1),
    _without(body(), "run_id"),
    body(run_id=""),
    body(run_id="r" * 101),
    body(run_id=7),
    _without(body(), "author"),
    body(author=""),
    body(author=3),
    body(model=5),
    body(dry_run="yes"),
    body(threshold=0),
    body(threshold=1.01),
    body(threshold="0.8"),
    body(threshold=True),
    body(neighbors=0),
    body(neighbors=51),
    body(neighbors=2.5),
    body(max_group=1),
    body(max_group=21),
    body(max_group_chars=499),
    body(max_actions=0),
    body(max_actions=501),
    body(max_actions=True),
    {**body(), "verdicts": []},
    {**body(), "verdicts": [item()] * 201},
    {**body(), "verdicts": "x"},
    {**body(), "verdicts": ["x"]},
    body(item(extra=1)),
    body(_without(item(), "group_key")),
    body(_without(item(), "reason")),
    body(item(namespace="nowhere")),
    body(item(namespace=3)),
    body(item(group_key="")),
    body(item(idempotency_key="")),
    body(item(idempotency_key="k" * 201)),
    body(item(member_ids=[A])),
    body(item(member_ids=[f"note:work:{i}" for i in range(21)])),
    body(item(member_ids=[A, 3])),
    body(item(member_ids=A)),
    body(item(action="delete")),
    body(item(reason="")),
    body(item(reason="r" * 1001)),
    body(item("keep", retire_ids=[A])),
    body(item("merge", retire_ids=[A], merged_text=MERGED)),
    body(item("keep", merged_text=MERGED)),
    body(item("retire", retire_ids=[A], merged_text=MERGED)),
    body(item("retire", retire_ids=A)),
    body(item("retire", retire_ids=[1])),
    body(item("merge", merged_text=7)),
]


@pytest.mark.parametrize("raw", BAD_BODIES)
def test_parse_batch_rejects_a_bad_body(raw):
    with pytest.raises(verdicts.RequestError):
        verdicts.parse_batch(raw, {NS})


def test_payload_hash_covers_the_item_run_author_model_and_group_params():
    batch, verdict = parsed()
    digest = verdicts.payload_hash(verdict, batch)
    assert digest == verdicts.payload_hash(*reversed(parsed()))
    for top in (
        {"run_id": "run-2"},
        {"author": "other"},
        {"model": "other"},
        {"threshold": 0.8},
        {"neighbors": 6},
        {"max_group": 5},
        {"max_group_chars": 600},
    ):
        other_batch, other_verdict = parsed(top=top)
        assert verdicts.payload_hash(other_verdict, other_batch) != digest
    other_batch, other_verdict = parsed(reason="another reason")
    assert verdicts.payload_hash(other_verdict, other_batch) != digest


def test_payload_hash_ignores_dry_run_and_the_action_cap():
    batch, verdict = parsed()
    for top in ({"dry_run": True}, {"max_actions": 3}):
        other_batch, other_verdict = parsed(top=top)
        assert verdicts.payload_hash(other_verdict, other_batch) == verdicts.payload_hash(
            verdict, batch
        )


# ---- token check ------------------------------------------------------------


def test_numbers_and_dates_are_tokens():
    found = verdicts.tokens(
        "Ship on 2026-09-30 at 04:00 for 1,500 users, offset -5, score 0.72, rate 1.5e3."
    )
    assert {"2026-09-30", "04:00", "1,500", "-5", "0.72", "1.5e3"} <= found


def test_digits_inside_identifiers_are_not_tokens():
    assert verdicts.tokens("see abc123 and note:x9 here") == set()


def test_backticked_spans_are_tokens():
    assert "uv sync --all-extras" in verdicts.tokens("run `uv sync --all-extras` first")


def test_capitalized_words_are_names_unless_they_start_a_sentence():
    assert verdicts.tokens("We met Alice in Paris.") == {"Alice", "Paris"}
    assert verdicts.tokens("Alice met Bob.") == {"Bob"}
    assert verdicts.tokens("Done. Alice left! Bob stayed? Carol: Dave came") == set()
    assert verdicts.tokens("first line\nAlice\n- Bob\n* Carol\n1. Dave\n 2) Erin") == set()


def test_a_capital_right_after_punctuation_without_a_space_is_a_name():
    assert verdicts.tokens("see foo.Bar and x:Baz or wow!Qux") == {"Bar", "Baz", "Qux"}
    assert verdicts.tokens("Done.Alice left.") == {"Alice"}


def test_the_single_word_i_is_not_a_name():
    assert verdicts.tokens("Then I left, and I said so.") == set()
    assert verdicts.token_check(["The deploy I run is nightly."], "The deploy is nightly.") is None


def test_a_name_moved_from_a_sentence_start_to_mid_sentence_is_not_added():
    members = ["Alice asked me to draft replies.", "Replies follow the size rule."]
    merged = "Replies follow the size rule, and Alice asked me to draft replies."
    assert verdicts.token_check(members, merged) is None


def test_a_name_moved_from_mid_sentence_to_a_sentence_start_is_not_dropped():
    members = ["We met Alice in Paris."]
    assert verdicts.token_check(members, "Alice lives in Paris.") is None


def test_a_hyphenated_name_moved_from_a_sentence_start_is_not_added():
    members = ["Youngwoo-kun asked me to draft replies.", "Replies follow the size rule."]
    merged = "Replies follow the size rule, and Youngwoo-kun asked me to draft replies."
    assert verdicts.token_check(members, merged) is None


def test_a_changed_spelling_of_a_name_is_dropped_and_added():
    reason = verdicts.token_check(["We asked Youngwoo today."], "We asked Youngwoo-kun today.")
    assert reason == "merged_text drops Youngwoo and adds Youngwoo-kun"


def test_a_new_mid_sentence_name_is_still_added():
    reason = verdicts.token_check(["Alice left."], "Alice left with Bob.")
    assert reason == "merged_text adds Bob"


def test_a_removed_mid_sentence_name_is_still_dropped():
    reason = verdicts.token_check(["We met Alice in Paris."], "We met in Paris.")
    assert reason == "merged_text drops Alice"


def test_the_single_word_i_is_never_a_name_at_a_sentence_start():
    assert verdicts.tokens("I left.", include_sentence_starts=True) == set()
    assert verdicts.token_check(["I left. Then I ran."], "I ran.") is None


def test_include_sentence_starts_counts_a_sentence_initial_name_but_not_a_list_marker():
    assert verdicts.tokens("Alice met Bob. Done.", include_sentence_starts=True) == {
        "Alice",
        "Bob",
        "Done",
    }
    assert verdicts.tokens("- Alice\n1. Bob", include_sentence_starts=True) == {"Alice", "Bob"}


def test_a_list_marker_is_not_a_number_but_a_number_in_the_line_is():
    assert verdicts.tokens("1. ship at 04:00\n2) retry 3 times") == {"04:00", "3"}
    assert verdicts.tokens("costs 1. That is all") == {"1"}
    assert verdicts.tokens("1.5 is the ratio") == {"1.5"}


def test_digits_inside_a_version_identifier_are_not_tokens():
    assert verdicts.tokens("upgrade to v2.0 today") == set()
    assert verdicts.token_check(["upgrade to v2.0 today"], "upgrade to v3.1 today") is None


def test_words_with_two_capitals_are_names_anywhere():
    assert verdicts.tokens("GLM and PR and iOS and McDonald") == {"GLM", "PR", "iOS", "McDonald"}
    assert verdicts.tokens("PostgreSQL runs here.") == {"PostgreSQL"}


def test_a_dropped_count_is_refused():
    reason = verdicts.token_check(["100 requests, 10 retries"], "100 requests")
    assert reason is not None and "10" in reason


def test_a_dropped_exponent_number_is_refused():
    reason = verdicts.token_check(["The rate limit is 1e3 per minute."], "The rate is high.")
    assert reason is not None and "1e3" in reason


def test_an_added_name_is_refused():
    reason = verdicts.token_check(["the deploy runs nightly"], "the deploy runs on Kubernetes")
    assert reason is not None and "Kubernetes" in reason


def test_every_members_tokens_must_survive():
    reason = verdicts.token_check(["staging uses port 5433", "prod uses port 5432"], "port 5433")
    assert reason is not None and "5432" in reason


def test_a_merge_keeping_every_token_passes():
    assert verdicts.token_check(list(TEXTS.values())[:3], MERGED) is None


@pytest.mark.parametrize(
    "members,merged",
    [
        (["Alice owns the deploy."], "Bob owns the deploy."),
        (["the user likes green tea"], "the user does not like green tea"),
        (["민수는 녹차를 좋아한다."], "지영은 녹차를 좋아한다."),
    ],
)
def test_known_limits_pass(members, merged):
    assert verdicts.token_check(members, merged) is None


# ---- plan: status paths -----------------------------------------------------


def test_keep_records_the_verdict_only():
    assert plan("keep") == verdicts.Plan("keep", (), (A, B, C), None, False, None)


def test_retire_archives_the_named_members():
    assert plan("retire", retire_ids=[C, A]) == verdicts.Plan(
        "retire", (A, C), (B,), None, False, None
    )


@pytest.mark.parametrize(
    "retire_ids",
    [None, [], [A, A], ["note:work:zzz"], [A, B, C]],
)
def test_a_bad_retire_is_rejected(retire_ids):
    fields = {} if retire_ids is None else {"retire_ids": retire_ids}
    result = plan("retire", **fields)
    assert result["status"] == "rejected"
    assert result["action_id"] is None


def test_a_merge_creates_a_replacement():
    rid = replacement_id()
    assert plan("merge", merged_text=f"  {MERGED}\n") == verdicts.Plan(
        "merge", (A, B, C), (rid,), rid, True, None
    )


def test_a_merge_equal_to_one_member_after_normalizing_is_a_retire_into_it():
    result = plan("merge", merged_text="  Deploys  run at\t04:00 on weekdays. ")
    assert isinstance(result, verdicts.Plan)
    assert (result.action, result.archived_ids, result.survivor_ids) == ("retire", (B, C), (A,))
    assert result.replacement_id is None
    assert A in result.reason


def test_a_merge_differing_only_in_case_is_not_a_retire():
    result = plan("merge", merged_text="deploys run at 04:00 on weekdays.")
    assert result.action == "merge"


def test_a_merge_reuses_an_identical_active_note():
    rid = replacement_id()
    existing = verdicts.Existing(rid, "agent_note", "work", MERGED, False)
    assert plan("merge", {"replacement": existing}, merged_text=MERGED) == verdicts.Plan(
        "merge", (A, B, C), (rid,), rid, False, None
    )


@pytest.mark.parametrize(
    "source_type,kind,text",
    [
        ("agent_note", "personal", MERGED),
        ("document", "work", MERGED),
        ("agent_note", "work", MERGED + " "),
    ],
)
def test_a_merge_onto_a_different_active_row_is_rejected(source_type, kind, text):
    existing = verdicts.Existing(replacement_id(), source_type, kind, text, False)
    result = plan("merge", {"replacement": existing}, merged_text=MERGED)
    assert result["status"] == "rejected"


def test_a_merge_identical_to_an_archived_note_is_rejected():
    rid = replacement_id()
    existing = verdicts.Existing(rid, "agent_note", "work", MERGED, True)
    result = plan("merge", {"replacement": existing}, merged_text=MERGED)
    assert result["status"] == "rejected"
    assert f"identical to archived note {rid}; restore it instead" in result["reason"]


@pytest.mark.parametrize(
    "fields,needle",
    [
        ({}, "content"),
        ({"merged_text": "   "}, "content"),
        ({"merged_text": "x" * 4001}, "4000"),
        ({"merged_text": MERGED + " ghp_abcdefghijklmnopqrstuvwxyz0123456789"}, "credential"),
        ({"merged_text": "Weekday deploys run."}, "04:00"),
        ({"merged_text": MERGED + " Owned by Infra."}, "Infra"),
    ],
)
def test_a_bad_merge_text_is_rejected(fields, needle):
    result = plan("merge", **fields)
    assert result["status"] == "rejected"
    assert needle in result["reason"]


def test_a_merge_across_kinds_is_rejected():
    mixed = {**NOTES, C: note(C, kind="personal")}
    batch, verdict = parsed("merge", merged_text=MERGED, group_key=group_key(NS, mixed.values()))
    result = verdicts.plan_verdict(verdict, batch, state(notes=mixed))
    assert result["status"] == "rejected"
    assert "kind" in result["reason"]


def test_a_retried_payload_is_a_duplicate_of_the_recorded_result():
    batch, verdict = parsed("retire", retire_ids=[A])
    recorded_result = {
        "group_key": KEY,
        "status": "applied",
        "reason": None,
        "action_id": 9,
        "archived_ids": [A],
        "survivor_ids": [B, C],
        "replacement_id": None,
        "current_groups": None,
    }
    recorded = verdicts.Recorded(verdicts.payload_hash(verdict, batch), recorded_result)
    result = verdicts.plan_verdict(
        verdict, batch, state(recorded=recorded, cached="group already judged")
    )
    assert result == {**recorded_result, "status": "duplicate"}


def test_an_idempotency_key_reused_with_another_payload_is_rejected():
    recorded = verdicts.Recorded("0" * 64, {"status": "applied"})
    result = plan("keep", {"recorded": recorded})
    assert result["status"] == "rejected"
    assert result["reason"] == "idempotency key reused with a different payload"


def test_a_cached_group_is_reported_cached():
    result = plan("retire", {"cached": "group already judged"}, retire_ids=[A])
    assert result["status"] == "cached"
    assert result["reason"] == "group already judged"


def _assert_stale(result, *expected_members):
    assert result["status"] == "stale"
    assert result["action_id"] is None
    assert [sorted(m["id"] for m in g["members"]) for g in result["current_groups"]] == [
        sorted(expected_members)
    ]


def test_a_member_text_change_is_stale():
    changed = {**NOTES, B: note(B, "Weekday deploys start at 05:00.")}
    _assert_stale(plan("keep", {"notes": changed}), A, B, C)


def test_a_new_clique_member_is_stale():
    grown = {**NOTES, D: note(D)}
    pairs = [*PAIRS, Pair(A, D, 0.77), Pair(B, D, 0.76), Pair(C, D, 0.74)]
    _assert_stale(plan("keep", {"notes": grown, "pairs": pairs}), A, B, C, D)


def test_a_submitted_subset_of_a_larger_issued_group_is_stale():
    subset_key = group_key(NS, [NOTES[A], NOTES[B]])
    result = plan("keep", group_key=subset_key, member_ids=[A, B])
    _assert_stale(result, A, B, C)


def test_member_ids_that_differ_from_the_issued_group_are_stale():
    _assert_stale(plan("keep", member_ids=[A, B]), A, B, C)


def test_a_group_with_no_current_overlap_is_stale_with_no_groups():
    result = plan("keep", {"pairs": []})
    assert result["status"] == "stale"
    assert result["current_groups"] == []


def test_the_issued_group_uses_the_requests_group_params():
    _assert_stale(plan("keep", top={"threshold": 0.79}), A, B)
    assert plan("keep", top={"max_group": 2})["status"] == "stale"


def test_retire_and_merge_stop_at_the_action_cap():
    over = {"used_actions": 3}
    assert plan("retire", over, top={"max_actions": 3}, retire_ids=[A]) == {
        "group_key": KEY,
        "status": "rejected",
        "reason": "action cap reached",
        "action_id": None,
        "archived_ids": [],
        "survivor_ids": [],
        "replacement_id": None,
        "current_groups": None,
    }
    assert plan("merge", over, top={"max_actions": 3}, merged_text=MERGED)["status"] == ("rejected")
    assert isinstance(plan("retire", over, top={"max_actions": 4}, retire_ids=[A]), verdicts.Plan)


def test_keep_is_allowed_at_the_action_cap():
    assert plan("keep", {"used_actions": 3}, top={"max_actions": 3}).action == "keep"


# ---- apply flow -------------------------------------------------------------


class FakeTx:
    def __init__(self, conn, options):
        self.conn = conn
        self.options = options

    async def __aenter__(self):
        self.conn.log.append(("begin", self.options))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.conn.log.append(("end", exc_type is None))
        return False


class FakeConn:
    def __init__(self, log, committed=None):
        self.log = log
        self.committed = committed

    def transaction(self, **options):
        return FakeTx(self, options)

    async def execute(self, query, *args):
        self.log.append(("execute", " ".join(query.split()), args))
        return "SET"

    async def fetchrow(self, query, *args):
        self.log.append(("fetchrow", " ".join(query.split()), args))
        return self.committed


APPLIED = {
    "group_key": KEY,
    "status": "applied",
    "reason": None,
    "action_id": 1,
    "archived_ids": [],
    "survivor_ids": [A, B, C],
    "replacement_id": None,
    "current_groups": None,
}


@pytest.fixture()
def flow(monkeypatch):
    """Stub the readers, the writer, and the embedder; record what the flow calls."""
    log: list = []
    states: list = []
    box = {"committed": None, "apply_error": None}

    @asynccontextmanager
    async def acquire(timeout=None):
        yield FakeConn(log, box["committed"])

    async def noop(conn):
        return None

    async def load_state(conn, verdict, batch):
        log.append(("load", verdict.idempotency_key))
        return states.pop(0) if states else state()

    async def lock_rows(conn, ids):
        log.append(("lock", tuple(ids)))
        return {i: {"id": i, "metadata": json.dumps({"tags": ["deploy"]})} for i in ids}

    async def apply_plan(conn, verdict, batch, plan_, state_, rows, embedding):
        log.append(("apply", plan_, embedding))
        if box["apply_error"] is not None:
            raise box["apply_error"]
        return {**APPLIED, "archived_ids": list(plan_.archived_ids)}

    async def embed_text(embedder, text):
        log.append(("embed", text))
        return "[0.1]"

    monkeypatch.setattr(verdicts.db, "acquire", acquire)
    monkeypatch.setattr(verdicts, "ensure_schema_once", noop)
    monkeypatch.setattr(verdicts, "load_state", load_state)
    monkeypatch.setattr(verdicts, "lock_rows", lock_rows)
    monkeypatch.setattr(verdicts, "apply_plan", apply_plan)
    monkeypatch.setattr(verdicts, "embed_text", embed_text)
    return {"log": log, "states": states, "box": box}


def _events(log, kind):
    return [entry for entry in log if entry[0] == kind]


def _run(batch):
    return asyncio.run(verdicts.process_batch(batch))


def test_a_dry_run_returns_the_plan_without_embedding_or_writing(flow):
    batch, _ = parsed("merge", top={"dry_run": True}, merged_text=MERGED)
    [result] = _run(batch)
    rid = replacement_id()
    assert result == {
        "group_key": KEY,
        "status": "planned",
        "reason": None,
        "action_id": None,
        "archived_ids": [A, B, C],
        "survivor_ids": [rid],
        "replacement_id": rid,
        "current_groups": None,
    }
    assert _events(flow["log"], "begin") == [
        ("begin", {"isolation": "repeatable_read", "readonly": True})
    ]
    assert _events(flow["log"], "embed") == []
    assert _events(flow["log"], "apply") == []
    assert _events(flow["log"], "lock") == []


def test_a_dry_run_of_an_invalid_verdict_is_rejected_not_planned(flow):
    batch, _ = parsed("retire", top={"dry_run": True}, retire_ids=[A, B, C])
    [result] = _run(batch)
    assert result["status"] == "rejected"


def test_the_preflight_reads_with_exact_search_settings(flow):
    batch, _ = parsed(top={"dry_run": True})
    _run(batch)
    statements = [entry[1] for entry in _events(flow["log"], "execute")]
    assert "SET LOCAL enable_indexscan = off" in statements
    assert "SET LOCAL enable_bitmapscan = off" in statements
    first_load = flow["log"].index(("load", "k-1"))
    assert all(flow["log"].index(e) < first_load for e in _events(flow["log"], "execute"))


def test_apply_embeds_first_then_locks_and_replans_inside_one_transaction(flow):
    batch, _ = parsed("merge", merged_text=f" {MERGED} ")
    [result] = _run(batch)
    assert result["status"] == "applied"
    log = flow["log"]
    kinds = [entry[0] for entry in log]
    embed_at = kinds.index("embed")
    second_begin = [i for i, k in enumerate(kinds) if k == "begin"][1]
    assert embed_at < second_begin
    assert log[embed_at] == ("embed", MERGED)
    assert log[second_begin] == ("begin", {})
    after = log[second_begin:]
    lock_sql = next(e for e in after if e[0] == "execute")
    assert "pg_advisory_xact_lock" in lock_sql[1]
    assert "hashtextextended('consolidate:' || $1, 0)" in lock_sql[1]
    assert lock_sql[2] == (NS,)
    order = [e[0] for e in after]
    assert order.index("lock") < order.index("load") < order.index("apply")
    assert _events(after, "lock") == [("lock", tuple(sorted([A, B, C, replacement_id()])))]
    [apply] = _events(log, "apply")
    assert apply[2] == "[0.1]"


def test_apply_without_a_new_replacement_does_not_embed(flow):
    batch, _ = parsed("retire", retire_ids=[A])
    [result] = _run(batch)
    assert result["status"] == "applied"
    assert _events(flow["log"], "embed") == []


def test_a_terminal_result_under_the_lock_is_returned_as_is(flow):
    batch, verdict = parsed("retire", retire_ids=[A])
    recorded = verdicts.Recorded(verdicts.payload_hash(verdict, batch), APPLIED)
    flow["states"].extend([state(), state(recorded=recorded)])
    [result] = _run(batch)
    assert result == {**APPLIED, "status": "duplicate"}
    assert _events(flow["log"], "apply") == []


def test_a_plan_that_changed_since_the_preflight_is_stale(flow):
    rid = replacement_id()
    existing = verdicts.Existing(rid, "agent_note", "work", MERGED, False)
    flow["states"].extend([state(replacement=existing), state()])
    batch, _ = parsed("merge", merged_text=MERGED)
    [result] = _run(batch)
    assert result["status"] == "stale"
    assert _events(flow["log"], "apply") == []


def test_each_verdict_is_processed_alone(flow):
    changed = {**NOTES, B: note(B, "Weekday deploys start at 05:00.")}
    flow["states"].extend([state(notes=changed), state(), state()])
    raw = body(item("keep"), item("retire", idempotency_key="k-2", retire_ids=[A]))
    results = _run(verdicts.parse_batch(raw, {NS}))
    assert [r["status"] for r in results] == ["stale", "applied"]


def test_a_rolled_back_verdict_returns_its_result_and_the_batch_continues(flow):
    stale = {**APPLIED, "status": "stale", "action_id": None}
    flow["box"]["apply_error"] = verdicts.Rollback(stale)
    raw = body(item("keep"), item("keep", idempotency_key="k-2"))
    results = _run(verdicts.parse_batch(raw, {NS}))
    assert results == [stale, stale]


def _unique(constraint):
    exc = asyncpg.UniqueViolationError("duplicate key")
    exc.constraint_name = constraint
    return exc


def test_a_concurrent_insert_of_the_same_payload_is_a_duplicate(flow):
    batch, verdict = parsed("retire", retire_ids=[A])
    flow["box"]["apply_error"] = _unique("consolidation_actions_idempotency_key_key")
    flow["box"]["committed"] = {
        "payload_hash": verdicts.payload_hash(verdict, batch),
        "result": json.dumps(APPLIED),
    }
    [result] = _run(batch)
    assert result == {**APPLIED, "status": "duplicate"}


def test_a_failing_embedder_fails_only_its_verdict(flow, monkeypatch):
    calls = []

    async def embed_text(embedder, text):
        calls.append(text)
        if len(calls) == 2:
            raise TimeoutError("embedding endpoint timed out")
        return "[0.1]"

    monkeypatch.setattr(verdicts, "embed_text", embed_text)
    raw = body(
        item("merge", idempotency_key="k-1", merged_text=MERGED),
        item("merge", idempotency_key="k-2", merged_text=MERGED),
        item("merge", idempotency_key="k-3", merged_text=MERGED),
    )
    results = _run(verdicts.parse_batch(raw, {NS}))
    assert [r["status"] for r in results] == ["applied", "failed", "applied"]
    failed = results[1]
    assert failed["reason"] == "TimeoutError: embedding endpoint timed out"
    assert failed["action_id"] is None
    assert failed["archived_ids"] == [] and failed["replacement_id"] is None
    assert len(_events(flow["log"], "apply")) == 2


def test_a_failing_transaction_fails_only_its_verdict(flow):
    flow["box"]["apply_error"] = asyncpg.exceptions.DeadlockDetectedError("deadlock detected")
    raw = body(item("keep", idempotency_key="k-1"))
    [result] = _run(verdicts.parse_batch(raw, {NS}))
    assert result["status"] == "failed"
    assert result["reason"] == "DeadlockDetectedError: deadlock detected"


def test_a_long_failure_message_is_cut_short(flow):
    flow["box"]["apply_error"] = RuntimeError("x" * 5000)
    [result] = _run(verdicts.parse_batch(body(item("keep")), {NS}))
    assert result["status"] == "failed"
    assert len(result["reason"]) <= verdicts.MAX_FAILURE_CHARS


def test_a_group_key_conflict_is_a_failure_not_a_cache_hit(flow):
    flow["box"]["apply_error"] = _unique("consolidation_actions_group_key_key")
    flow["box"]["committed"] = None
    [result] = _run(verdicts.parse_batch(body(item("keep")), {NS}))
    assert result["status"] == "failed"
    assert result["reason"].startswith("UniqueViolationError")


class StateConn:
    """Answers load_state's reads: no recorded verdict, no judged key, an undone member set."""

    def __init__(self, undone):
        self.undone = undone
        self.calls: list[tuple[str, tuple]] = []

    async def fetchrow(self, query, *args):
        self.calls.append((" ".join(query.split()), args))
        return None

    async def fetchval(self, query, *args):
        self.calls.append((" ".join(query.split()), args))
        if "undone_at IS NOT NULL" in query:
            return self.undone
        if "group_key = $1" in query:
            return False
        raise AssertionError(query)

    async def fetch(self, query, *args):
        raise AssertionError("a cached group needs no discovery")


def test_a_member_set_equal_to_an_undone_actions_is_cached():
    batch, verdict = parsed("retire", retire_ids=[A], member_ids=[C, A, B])
    conn = StateConn(undone=True)
    loaded = asyncio.run(verdicts.load_state(conn, verdict, batch))
    assert loaded.cached is not None and "undone" in loaded.cached
    [undone_query] = [c for c in conn.calls if "undone_at IS NOT NULL" in c[0]]
    assert undone_query[1] == (NS, [A, B, C])
    result = verdicts.plan_verdict(verdict, batch, loaded)
    assert result["status"] == "cached"
    assert result["reason"] == loaded.cached


def test_a_created_replacement_without_an_embedding_is_refused(no_owning_helpers):
    batch, verdict, loaded, planned, rows, _ = _merge_inputs()
    with pytest.raises(ValueError, match="embedding"):
        asyncio.run(
            verdicts.apply_plan(RecordingConn(), verdict, batch, planned, loaded, rows, None)
        )


def test_a_concurrent_insert_of_another_payload_is_rejected(flow):
    batch, _ = parsed("retire", retire_ids=[A])
    flow["box"]["apply_error"] = _unique("consolidation_actions_idempotency_key_key")
    flow["box"]["committed"] = {"payload_hash": "0" * 64, "result": json.dumps(APPLIED)}
    [result] = _run(batch)
    assert result["status"] == "rejected"
    assert result["reason"] == "idempotency key reused with a different payload"


# ---- the write --------------------------------------------------------------


class RecordingConn:
    def __init__(self, action_id=7, inserted=True):
        self.action_id = action_id
        self.inserted = inserted
        self.calls: list[tuple[str, str, tuple]] = []

    def _record(self, kind, query, args):
        self.calls.append((kind, " ".join(query.split()), args))

    async def fetchval(self, query, *args):
        self._record("fetchval", query, args)
        if "nextval" in query:
            return self.action_id
        if "RETURNING id" in query:
            return args[0] if self.inserted else None
        raise AssertionError(query)

    async def execute(self, query, *args):
        self._record("execute", query, args)
        return "UPDATE 1"

    def find(self, needle):
        return [call for call in self.calls if needle in call[1]]


@pytest.fixture()
def no_owning_helpers(monkeypatch):
    async def refuse(*args, **kwargs):
        raise AssertionError("connection-owning helper called inside the apply transaction")

    monkeypatch.setattr(notes, "save_note", refuse)
    monkeypatch.setattr(admin, "archive_rows", refuse)
    monkeypatch.setattr(admin, "restore_rows", refuse)


def _merge_inputs():
    members = {
        A: note(A, occurred_at=1_690_000_000.0, tags=("deploy", "ops")),
        B: note(B, occurred_at=1_695_000_000.0, saved=1_700_000_100.0),
        C: note(C, tags=("schedule",)),
    }
    batch, verdict = parsed(
        "merge", merged_text=f" {MERGED} ", group_key=group_key(NS, members.values())
    )
    loaded = state(notes=members)
    planned = verdicts.plan_verdict(verdict, batch, loaded)
    rows = {
        i: {"id": i, "metadata": json.dumps({"tags": list(members[i].tags), "author": "x"})}
        for i in members
    }
    return batch, verdict, loaded, planned, rows, members


def test_apply_plan_writes_the_replacement_archives_members_and_records_the_action(
    no_owning_helpers,
):
    batch, verdict, loaded, planned, rows, members = _merge_inputs()
    conn = RecordingConn()
    result = asyncio.run(verdicts.apply_plan(conn, verdict, batch, planned, loaded, rows, "[0.1]"))
    rid = replacement_id()
    assert result == {
        "group_key": verdict.group_key,
        "status": "applied",
        "reason": None,
        "action_id": 7,
        "archived_ids": [A, B, C],
        "survivor_ids": [rid],
        "replacement_id": rid,
        "current_groups": None,
    }
    [insert] = conn.find("RETURNING id")
    assert "ON CONFLICT (id) DO NOTHING" in insert[1]
    args = insert[2]
    assert args[0] == rid
    assert MERGED in args and "[0.1]" in args and NS in args and "work" in args
    assert 1_695_000_000.0 in args
    metadata = next(json.loads(a) for a in args if isinstance(a, str) and a.startswith("{"))
    assert metadata["tags"] == ["deploy", "ops", "schedule"]
    assert metadata["author"] == "consolidator"
    assert metadata["merged_from"] == [A, B, C]
    assert metadata["consolidation_action"] == 7
    assert metadata["merged_dates"][A] == {
        "saved": "2023-11-14T22:13:20+00:00",
        "occurred_at": "2023-07-22T04:26:40+00:00",
    }
    assert metadata["merged_dates"][C]["occurred_at"] is None

    [archive] = conn.find("SET archived_at")
    assert "'consolidated_into'" in archive[1] and "'archived_by'" in archive[1]
    ids, applied_at, author, into = archive[2]
    assert (ids, author, json.loads(into)) == ([A, B, C], "consolidator", [rid])

    [action] = conn.find("INSERT INTO")[-1:]
    assert "consolidation_actions" in action[1]
    values = action[2]
    assert 7 in values and verdict.idempotency_key in values and applied_at in values
    prior = next(json.loads(v) for v in values if isinstance(v, str) and v.startswith('{"note'))
    assert prior == {i: json.loads(rows[i]["metadata"]) for i in (A, B, C)}
    assert json.loads(next(v for v in values if isinstance(v, str) and '"applied"' in v)) == (
        result
    )


def test_apply_plan_rolls_back_when_the_replacement_appeared_concurrently(no_owning_helpers):
    batch, verdict, loaded, planned, rows, _ = _merge_inputs()
    conn = RecordingConn(inserted=False)
    with pytest.raises(verdicts.Rollback) as caught:
        asyncio.run(verdicts.apply_plan(conn, verdict, batch, planned, loaded, rows, "[0.1]"))
    assert caught.value.result["status"] == "stale"
    assert caught.value.result["reason"] == "replacement appeared concurrently"


def test_apply_plan_records_a_keep_without_touching_notes(no_owning_helpers):
    batch, verdict = parsed("keep")
    loaded = state()
    planned = verdicts.plan_verdict(verdict, batch, loaded)
    conn = RecordingConn()
    result = asyncio.run(verdicts.apply_plan(conn, verdict, batch, planned, loaded, {}, None))
    assert result["status"] == "applied"
    assert result["survivor_ids"] == [A, B, C]
    assert conn.find("memory_chunks") == []
    assert len(conn.find("consolidation_actions")) == 2


def test_apply_plan_leaves_a_retire_survivor_unchanged(no_owning_helpers):
    batch, verdict = parsed("retire", retire_ids=[A])
    loaded = state()
    planned = verdicts.plan_verdict(verdict, batch, loaded)
    rows = {i: {"id": i, "metadata": "{}"} for i in (A, B, C)}
    conn = RecordingConn()
    asyncio.run(verdicts.apply_plan(conn, verdict, batch, planned, loaded, rows, None))
    [archive] = conn.find("SET archived_at")
    assert archive[2][0] == [A]
    assert json.loads(archive[2][3]) == [B, C]
    assert conn.find("RETURNING id") == []


# ---- route ------------------------------------------------------------------


def _use_identity(monkeypatch, is_admin=True, authors=("consolidator",)):
    identity = auth.KeyIdentity(
        key_id="consolidator-key-hash",
        label="consolidator",
        home="default",
        is_admin=is_admin,
        allowed=frozenset({"default"}),
        authors=frozenset(authors),
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "test-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)


@pytest.fixture()
def route(monkeypatch):
    async def fake_list_namespaces():
        return [{"name": name, "visibility": "public", "owner": None} for name in ("default", NS)]

    calls = []

    async def fake_process_batch(batch):
        calls.append(batch)
        return [{**APPLIED, "group_key": v.group_key} for v in batch.verdicts]

    monkeypatch.setattr(namespaces, "list_namespaces", fake_list_namespaces)
    monkeypatch.setattr(verdicts, "process_batch", fake_process_batch)
    return calls


def test_route_refuses_a_non_admin_key(monkeypatch, route):
    _use_identity(monkeypatch, is_admin=False)
    response = client.post("/admin/consolidate/verdicts", json=body())
    assert response.status_code == 403
    assert route == []


def test_route_refuses_an_admin_key_without_the_consolidator_author(route):
    response = client.post("/admin/consolidate/verdicts", json=body())
    assert response.status_code == 403
    assert "consolidator" in response.json()["error"]


def test_route_refuses_an_author_outside_the_keys_authors(monkeypatch, route):
    _use_identity(monkeypatch)
    response = client.post("/admin/consolidate/verdicts", json=body(author="natsume"))
    assert response.status_code == 403
    assert route == []


@pytest.mark.parametrize(
    "raw",
    [
        body(extra=1),
        body(item(namespace="nowhere")),
        body(item("merge", retire_ids=[A], merged_text=MERGED)),
        body(threshold=0),
    ],
)
def test_route_rejects_a_bad_body_whole(monkeypatch, route, raw):
    _use_identity(monkeypatch)
    response = client.post("/admin/consolidate/verdicts", json=raw)
    assert response.status_code == 400
    assert "error" in response.json()
    assert route == []


def test_route_rejects_a_non_object_body(monkeypatch, route):
    _use_identity(monkeypatch)
    response = client.post("/admin/consolidate/verdicts", json=[body()])
    assert response.status_code == 400


def test_route_returns_one_result_per_verdict(monkeypatch, route):
    _use_identity(monkeypatch)
    raw = body(item(), item(idempotency_key="k-2"))
    response = client.post("/admin/consolidate/verdicts", json=raw)
    assert response.status_code == 200
    assert response.json() == {"results": [APPLIED, APPLIED]}
    [batch] = route
    assert [v.idempotency_key for v in batch.verdicts] == ["k-1", "k-2"]
