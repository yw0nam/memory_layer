"""Unit tests for GET /admin/consolidate/groups and its pure grouping functions.

No DB, no network: grouping and the group key are pure functions over fixtures, and the
route reads through a fake connection behind ``db.acquire`` with the namespace registry
stubbed.
"""

from __future__ import annotations

import dataclasses
import json
import random
from contextlib import asynccontextmanager

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api, auth, namespaces
from memory_base.serve.consolidation import groups
from memory_base.serve.consolidation.groups import (
    Deferred,
    Group,
    Note,
    Pair,
    acknowledged_pairs,
    build_groups,
    group_key,
)

client = TestClient(api.app, headers={"X-API-Key": "test-key"})


def note(note_id, text="x" * 10, **fields):
    values = {
        "id": note_id,
        "kind": "work",
        "author": "claude-code",
        "saved": 1_700_000_000.0,
        "occurred_at": None,
        "tags": ("rules",),
        "similar_ack": (),
        "supersedes": None,
        "text": text,
    }
    values.update(fields)
    return Note(**values)


def notes_for(*ids, **texts):
    return {note_id: note(note_id, texts.get(note_id, "x" * 10)) for note_id in ids}


def group(*members, min_score, max_score):
    return Group(members=tuple(members), min_score=min_score, max_score=max_score)


def run(pairs, notes, threshold=0.72, max_group=6, max_group_chars=12000):
    return build_groups(pairs, notes, threshold, max_group, max_group_chars)


# ---- grouping ---------------------------------------------------------------


def test_chain_without_the_closing_edge_never_forms_one_group():
    pairs = [Pair("A", "B", 0.80), Pair("B", "C", 0.78)]
    groups, deferred = run(pairs, notes_for("A", "B", "C"))
    assert groups == [group("A", "B", min_score=0.80, max_score=0.80)]
    assert deferred == [Deferred("C", "no clique")]


def test_growth_picks_the_candidate_with_the_highest_minimum_edge():
    pairs = [
        Pair("A", "B", 0.95),
        Pair("A", "C", 0.90),
        Pair("B", "C", 0.75),
        Pair("A", "D", 0.80),
        Pair("B", "D", 0.80),
    ]
    groups, deferred = run(pairs, notes_for("A", "B", "C", "D"))
    assert groups == [group("A", "B", "D", min_score=0.80, max_score=0.95)]
    assert deferred == [Deferred("C", "no clique")]


def test_growth_breaks_a_tie_by_id():
    pairs = [
        Pair("A", "B", 0.95),
        Pair("A", "C", 0.80),
        Pair("B", "C", 0.80),
        Pair("A", "D", 0.80),
        Pair("B", "D", 0.80),
    ]
    groups, deferred = run(pairs, notes_for("A", "B", "C", "D"))
    assert groups == [group("A", "B", "C", min_score=0.80, max_score=0.95)]
    assert deferred == [Deferred("D", "no clique")]


def test_max_group_stops_growth_and_reports_the_leftover():
    pairs = [
        Pair("A", "B", 0.95),
        Pair("A", "C", 0.90),
        Pair("A", "D", 0.90),
        Pair("B", "C", 0.90),
        Pair("B", "D", 0.90),
        Pair("C", "D", 0.90),
    ]
    groups, deferred = run(pairs, notes_for("A", "B", "C", "D"), max_group=3)
    assert groups == [group("A", "B", "C", min_score=0.90, max_score=0.95)]
    assert deferred == [Deferred("D", "over max_group")]


def test_a_candidate_over_the_char_cap_is_skipped_and_the_next_one_tried():
    pairs = [
        Pair("A", "B", 0.95),
        Pair("A", "C", 0.90),
        Pair("B", "C", 0.90),
        Pair("A", "D", 0.85),
        Pair("B", "D", 0.85),
        Pair("C", "D", 0.85),
    ]
    notes = notes_for("A", "B", "C", "D", C="c" * 50)
    groups, deferred = run(pairs, notes, max_group_chars=35)
    assert groups == [group("A", "B", "D", min_score=0.85, max_score=0.95)]
    assert deferred == [Deferred("C", "over max_group_chars")]


def test_a_pair_over_the_char_cap_defers_each_note():
    notes = notes_for("A", "B", A="a" * 30, B="b" * 30)
    groups, deferred = run([Pair("A", "B", 0.95)], notes, max_group_chars=50)
    assert groups == []
    assert deferred == [
        Deferred("A", "over max_group_chars"),
        Deferred("B", "over max_group_chars"),
    ]


def test_a_note_in_two_over_cap_pairs_is_deferred_once():
    pairs = [Pair("A", "B", 0.95), Pair("A", "C", 0.90)]
    notes = notes_for("A", "B", "C", A="a" * 30, B="b" * 30, C="c" * 30)
    groups, deferred = run(pairs, notes, max_group_chars=50)
    assert groups == []
    assert deferred == [
        Deferred("A", "over max_group_chars"),
        Deferred("B", "over max_group_chars"),
        Deferred("C", "over max_group_chars"),
    ]


def test_a_deferred_note_keeps_the_first_reason_recorded_for_it():
    pairs = [
        Pair("A", "B", 0.95),
        Pair("A", "C", 0.90),
        Pair("A", "D", 0.90),
        Pair("B", "C", 0.90),
        Pair("B", "D", 0.90),
        Pair("C", "D", 0.90),
        Pair("D", "E", 0.80),
    ]
    notes = notes_for("A", "B", "C", "D", "E", E="e" * 600)
    groups, deferred = run(pairs, notes, max_group=3, max_group_chars=500)
    assert groups == [group("A", "B", "C", min_score=0.90, max_score=0.95)]
    assert deferred == [
        Deferred("D", "over max_group"),
        Deferred("E", "over max_group_chars"),
    ]


def test_a_deferred_pair_drops_the_note_that_later_joins_a_group():
    pairs = [Pair("A", "B", 0.95), Pair("A", "C", 0.90)]
    notes = notes_for("A", "B", "C", A="a" * 30, B="b" * 30)
    groups, deferred = run(pairs, notes, max_group_chars=50)
    assert groups == [group("A", "C", min_score=0.90, max_score=0.90)]
    assert deferred == [Deferred("B", "over max_group_chars")]


def test_a_deferred_pair_disappears_when_both_notes_later_join_groups():
    pairs = [Pair("A", "B", 0.95), Pair("A", "C", 0.90), Pair("B", "D", 0.85)]
    notes = notes_for("A", "B", "C", "D", A="a" * 30, B="b" * 30)
    groups, deferred = run(pairs, notes, max_group_chars=50)
    assert groups == [
        group("A", "C", min_score=0.90, max_score=0.90),
        group("B", "D", min_score=0.85, max_score=0.85),
    ]
    assert deferred == []


def test_edges_below_the_threshold_do_not_count():
    pairs = [Pair("A", "B", 0.95), Pair("A", "C", 0.90), Pair("B", "C", 0.60)]
    groups, deferred = run(pairs, notes_for("A", "B", "C"), threshold=0.72)
    assert groups == [group("A", "B", min_score=0.95, max_score=0.95)]
    assert deferred == [Deferred("C", "no clique")]


@pytest.mark.parametrize("lister,listed", [("A", "B"), ("B", "A")])
def test_an_acknowledged_pair_in_either_direction_is_ignored(lister, listed):
    notes = notes_for("A", "B")
    notes[lister] = dataclasses.replace(notes[lister], similar_ack=(listed,))
    assert run([Pair("A", "B", 0.90)], notes) == ([], [])
    assert acknowledged_pairs([Pair("A", "B", 0.90)], notes) == 1


def test_an_acknowledged_pair_does_not_hide_an_unacknowledged_duplicate():
    pairs = [Pair("A", "B", 0.90), Pair("A", "C", 0.80)]
    notes = notes_for("A", "B", "C")
    notes["A"] = dataclasses.replace(notes["A"], similar_ack=("B",))
    groups, deferred = run(pairs, notes)
    assert groups == [group("A", "C", min_score=0.80, max_score=0.80)]
    assert deferred == []
    assert acknowledged_pairs(pairs, notes) == 1


def test_an_acknowledged_edge_breaks_a_clique():
    pairs = [Pair("A", "B", 0.95), Pair("A", "C", 0.90), Pair("B", "C", 0.90)]
    notes = notes_for("A", "B", "C")
    notes["C"] = dataclasses.replace(notes["C"], similar_ack=("B",))
    groups, deferred = run(pairs, notes)
    assert groups == [group("A", "B", min_score=0.95, max_score=0.95)]
    assert deferred == [Deferred("C", "no clique")]


def test_groups_are_ordered_by_highest_edge_then_smallest_id():
    pairs = [Pair("C", "D", 0.90), Pair("A", "B", 0.90), Pair("E", "F", 0.95)]
    groups, _ = run(pairs, notes_for("A", "B", "C", "D", "E", "F"))
    assert [g.members for g in groups] == [("E", "F"), ("A", "B"), ("C", "D")]


def test_shuffled_input_gives_identical_output():
    pairs = [
        Pair("A", "B", 0.90),
        Pair("A", "C", 0.90),
        Pair("B", "C", 0.85),
        Pair("C", "D", 0.88),
        Pair("D", "E", 0.80),
        Pair("B", "E", 0.80),
        Pair("F", "G", 0.90),
        Pair("E", "F", 0.75),
    ]
    notes = notes_for("A", "B", "C", "D", "E", "F", "G", G="g" * 40)
    expected = run(pairs, notes, max_group_chars=45)
    rng = random.Random(7)
    for _ in range(20):
        shuffled = [
            Pair(p.b, p.a, p.score) if rng.random() < 0.5 else p
            for p in rng.sample(pairs, len(pairs))
        ]
        assert run(shuffled, notes, max_group_chars=45) == expected


# ---- group key --------------------------------------------------------------


def _members():
    return [
        note("A", "first text", occurred_at=1_690_000_000.0, tags=("b", "a")),
        note("B", "second text", supersedes="note:old"),
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", "edited text"),
        ("kind", "personal"),
        ("author", "hermes"),
        ("saved", 1_700_000_001.0),
        ("occurred_at", 1_690_000_001.0),
        ("tags", ("a", "c")),
        ("supersedes", "note:other"),
    ],
)
def test_group_key_changes_with_each_prompt_visible_field(field, value):
    members = _members()
    changed = [dataclasses.replace(members[0], **{field: value}), members[1]]
    assert group_key("default", changed) != group_key("default", members)


def test_group_key_changes_with_membership():
    members = _members()
    assert group_key("default", [*members, note("C")]) != group_key("default", members)
    assert group_key("default", members[:1] + [note("C")]) != group_key("default", members)


def test_group_key_ignores_input_order_tag_order_and_similar_ack():
    members = _members()
    reordered = [
        members[1],
        dataclasses.replace(members[0], tags=("a", "b"), similar_ack=("B",)),
    ]
    assert group_key("default", reordered) == group_key("default", members)


def test_group_key_tells_a_null_timestamp_from_a_zero_one():
    members = _members()
    absent = [dataclasses.replace(members[0], occurred_at=None), members[1]]
    zero = [dataclasses.replace(members[0], occurred_at=0.0), members[1]]
    assert group_key("default", absent) != group_key("default", zero)


def test_group_key_is_sha256_of_canonical_json():
    import hashlib

    members = [note("A", "t", saved=0.0, tags=("z", "y"), author=None)]
    payload = {
        "v": "1",
        "namespace": "ns",
        "members": [
            [
                "A",
                hashlib.sha256(b"t").hexdigest(),
                "work",
                None,
                0.0,
                None,
                ["y", "z"],
                None,
            ]
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert group_key("ns", members) == hashlib.sha256(canonical.encode()).hexdigest()


# ---- route ------------------------------------------------------------------


def _identity(is_admin=True, authors=("consolidator",)):
    return auth.KeyIdentity(
        key_id="consolidator-key-hash",
        label="consolidator",
        home="default",
        is_admin=is_admin,
        allowed=frozenset({"default"}),
        authors=frozenset(authors),
    )


def _use_identity(monkeypatch, identity):
    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "test-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)


class FakeTransaction:
    def __init__(self, conn, options):
        self.conn = conn
        self.options = options

    async def __aenter__(self):
        self.conn.transactions.append(self.options)
        return self

    async def __aexit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, pair_rows, note_rows, action_rows=()):
        self.pair_rows = pair_rows
        self.note_rows = note_rows
        self.action_rows = list(action_rows)
        self.transactions = []
        self.executed = []
        self.fetched = []

    def transaction(self, **options):
        return FakeTransaction(self, options)

    async def execute(self, query, *args):
        assert self.transactions, "settings must run inside the snapshot transaction"
        self.executed.append(query)
        return "SET"

    async def fetch(self, query, *args):
        assert self.transactions, "reads must run inside the snapshot transaction"
        self.fetched.append((query, args))
        if "consolidation_actions" in query:
            return self.action_rows
        return self.pair_rows if "<=>" in query else self.note_rows


def _note_row(note_id, text, saved, **metadata):
    return {
        "id": note_id,
        "kind": "work",
        "ts_last_active": saved,
        "occurred_at": metadata.pop("occurred_at", None),
        "text": text,
        "metadata": json.dumps({"author": "claude-code", "tags": ["rules"], **metadata}),
    }


@pytest.fixture()
def registry(monkeypatch):
    registered = ["work", "default"]

    async def fake_list_namespaces():
        return [{"name": name, "visibility": "public", "owner": None} for name in registered]

    monkeypatch.setattr(namespaces, "list_namespaces", fake_list_namespaces)
    return registered


@pytest.fixture()
def fake_db(monkeypatch, registry):
    conn = FakeConnection([], [])

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(groups.db, "acquire", acquire)
    return conn


@pytest.fixture()
def consolidator(monkeypatch):
    _use_identity(monkeypatch, _identity())


def test_route_refuses_a_non_admin_key(monkeypatch, fake_db):
    _use_identity(monkeypatch, _identity(is_admin=False))
    response = client.get("/admin/consolidate/groups")
    assert response.status_code == 403
    assert "consolidator" in response.json()["error"]


def test_route_refuses_an_admin_key_without_the_consolidator_author(fake_db):
    response = client.get("/admin/consolidate/groups")
    assert response.status_code == 403
    assert "consolidator" in response.json()["error"]


@pytest.mark.parametrize(
    "params",
    [
        {"threshold": "0"},
        {"threshold": "1.01"},
        {"threshold": "-0.5"},
        {"threshold": "nan"},
        {"threshold": "inf"},
        {"threshold": "high"},
        {"neighbors": "0"},
        {"neighbors": "51"},
        {"neighbors": "2.5"},
        {"max_group": "1"},
        {"max_group": "21"},
        {"max_group_chars": "499"},
        {"max_group_chars": "many"},
        {"limit": "0"},
        {"limit": "1001"},
        {"namespace": " "},
        {"namespace": ""},
        {"namespace": "nowhere"},
        [("threshold", "0.8"), ("threshold", "0.9")],
        [("limit", "5"), ("limit", "5")],
    ],
)
def test_route_rejects_a_bad_parameter(consolidator, fake_db, params):
    response = client.get("/admin/consolidate/groups", params=params)
    assert response.status_code == 400
    assert "error" in response.json()
    assert fake_db.fetched == []


def test_route_resolves_defaults_and_every_registered_namespace(consolidator, fake_db):
    response = client.get("/admin/consolidate/groups")
    assert response.status_code == 200
    body = response.json()
    assert body["params"] == {
        "namespace": ["default", "work"],
        "threshold": 0.72,
        "neighbors": 5,
        "max_group": 6,
        "max_group_chars": 12000,
        "limit": 200,
    }
    assert body["procedure_version"] == "1"
    assert list(body["namespaces"]) == ["default", "work"]
    assert body["namespaces"]["work"] == {
        "active_notes": 0,
        "pairs": 0,
        "acknowledged": 0,
        "cached": 0,
        "groups": [],
        "deferred": [],
        "truncated": False,
    }


def test_route_deduplicates_and_sorts_namespaces(consolidator, fake_db):
    response = client.get(
        "/admin/consolidate/groups",
        params=[("namespace", "work"), ("namespace", "default"), ("namespace", "work")],
    )
    assert response.status_code == 200
    assert response.json()["params"]["namespace"] == ["default", "work"]
    assert list(response.json()["namespaces"]) == ["default", "work"]


def test_route_reads_one_exact_snapshot_per_namespace(consolidator, fake_db):
    response = client.get(
        "/admin/consolidate/groups",
        params={"namespace": "work", "threshold": "0.8", "neighbors": "7"},
    )
    assert response.status_code == 200
    assert fake_db.transactions == [{"isolation": "repeatable_read", "readonly": True}]
    assert "SET LOCAL enable_indexscan = off" in fake_db.executed
    assert "SET LOCAL enable_bitmapscan = off" in fake_db.executed
    pair_query, pair_args = next(f for f in fake_db.fetched if "<=>" in f[0])
    assert "source_type = 'agent_note'" in pair_query
    assert "archived_at IS NULL" in pair_query
    assert set(pair_args) >= {"work", 0.8, 7}


def test_route_response_shape_and_member_ordering(consolidator, fake_db):
    fake_db.note_rows = [
        _note_row("note:default:a", "rule one", 1_700_086_400.0, occurred_at=0.0),
        _note_row("note:default:b", "rule two", 1_700_000_000.0, supersedes="note:default:z"),
        _note_row("note:default:c", "unrelated", 1_700_000_000.0),
    ]
    fake_db.pair_rows = [
        {"a_id": "note:default:a", "b_id": "note:default:b", "score": 0.80},
        {"a_id": "note:default:b", "b_id": "note:default:a", "score": 0.81},
    ]
    response = client.get("/admin/consolidate/groups", params={"namespace": "default"})
    assert response.status_code == 200
    section = response.json()["namespaces"]["default"]
    assert section["active_notes"] == 3
    assert section["pairs"] == 1
    assert section["acknowledged"] == 0
    assert section["deferred"] == []
    assert section["truncated"] is False
    [only] = section["groups"]
    assert set(only) == {"key", "min_score", "max_score", "members"}
    assert only["min_score"] == only["max_score"] == 0.81
    assert [m["id"] for m in only["members"]] == ["note:default:b", "note:default:a"]
    assert only["members"][0] == {
        "id": "note:default:b",
        "kind": "work",
        "author": "claude-code",
        "saved": "2023-11-14",
        "occurred_at": None,
        "tags": ["rules"],
        "supersedes": "note:default:z",
        "text": "rule two",
    }
    assert only["members"][1]["saved"] == "2023-11-15"
    assert only["members"][1]["occurred_at"] == "1970-01-01T00:00:00+00:00"
    members = [
        note(
            "note:default:a",
            "rule one",
            saved=1_700_086_400.0,
            occurred_at=0.0,
            tags=("rules",),
        ),
        note(
            "note:default:b",
            "rule two",
            saved=1_700_000_000.0,
            tags=("rules",),
            supersedes="note:default:z",
        ),
    ]
    assert only["key"] == group_key("default", members)


def test_route_reports_deferred_notes(consolidator, fake_db):
    fake_db.note_rows = [
        _note_row("note:default:a", "a" * 400, 1.0),
        _note_row("note:default:b", "b" * 400, 2.0),
    ]
    fake_db.pair_rows = [{"a_id": "note:default:a", "b_id": "note:default:b", "score": 0.9}]
    response = client.get(
        "/admin/consolidate/groups", params={"namespace": "default", "max_group_chars": "500"}
    )
    section = response.json()["namespaces"]["default"]
    assert section["groups"] == []
    assert section["deferred"] == [
        {"id": "note:default:a", "reason": "over max_group_chars"},
        {"id": "note:default:b", "reason": "over max_group_chars"},
    ]


def test_route_counts_acknowledged_pairs_and_applies_the_limit(consolidator, fake_db):
    fake_db.note_rows = [
        _note_row("note:default:a", "a", 1.0),
        _note_row("note:default:b", "b", 2.0, similar_ack=["note:default:a"]),
        _note_row("note:default:c", "c", 3.0),
        _note_row("note:default:d", "d", 4.0),
        _note_row("note:default:e", "e", 5.0),
        _note_row("note:default:f", "f", 6.0),
    ]
    fake_db.pair_rows = [
        {"a_id": "note:default:a", "b_id": "note:default:b", "score": 0.95},
        {"a_id": "note:default:c", "b_id": "note:default:d", "score": 0.90},
        {"a_id": "note:default:e", "b_id": "note:default:f", "score": 0.80},
    ]
    limited = client.get(
        "/admin/consolidate/groups", params={"namespace": "default", "limit": "1"}
    ).json()["namespaces"]["default"]
    assert limited["acknowledged"] == 1
    assert [[m["id"] for m in g["members"]] for g in limited["groups"]] == [
        ["note:default:c", "note:default:d"]
    ]
    assert limited["truncated"] is True
    exact = client.get(
        "/admin/consolidate/groups", params={"namespace": "default", "limit": "2"}
    ).json()["namespaces"]["default"]
    assert len(exact["groups"]) == 2
    assert exact["truncated"] is False


def test_route_surfaces_a_duplicate_beside_an_acknowledged_pair(consolidator, fake_db):
    fake_db.note_rows = [
        _note_row("note:default:a", "a", 1.0, similar_ack=["note:default:b"]),
        _note_row("note:default:b", "b", 2.0),
        _note_row("note:default:c", "c", 3.0),
    ]
    fake_db.pair_rows = [
        {"a_id": "note:default:a", "b_id": "note:default:b", "score": 0.90},
        {"a_id": "note:default:a", "b_id": "note:default:c", "score": 0.80},
    ]
    section = client.get("/admin/consolidate/groups", params={"namespace": "default"}).json()[
        "namespaces"
    ]["default"]
    assert section["pairs"] == 2
    assert section["acknowledged"] == 1
    assert [[m["id"] for m in g["members"]] for g in section["groups"]] == [
        ["note:default:a", "note:default:c"]
    ]
    assert section["deferred"] == []


@pytest.mark.parametrize(
    "bound,value,param,rejected",
    [
        ("MAX_NEIGHBORS", 3, "neighbors", "4"),
        ("MAX_MAX_GROUP", 4, "max_group", "5"),
        ("MIN_MAX_GROUP_CHARS", 1000, "max_group_chars", "999"),
        ("MAX_LIMIT", 10, "limit", "11"),
    ],
)
def test_route_takes_its_bounds_from_the_module(
    monkeypatch, consolidator, fake_db, bound, value, param, rejected
):
    monkeypatch.setattr(groups, bound, value)
    response = client.get("/admin/consolidate/groups", params={param: rejected})
    assert response.status_code == 400
    assert str(value) in response.json()["error"]


def _three_pairs(fake_db):
    fake_db.note_rows = [
        _note_row("note:default:a", "a", 1.0),
        _note_row("note:default:b", "b", 2.0),
        _note_row("note:default:c", "c", 3.0),
        _note_row("note:default:d", "d", 4.0),
        _note_row("note:default:e", "e", 5.0),
        _note_row("note:default:f", "f", 6.0),
    ]
    fake_db.pair_rows = [
        {"a_id": "note:default:a", "b_id": "note:default:b", "score": 0.95},
        {"a_id": "note:default:c", "b_id": "note:default:d", "score": 0.90},
        {"a_id": "note:default:e", "b_id": "note:default:f", "score": 0.80},
    ]


def _section(**params):
    return client.get(
        "/admin/consolidate/groups", params={"namespace": "default", **params}
    ).json()["namespaces"]["default"]


def _member_ids(section):
    return [[m["id"] for m in g["members"]] for g in section["groups"]]


def test_route_skips_a_group_whose_key_has_a_recorded_verdict(consolidator, fake_db):
    _three_pairs(fake_db)
    first = _section()["groups"][0]
    fake_db.action_rows = [{"group_key": first["key"], "member_ids": ["x"], "undone": False}]
    section = _section(limit="1")
    assert section["cached"] == 1
    assert _member_ids(section) == [["note:default:c", "note:default:d"]]
    assert section["truncated"] is True
    actions_query, actions_args = next(
        f for f in fake_db.fetched if "consolidation_actions" in f[0]
    )
    assert actions_args == ("default",)
    assert fake_db.transactions[-1] == {"isolation": "repeatable_read", "readonly": True}


def test_route_skips_the_member_set_of_an_undone_action_across_procedure_versions(
    monkeypatch, consolidator, fake_db
):
    _three_pairs(fake_db)
    monkeypatch.setattr(groups, "PROCEDURE_VERSION", "2")
    fake_db.action_rows = [
        {"group_key": "an-old-key", "member_ids": ["note:default:d", "note:default:c"],
         "undone": True},
        {"group_key": "another-old-key", "member_ids": ["note:default:e", "note:default:f"],
         "undone": False},
    ]  # fmt: skip
    section = _section()
    assert section["cached"] == 1
    assert _member_ids(section) == [
        ["note:default:a", "note:default:b"],
        ["note:default:e", "note:default:f"],
    ]
