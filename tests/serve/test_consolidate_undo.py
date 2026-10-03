"""Unit tests for POST /admin/consolidate/undo and GET /admin/consolidate/actions.

No DB, no network: undo runs on a fake connection that answers its reads, its refusal
rules are a pure function, and the routes delegate to stubbed module functions.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api, auth, verdicts

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

NS = "work"
A, B, C = "note:work:a", "note:work:b", "note:work:c"
R, S = "note:work:r", "note:work:s"
APPLIED_AT = 1_759_000_000.5
PRIOR = {
    A: {"tags": ["deploy"], "author": "claude-code", "similar_ack": [B]},
    B: {"tags": ["deploy", "ops"], "author": "hermes"},
    C: {"tags": ["deploy"], "author": "claude-code", "supersedes": "note:work:old"},
}


def action_row(**over):
    row = {
        "id": 5,
        "namespace": NS,
        "action": "merge",
        "member_ids": [A, B, C],
        "archived_ids": [A, B, C],
        "survivor_ids": [R],
        "replacement_id": R,
        "replacement_created": True,
        "prior": json.dumps(PRIOR),
        "applied_at": APPLIED_AT,
        "undone_at": None,
        "undone_by": None,
        "undo_result": None,
    }
    row.update(over)
    return row


def archived(note_id, into=(R,), at=APPLIED_AT):
    metadata = {**PRIOR[note_id], "archived_by": "consolidator", "consolidated_into": list(into)}
    return {"id": note_id, "archived_at": at, "metadata": json.dumps(metadata)}


def replacement(at=None):
    return {"id": R, "archived_at": at, "metadata": json.dumps({"merged_from": [A, B, C]})}


def rows(*entries):
    return {entry["id"]: entry for entry in entries}


# ---- refusal rules ----------------------------------------------------------


def test_an_untouched_merge_can_be_undone():
    found = rows(archived(A), archived(B), archived(C), replacement())
    assert verdicts.undo_refusal(action_row(), found, None, False) is None


@pytest.mark.parametrize(
    "found,needle",
    [
        (rows(archived(B), archived(C), replacement()), A),
        (rows(archived(A, at=APPLIED_AT + 1), archived(B), archived(C), replacement()), A),
        (rows(archived(A), archived(B, at=None), archived(C), replacement()), B),
        (rows(archived(A), archived(B), archived(C, into=()), replacement()), C),
        (rows(archived(A), archived(B), archived(C)), R),
        (rows(archived(A), archived(B), archived(C), replacement(at=APPLIED_AT + 5)), R),
    ],
)
def test_a_changed_note_refuses_the_undo(found, needle):
    reason = verdicts.undo_refusal(action_row(), found, None, False)
    assert reason is not None and needle in reason


def test_a_metadata_without_consolidated_into_refuses_the_undo():
    plain = {"id": A, "archived_at": APPLIED_AT, "metadata": json.dumps(PRIOR[A])}
    found = rows(plain, archived(B), archived(C), replacement())
    assert A in verdicts.undo_refusal(action_row(), found, None, False)


def test_an_active_descendant_of_the_replacement_refuses_the_undo():
    found = rows(archived(A), archived(B), archived(C), replacement())
    assert S in verdicts.undo_refusal(action_row(), found, S, False)


def test_a_later_action_on_the_replacement_refuses_the_undo():
    found = rows(archived(A), archived(B), archived(C), replacement())
    assert R in verdicts.undo_refusal(action_row(), found, None, True)


def test_only_a_later_retire_or_merge_blocks_the_undo():
    query = " ".join(verdicts.LATER_ACTION_SQL.split())
    assert "action IN ('retire', 'merge')" in query
    assert "undone_at IS NULL" in query
    assert "$1 = ANY(member_ids)" in query


def test_a_reused_replacement_and_retire_survivors_are_not_checked():
    reused = action_row(replacement_created=False)
    assert (
        verdicts.undo_refusal(reused, rows(archived(A), archived(B), archived(C)), None, True)
        is None
    )
    retire = action_row(
        action="retire", archived_ids=[A], survivor_ids=[B, C], replacement_id=None,
        replacement_created=False,
    )  # fmt: skip
    assert verdicts.undo_refusal(retire, rows(archived(A, into=(B, C))), None, False) is None


# ---- undo on a fake connection ----------------------------------------------


class _Tx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.conn.calls.append(("begin", "", ()))

    async def __aexit__(self, exc_type, exc, tb):
        self.conn.calls.append(("end", "", ()))
        return False


class UndoConn:
    def __init__(self, action, found, descendant=None, later=False):
        self.action = action
        self.found = found
        self.descendant = descendant
        self.later = later
        self.calls: list[tuple[str, str, tuple]] = []

    def transaction(self):
        return _Tx(self)

    def _record(self, kind, query, args):
        self.calls.append((kind, " ".join(query.split()), args))

    async def fetchval(self, query, *args):
        self._record("fetchval", query, args)
        if "RECURSIVE" in query:
            return self.descendant
        if "ANY(member_ids)" in query:
            return self.later
        if "SELECT namespace" in query:
            return None if self.action is None else self.action["namespace"]
        raise AssertionError(query)

    async def fetchrow(self, query, *args):
        self._record("fetchrow", query, args)
        assert "FOR UPDATE" in query
        return self.action

    async def fetch(self, query, *args):
        self._record("fetch", query, args)
        assert "FOR UPDATE" in query and "ORDER BY id" in query
        return [self.found[i] for i in sorted(args[0]) if i in self.found]

    async def execute(self, query, *args):
        self._record("execute", query, args)
        return "UPDATE 1"

    def writes(self):
        return [c for c in self.calls if c[0] == "execute" and "UPDATE" in c[1]]


def _undo(monkeypatch, conn, author="consolidator"):
    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    async def noop(conn):
        return None

    monkeypatch.setattr(verdicts.db, "acquire", acquire)
    monkeypatch.setattr(verdicts, "ensure_schema_once", noop)
    return asyncio.run(verdicts.undo(5, author))


def test_undo_restores_prior_metadata_exactly_and_archives_the_created_replacement(monkeypatch):
    conn = UndoConn(action_row(), rows(archived(A), archived(B), archived(C), replacement()))
    result = _undo(monkeypatch, conn)
    assert result["action_id"] == 5
    assert result["restored_ids"] == [A, B, C]
    assert result["archived_ids"] == [R]
    assert result["undone_by"] == "consolidator"
    assert result["undone_at"].endswith("+00:00")
    lock = next(c for c in conn.calls if c[0] == "execute")
    assert "pg_advisory_xact_lock" in lock[1] and lock[2] == (NS,)
    [locked] = [c for c in conn.calls if c[0] == "fetch"]
    assert sorted(locked[2][0]) == [A, B, C, R]
    restores = [c for c in conn.writes() if "archived_at = NULL" in c[1]]
    assert {c[2][0]: json.loads(c[2][1]) for c in restores} == PRIOR
    [archive] = [c for c in conn.writes() if "'undone_action'" in c[1]]
    assert archive[2][0] == R and "consolidator" in archive[2] and 5 in archive[2]
    [record] = [c for c in conn.writes() if "consolidation_actions" in c[1]]
    assert "undone_at" in record[1] and "undo_result" in record[1]
    assert json.loads(next(a for a in record[2] if isinstance(a, str) and a.startswith("{"))) == (
        result
    )


def test_undo_of_a_retire_restores_only_what_it_archived(monkeypatch):
    retire = action_row(
        action="retire", archived_ids=[A], survivor_ids=[B, C], replacement_id=None,
        replacement_created=False, prior=json.dumps({A: PRIOR[A]}),
    )  # fmt: skip
    conn = UndoConn(retire, rows(archived(A, into=(B, C))))
    result = _undo(monkeypatch, conn)
    assert result["restored_ids"] == [A]
    assert result["archived_ids"] == []
    [locked] = [c for c in conn.calls if c[0] == "fetch"]
    assert locked[2][0] == [A]
    assert [c[2][0] for c in conn.writes() if "memory_chunks" in c[1]] == [A]
    assert not [c for c in conn.calls if "RECURSIVE" in c[1] or "ANY(member_ids)" in c[1]]


def test_undo_never_touches_a_reused_replacement(monkeypatch):
    conn = UndoConn(
        action_row(replacement_created=False), rows(archived(A), archived(B), archived(C))
    )
    result = _undo(monkeypatch, conn)
    assert result["archived_ids"] == []
    assert R not in [c[2][0] for c in conn.writes() if "memory_chunks" in c[1]]


def test_an_undone_action_returns_its_recorded_result(monkeypatch):
    recorded = {"action_id": 5, "restored_ids": [A], "archived_ids": [R]}
    conn = UndoConn(action_row(undone_at=1.0, undo_result=json.dumps(recorded)), {})
    assert _undo(monkeypatch, conn) == recorded
    assert conn.writes() == []


def test_a_keep_has_nothing_to_undo(monkeypatch):
    keep = action_row(action="keep", archived_ids=[], survivor_ids=[A, B, C], replacement_id=None)
    with pytest.raises(verdicts.UndoRefused, match="nothing to undo"):
        _undo(monkeypatch, UndoConn(keep, {}))


def test_an_unknown_action_is_not_found(monkeypatch):
    with pytest.raises(verdicts.UndoNotFound):
        _undo(monkeypatch, UndoConn(None, {}))


def test_a_refused_undo_changes_nothing(monkeypatch):
    found = rows(archived(A, at=APPLIED_AT + 1), archived(B), archived(C), replacement())
    conn = UndoConn(action_row(), found)
    with pytest.raises(verdicts.UndoRefused, match=A):
        _undo(monkeypatch, conn)
    assert conn.writes() == []


def test_an_active_descendant_refuses_through_the_recursive_query(monkeypatch):
    conn = UndoConn(
        action_row(), rows(archived(A), archived(B), archived(C), replacement()), descendant=S
    )
    with pytest.raises(verdicts.UndoRefused, match=S):
        _undo(monkeypatch, conn)
    [walk] = [c for c in conn.calls if "RECURSIVE" in c[1]]
    assert walk[2][0] == R
    assert "supersedes" in walk[1] and "archived_at IS NULL" in walk[1]
    assert conn.writes() == []


# ---- routes -----------------------------------------------------------------


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
def undo_calls(monkeypatch):
    calls = []
    outcome = {"raise": None, "result": {"action_id": 5}}

    async def fake_undo(action_id, author):
        calls.append((action_id, author))
        if outcome["raise"] is not None:
            raise outcome["raise"]
        return outcome["result"]

    monkeypatch.setattr(verdicts, "undo", fake_undo)
    return calls, outcome


def test_undo_route_refuses_a_non_admin_key(monkeypatch, undo_calls):
    _use_identity(monkeypatch, is_admin=False)
    response = client.post(
        "/admin/consolidate/undo", json={"action_id": 5, "author": "consolidator"}
    )
    assert response.status_code == 403
    assert undo_calls[0] == []


def test_undo_route_refuses_an_admin_key_without_the_consolidator_author(undo_calls):
    response = client.post("/admin/consolidate/undo", json={"action_id": 5, "author": "natsume"})
    assert response.status_code == 403
    assert undo_calls[0] == []


def test_undo_route_refuses_an_author_outside_the_keys_authors(monkeypatch, undo_calls):
    _use_identity(monkeypatch)
    response = client.post("/admin/consolidate/undo", json={"action_id": 5, "author": "natsume"})
    assert response.status_code == 403
    assert undo_calls[0] == []


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"author": "consolidator"},
        {"action_id": 5},
        {"action_id": "5", "author": "consolidator"},
        {"action_id": True, "author": "consolidator"},
        {"action_id": 0, "author": "consolidator"},
        {"action_id": 5, "author": ""},
        {"action_id": 5, "author": "consolidator", "confirm": True},
    ],
)
def test_undo_route_rejects_a_bad_body(monkeypatch, undo_calls, raw):
    _use_identity(monkeypatch)
    response = client.post("/admin/consolidate/undo", json=raw)
    assert response.status_code == 400
    assert undo_calls[0] == []


@pytest.mark.parametrize(
    "error,status",
    [(verdicts.UndoNotFound("no action 5"), 404), (verdicts.UndoRefused("nothing to undo"), 409)],
)
def test_undo_route_maps_errors(monkeypatch, undo_calls, error, status):
    _use_identity(monkeypatch)
    undo_calls[1]["raise"] = error
    response = client.post(
        "/admin/consolidate/undo", json={"action_id": 5, "author": "consolidator"}
    )
    assert response.status_code == status
    assert response.json() == {"error": str(error)}


def test_undo_route_returns_the_undo_result(monkeypatch, undo_calls):
    _use_identity(monkeypatch)
    response = client.post(
        "/admin/consolidate/undo", json={"action_id": 5, "author": "consolidator"}
    )
    assert response.status_code == 200
    assert response.json() == {"action_id": 5}
    assert undo_calls[0] == [(5, "consolidator")]


@pytest.fixture()
def listing(monkeypatch):
    calls = []

    async def fake_list_actions(**kwargs):
        calls.append(kwargs)
        return {"actions": [], "notes": {}}

    monkeypatch.setattr(verdicts, "list_actions", fake_list_actions)
    return calls


def test_actions_route_refuses_without_the_consolidator_author(listing):
    assert client.get("/admin/consolidate/actions").status_code == 403
    assert listing == []


def test_actions_route_defaults(monkeypatch, listing):
    _use_identity(monkeypatch)
    response = client.get("/admin/consolidate/actions")
    assert response.status_code == 200
    assert response.json() == {"actions": [], "notes": {}}
    assert listing == [{"namespace": None, "run_id": None, "note_id": None, "limit": 50}]


def test_actions_route_forwards_filters(monkeypatch, listing):
    _use_identity(monkeypatch)
    response = client.get(
        "/admin/consolidate/actions",
        params={"namespace": NS, "run_id": "run-1", "note_id": A, "limit": "7"},
    )
    assert response.status_code == 200
    assert listing == [{"namespace": NS, "run_id": "run-1", "note_id": A, "limit": 7}]


@pytest.mark.parametrize(
    "params",
    [
        {"limit": "0"},
        {"limit": "501"},
        {"limit": "many"},
        {"namespace": " "},
        {"run_id": ""},
        {"note_id": " "},
        [("run_id", "a"), ("run_id", "b")],
    ],
)
def test_actions_route_rejects_a_bad_parameter(monkeypatch, listing, params):
    _use_identity(monkeypatch)
    response = client.get("/admin/consolidate/actions", params=params)
    assert response.status_code == 400
    assert listing == []


class ListingConn:
    def __init__(self, action_rows, note_rows):
        self.action_rows = action_rows
        self.note_rows = note_rows
        self.calls: list[tuple[str, tuple]] = []

    async def fetch(self, query, *args):
        self.calls.append((" ".join(query.split()), args))
        return self.action_rows if "consolidation_actions" in query else self.note_rows


def test_list_actions_returns_actions_newest_first_with_the_notes_they_reference(monkeypatch):
    stored = action_row(
        payload_hash="f" * 64, idempotency_key="k-1", run_id="run-1", group_key="g" * 64,
        author="consolidator", model="m", reason="same rule",
        result=json.dumps({"status": "applied"}),
    )  # fmt: skip
    note_rows = [
        {
            "id": A,
            "kind": "work",
            "text": "x" * 3000,
            "metadata": json.dumps(
                {
                    "author": "claude-code",
                    "tags": ["deploy"],
                    "consolidated_into": [R],
                    "archived_by": "consolidator",
                    "similar_ack": [B],
                }
            ),  # fmt: skip
            "ts_last_active": 1_700_000_000.0,
            "occurred_at": None,
            "archived_at": APPLIED_AT,
        },
        {
            "id": R,
            "kind": "work",
            "text": "merged",
            "metadata": json.dumps(
                {
                    "author": "consolidator",
                    "tags": ["deploy"],
                    "merged_from": [A, B, C],
                    "consolidation_action": 5,
                }
            ),  # fmt: skip
            "ts_last_active": APPLIED_AT,
            "occurred_at": 1_690_000_000.0,
            "archived_at": None,
        },
    ]
    conn = ListingConn([stored], note_rows)

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    monkeypatch.setattr(verdicts.db, "acquire", acquire)
    out = asyncio.run(verdicts.list_actions(namespace=NS, run_id="run-1", note_id=A, limit=10))

    query, args = conn.calls[0]
    assert "ORDER BY id DESC" in query
    assert args == (NS, "run-1", A, 10)
    for column in ("member_ids", "archived_ids", "survivor_ids", "replacement_id"):
        assert column in query
    _, note_args = conn.calls[1]
    assert sorted(note_args[0]) == sorted({A, B, C, R})

    [action] = out["actions"]
    assert "payload_hash" not in action
    assert action["id"] == 5
    assert action["applied_at"] == "2025-09-27T19:06:40.500000+00:00"
    assert action["undone_at"] is None
    assert action["result"] == {"status": "applied"}
    assert action["prior"] == PRIOR
    assert set(out["notes"]) == {A, R}
    assert out["notes"][A] == {
        "kind": "work",
        "author": "claude-code",
        "text": "x" * 3000,
        "saved": "2023-11-14T22:13:20+00:00",
        "occurred_at": None,
        "archived": True,
        "consolidated_into": [R],
        "archived_by": "consolidator",
    }
    assert out["notes"][R]["merged_from"] == [A, B, C]
    assert out["notes"][R]["consolidation_action"] == 5
    assert out["notes"][R]["archived"] is False
    assert out["notes"][R]["occurred_at"] == "2023-07-22T04:26:40+00:00"
