"""Unit tests for profile slots: the source hash, the work-rules rendering, and the routes.

No DB, no network: the routes read and write through a fake connection behind
``db.acquire`` that answers each statement by a marker in its SQL.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from contextlib import asynccontextmanager

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api, auth, namespaces, profiles
from memory_base.serve.consolidate import Note

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

NS = "default"
SAVED = 1_700_000_000.0


def note(note_id, text="text", **fields):
    values = {
        "id": note_id,
        "kind": "work",
        "author": "claude-code",
        "saved": SAVED,
        "occurred_at": None,
        "tags": ("rules",),
        "similar_ack": (),
        "supersedes": None,
        "text": text,
    }
    values.update(fields)
    return Note(**values)


# ---- source hash ------------------------------------------------------------


def test_source_hash_is_sha256_over_the_documented_canonical_json():
    notes = [
        note("B", "second", kind="personal", saved=0.0, tags=("z", "y"), author=None),
        note("A", "first", kind="personal", occurred_at=1_600_000_000.0),
    ]
    payload = {
        "v": "1",
        "slot": "user",
        "namespace": "ns",
        "notes": [
            [
                "A",
                hashlib.sha256(b"first").hexdigest(),
                "personal",
                "claude-code",
                SAVED,
                1_600_000_000.0,
                ["rules"],
            ],
            ["B", hashlib.sha256(b"second").hexdigest(), "personal", None, 0.0, None, ["y", "z"]],
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert profiles.PROFILE_VERSION == "1"
    assert (
        profiles.source_hash("ns", "user", notes) == hashlib.sha256(canonical.encode()).hexdigest()
    )


def test_source_hash_ignores_input_order():
    a, b = note("A"), note("B")
    assert profiles.source_hash(NS, "work-rules", [a, b]) == profiles.source_hash(
        NS, "work-rules", [b, a]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"text": "other text"},
        {"kind": "personal"},
        {"author": "natsume"},
        {"author": None},
        {"occurred_at": 0.0},
        {"occurred_at": 1.0},
        {"tags": ("rules", "delegation")},
        {"saved": SAVED + 1},
    ],
)
def test_source_hash_changes_with_every_field(change):
    base = [note("A"), note("B")]
    changed = [note("A"), dataclasses.replace(note("B"), **change)]
    assert profiles.source_hash(NS, "work-rules", base) != profiles.source_hash(
        NS, "work-rules", changed
    )


def test_source_hash_changes_when_a_note_is_added_or_removed():
    base = profiles.source_hash(NS, "work-rules", [note("A"), note("B")])
    assert profiles.source_hash(NS, "work-rules", [note("A"), note("B"), note("C")]) != base
    assert profiles.source_hash(NS, "work-rules", [note("A")]) != base


def test_source_hash_binds_the_slot_the_namespace_and_the_version(monkeypatch):
    notes = [note("A")]
    base = profiles.source_hash(NS, "work-rules", notes)
    assert profiles.source_hash(NS, "user", notes) != base
    assert profiles.source_hash("other", "work-rules", notes) != base
    monkeypatch.setattr(profiles, "PROFILE_VERSION", "2")
    assert profiles.source_hash(NS, "work-rules", notes) != base


# ---- work-rules rendering ---------------------------------------------------


def test_render_rules_is_byte_exact_and_keeps_the_submitted_order():
    notes = {
        "A": note("A", "Delegate coding to a worktree subagent."),
        "B": note("B", "Reviews:\n- run the tests\n- read the diff"),
        "C": note("C", "Ask before merging."),
    }
    assert profiles.render_rules(notes, ["C", "B", "A"]) == (
        "- Ask before merging.\n"
        "- Reviews:\n"
        "  - run the tests\n"
        "  - read the diff\n"
        "- Delegate coding to a worktree subagent."
    )


def test_render_rules_of_an_empty_selection_is_empty():
    assert profiles.render_rules({"A": note("A")}, []) == ""


# ---- fake database ----------------------------------------------------------


class FakeTransaction:
    def __init__(self, conn, options):
        self.conn = conn
        self.options = options

    async def __aenter__(self):
        self.conn.transactions.append(self.options)
        self.conn.depth += 1
        return self

    async def __aexit__(self, *exc):
        self.conn.depth -= 1
        return False


class FakeConnection:
    """Answers each profile statement by its SQL marker, emulating its filter and order."""

    def __init__(self, registered=("default", "work")):
        self.registered = set(registered)
        self.notes: list[dict] = []
        self.profiles: list[dict] = []
        self.transactions: list[dict] = []
        self.depth = 0
        self.calls: list[tuple[str, tuple]] = []

    def transaction(self, **options):
        return FakeTransaction(self, options)

    def _record(self, query, args):
        self.calls.append((query, args))

    async def execute(self, query, *args):
        self._record(query, args)
        assert self.depth, "statements run inside a transaction"
        return "SELECT 1"

    async def fetchval(self, query, *args):
        self._record(query, args)
        if "FOR SHARE" in query:
            assert self.depth, "the namespace is held inside the write transaction"
            return 1 if args[0] in self.registered else None
        if "INSERT INTO" in query:
            assert self.depth
            namespace, slot, version, content, source_ids, source_hash, author, model, created = (
                args
            )
            self.profiles.append(
                {
                    "namespace": namespace,
                    "slot": slot,
                    "version": version,
                    "content": content,
                    "source_ids": list(source_ids),
                    "source_hash": source_hash,
                    "author": author,
                    "model": model,
                    "created_at": created,
                }
            )
            return version
        raise AssertionError(f"unexpected fetchval: {query}")

    async def fetchrow(self, query, *args):
        self._record(query, args)
        namespace, slot = args
        rows = self._versions(namespace, slot)
        return rows[0] if rows else None

    async def fetch(self, query, *args):
        self._record(query, args)
        if "memory_chunks" in query:
            namespace, kind = args
            rows = [
                row
                for row in self.notes
                if row["namespace"] == namespace
                and row["kind"] == kind
                and row["archived_at"] is None
            ]
            return sorted(rows, key=lambda row: (row["ts_last_active"], row["id"]))
        if "DISTINCT ON" in query:
            (scope,) = args
            latest = {}
            for row in self.profiles:
                if scope is not None and row["namespace"] not in scope:
                    continue
                key = (row["namespace"], row["slot"])
                if key not in latest or row["version"] > latest[key]["version"]:
                    latest[key] = row
            return [latest[key] for key in sorted(latest)]
        namespace, slot, limit = args
        return self._versions(namespace, slot)[:limit]

    def _versions(self, namespace, slot):
        rows = [r for r in self.profiles if r["namespace"] == namespace and r["slot"] == slot]
        return sorted(rows, key=lambda r: -r["version"])

    def add_note(self, note_id, text, kind="work", namespace=NS, saved=SAVED, **metadata):
        self.notes.append(
            {
                "id": note_id,
                "namespace": namespace,
                "kind": kind,
                "ts_last_active": saved,
                "occurred_at": metadata.pop("occurred_at", None),
                "archived_at": metadata.pop("archived_at", None),
                "text": text,
                "metadata": json.dumps({"author": "claude-code", "tags": ["rules"], **metadata}),
            }
        )

    def writes(self):
        return [q for q, _ in self.calls if "INSERT INTO" in q]


@pytest.fixture()
def fake_db(monkeypatch):
    conn = FakeConnection()

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    async def noop(conn):
        return None

    async def fake_list_namespaces():
        return [{"name": n, "visibility": "public", "owner": None} for n in sorted(conn.registered)]

    monkeypatch.setattr(profiles.db, "acquire", acquire)
    monkeypatch.setattr(profiles, "ensure_schema_once", noop)
    monkeypatch.setattr(namespaces, "list_namespaces", fake_list_namespaces)
    return conn


def _use_identity(monkeypatch, is_admin=True, authors=("consolidator",), allowed=("default",)):
    identity = auth.KeyIdentity(
        key_id="consolidator-key-hash",
        label="consolidator",
        home="default",
        is_admin=is_admin,
        allowed=frozenset(allowed),
        authors=frozenset(authors),
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "test-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)


@pytest.fixture()
def consolidator(monkeypatch):
    _use_identity(monkeypatch)


def _sources(slot, namespace=NS):
    response = client.get("/admin/profiles/sources", params={"namespace": namespace, "slot": slot})
    assert response.status_code == 200, response.json()
    return response.json()


def _put(**body):
    return client.put("/admin/profiles", json=body)


def _user_body(fake_db, **overrides):
    body = {
        "namespace": NS,
        "slot": "user",
        "source_hash": _sources("user")["source_hash"],
        "author": "consolidator",
        "model": "test-model",
        "content": "The user lives in Seoul.",
    }
    body.update(overrides)
    return body


def _rules_body(fake_db, note_ids, **overrides):
    body = {
        "namespace": NS,
        "slot": "work-rules",
        "source_hash": _sources("work-rules")["source_hash"],
        "author": "consolidator",
        "model": None,
        "note_ids": note_ids,
    }
    body.update(overrides)
    return body


# ---- authorization ----------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/admin/profiles/sources?namespace=default&slot=user"),
        ("put", "/admin/profiles"),
        ("get", "/admin/profiles/versions?namespace=default&slot=user"),
    ],
)
@pytest.mark.parametrize("identity", [{"is_admin": False}, {"authors": ("claude-code",)}])
def test_admin_profile_routes_need_an_admin_key_with_the_consolidator_author(
    monkeypatch, fake_db, method, path, identity
):
    _use_identity(monkeypatch, **identity)
    response = getattr(client, method)(path, **({"json": {}} if method == "put" else {}))
    assert response.status_code == 403
    assert "consolidator" in response.json()["error"]
    assert fake_db.calls == []


# ---- sources ----------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"slot": "user"},
        {"namespace": NS},
        {"namespace": NS, "slot": "notes"},
        {"namespace": "nowhere", "slot": "user"},
        {"namespace": " ", "slot": "user"},
        [("namespace", NS), ("namespace", "work"), ("slot", "user")],
    ],
)
def test_sources_rejects_a_bad_parameter(consolidator, fake_db, params):
    response = client.get("/admin/profiles/sources", params=params)
    assert response.status_code == 400
    assert "error" in response.json()
    assert fake_db.calls == []


def test_sources_returns_the_eligible_notes_in_save_order_and_their_hash(consolidator, fake_db):
    fake_db.add_note("note:default:b", "Lives in Seoul.", kind="personal", saved=SAVED + 10)
    fake_db.add_note(
        "note:default:a", "Vegetarian.", kind="personal", saved=SAVED + 10, occurred_at=0.0
    )
    fake_db.add_note("note:default:c", "Born 1990.", kind="personal", saved=SAVED)
    fake_db.add_note("note:default:w", "Review every PR.", kind="work")
    fake_db.add_note("note:default:x", "Old fact.", kind="personal", archived_at=SAVED)
    fake_db.add_note("note:work:p", "Other namespace.", kind="personal", namespace="work")
    body = _sources("user")
    assert body["namespace"] == NS
    assert body["slot"] == "user"
    assert body["profile_version"] == "1"
    assert body["current"] is None
    assert body["stale"] is True
    assert [n["id"] for n in body["notes"]] == [
        "note:default:c",
        "note:default:a",
        "note:default:b",
    ]
    assert body["notes"][1] == {
        "id": "note:default:a",
        "kind": "personal",
        "author": "claude-code",
        "saved": "2023-11-14T22:13:30+00:00",
        "occurred_at": "1970-01-01T00:00:00+00:00",
        "tags": ["rules"],
        "text": "Vegetarian.",
    }
    eligible = [
        note("note:default:c", "Born 1990.", kind="personal"),
        note("note:default:a", "Vegetarian.", kind="personal", saved=SAVED + 10, occurred_at=0.0),
        note("note:default:b", "Lives in Seoul.", kind="personal", saved=SAVED + 10),
    ]
    assert body["source_hash"] == profiles.source_hash(NS, "user", eligible)


def test_sources_reads_one_snapshot(consolidator, fake_db):
    _sources("user")
    assert fake_db.transactions == [{"isolation": "repeatable_read", "readonly": True}]


def test_sources_hash_ignores_another_namespace_and_another_kind(consolidator, fake_db):
    fake_db.add_note("note:default:a", "Lives in Seoul.", kind="personal")
    before = _sources("user")["source_hash"]
    fake_db.add_note("note:default:w", "Review every PR.", kind="work")
    fake_db.add_note("note:work:p", "Lives in Busan.", kind="personal", namespace="work")
    assert _sources("user")["source_hash"] == before


def test_sources_hash_changes_when_a_note_is_archived(consolidator, fake_db):
    fake_db.add_note("note:default:a", "Lives in Seoul.", kind="personal")
    fake_db.add_note("note:default:b", "Vegetarian.", kind="personal")
    before = _sources("user")["source_hash"]
    fake_db.notes[1]["archived_at"] = SAVED
    assert _sources("user")["source_hash"] != before


def test_sources_reports_the_current_version_and_whether_it_is_stale(consolidator, fake_db):
    fake_db.add_note("note:default:a", "Lives in Seoul.", kind="personal")
    assert _put(**_user_body(fake_db)).json()["status"] == "written"
    body = _sources("user")
    assert body["stale"] is False
    assert body["current"]["version"] == 1
    assert body["current"]["source_hash"] == body["source_hash"]
    assert set(body["current"]) == {"version", "source_hash", "created_at"}
    fake_db.add_note("note:default:b", "Vegetarian.", kind="personal")
    after = _sources("user")
    assert after["stale"] is True
    assert after["current"]["version"] == 1


# ---- write: schema ----------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"extra": 1},
        {"slot": "notes"},
        {"namespace": ""},
        {"source_hash": ""},
        {"author": ""},
        {"model": 3},
        {"dry_run": "yes"},
        {"content": 5},
        {"note_ids": ["note:default:a"]},
        {"max_chars": 199},
        {"max_chars": 20001},
        {"max_chars": True},
        {"max_chars": 500.0},
    ],
)
def test_write_user_refuses_a_schema_violation(consolidator, fake_db, overrides):
    body = _user_body(fake_db, **overrides)
    fake_db.calls.clear()
    response = _put(**body)
    assert response.status_code == 400
    assert "error" in response.json()
    assert fake_db.calls == []


def test_write_user_requires_content(consolidator, fake_db):
    body = _user_body(fake_db)
    del body["content"]
    assert _put(**body).status_code == 400


@pytest.mark.parametrize(
    "overrides",
    [
        {"content": "- a rule"},
        {"note_ids": "note:default:a"},
        {"note_ids": [1]},
        {"note_ids": ["note:default:a", "note:default:a"]},
        {"note_ids": [f"note:default:{i}" for i in range(201)]},
    ],
)
def test_write_rules_refuses_a_schema_violation(consolidator, fake_db, overrides):
    body = {**_rules_body(fake_db, []), **overrides}
    fake_db.calls.clear()
    response = _put(**body)
    assert response.status_code == 400
    assert fake_db.calls == []


def test_write_rules_requires_note_ids(consolidator, fake_db):
    body = _rules_body(fake_db, [])
    del body["note_ids"]
    assert _put(**body).status_code == 400


def test_write_refuses_an_author_the_key_does_not_hold(consolidator, fake_db):
    response = _put(**_user_body(fake_db, author="claude-code"))
    assert response.status_code == 403
    assert fake_db.writes() == []


def test_write_refuses_an_unregistered_namespace(consolidator, fake_db):
    response = _put(**_user_body(fake_db, namespace="nowhere"))
    assert response.status_code == 400
    assert "unregistered" in response.json()["error"]
    assert fake_db.writes() == []


# ---- write: flow ------------------------------------------------------------


def test_write_runs_in_one_transaction_under_the_slot_lock(consolidator, fake_db):
    fake_db.add_note("note:default:a", "Lives in Seoul.", kind="personal")
    body = _user_body(fake_db)
    fake_db.calls.clear()
    fake_db.transactions.clear()
    assert _put(**body).json()["status"] == "written"
    assert fake_db.transactions == [{}]
    lock, lock_args = fake_db.calls[0]
    assert "pg_advisory_xact_lock(hashtextextended('profile:' ||" in lock
    assert lock_args == (NS, "user")
    assert "FOR SHARE" in fake_db.calls[1][0]


def test_a_stale_hash_is_409_with_the_current_hash_and_writes_nothing(consolidator, fake_db):
    fake_db.add_note("note:default:a", "Lives in Seoul.", kind="personal")
    body = _user_body(fake_db)
    fake_db.add_note("note:default:b", "Vegetarian.", kind="personal")
    response = _put(**body)
    assert response.status_code == 409
    assert response.json() == {"error": "stale", "source_hash": _sources("user")["source_hash"]}
    assert fake_db.profiles == []


def test_write_user_stores_the_stripped_content_with_every_eligible_note(consolidator, fake_db):
    fake_db.add_note("note:default:b", "Vegetarian.", kind="personal", saved=SAVED + 1)
    fake_db.add_note("note:default:a", "Lives in Seoul.", kind="personal", saved=SAVED + 2)
    fake_db.add_note("note:default:w", "Review every PR.", kind="work")
    response = _put(**_user_body(fake_db, content="  Lives in Seoul; vegetarian.\n"))
    assert response.status_code == 200
    assert response.json() == {"status": "written", "version": 1, "chars": 27}
    [row] = fake_db.profiles
    assert row["content"] == "Lives in Seoul; vegetarian."
    assert row["source_ids"] == ["note:default:b", "note:default:a"]
    assert row["source_hash"] == _sources("user")["source_hash"]
    assert row["author"] == "consolidator"
    assert row["model"] == "test-model"


@pytest.mark.parametrize("content", ["", "   \n\t"])
def test_write_user_refuses_blank_content_while_it_has_sources(consolidator, fake_db, content):
    fake_db.add_note("note:default:a", "Lives in Seoul.", kind="personal")
    response = _put(**_user_body(fake_db, content=content))
    assert response.status_code == 400
    assert fake_db.profiles == []


@pytest.mark.parametrize("content", ["", "   \n\t"])
def test_write_user_without_sources_stores_blank_content_as_empty(consolidator, fake_db, content):
    response = _put(**_user_body(fake_db, content=content))
    assert response.status_code == 200
    assert response.json() == {"status": "written", "version": 1, "chars": 0}
    [row] = fake_db.profiles
    assert row["content"] == ""
    assert row["source_ids"] == []


def test_a_user_profile_is_cleared_when_every_source_is_archived(consolidator, fake_db):
    fake_db.add_note("note:default:a", "Lives in Seoul.", kind="personal")
    assert _put(**_user_body(fake_db)).json()["status"] == "written"
    assert [p["slot"] for p in client.get("/profiles").json()] == ["user"]
    fake_db.notes[0]["archived_at"] = SAVED
    sources = _sources("user")
    assert sources["notes"] == []
    assert sources["stale"] is True
    cleared = _put(**_user_body(fake_db, content=""))
    assert cleared.json() == {"status": "written", "version": 2, "chars": 0}
    assert client.get("/profiles").json() == []
    assert _sources("user")["stale"] is False
    again = _put(**_user_body(fake_db, content=""))
    assert again.json() == {"status": "unchanged", "version": 2}
    assert len(fake_db.profiles) == 2


def test_write_user_refuses_content_over_the_budget(consolidator, fake_db):
    over = _put(**_user_body(fake_db, content="x" * 1501))
    assert over.status_code == 400
    assert "1501" in over.json()["error"]
    assert _put(**_user_body(fake_db, content="x" * 1500)).json()["status"] == "written"
    assert _put(**_user_body(fake_db, content="y" * 300, max_chars=299)).status_code == 400
    assert len(fake_db.profiles) == 1


def test_write_user_refuses_a_credential(consolidator, fake_db):
    token = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
    response = _put(**_user_body(fake_db, content=f"The user's token is {token}."))
    assert response.status_code == 400
    assert "credential" in response.json()["error"]
    assert token not in response.json()["error"]
    assert fake_db.profiles == []


def _seed_rules(fake_db):
    fake_db.add_note("note:default:a", "Delegate coding to a worktree subagent.", saved=SAVED + 1)
    fake_db.add_note("note:default:b", "Reviews:\n- run the tests\n- read the diff", saved=SAVED)
    fake_db.add_note("note:default:c", "Ask before merging.", saved=SAVED + 2)
    fake_db.add_note("note:default:p", "Lives in Seoul.", kind="personal")
    fake_db.add_note("note:default:z", "Archived rule.", archived_at=SAVED)


def test_write_rules_renders_the_selected_notes_verbatim_in_the_submitted_order(
    consolidator, fake_db
):
    _seed_rules(fake_db)
    ids = ["note:default:c", "note:default:b", "note:default:a"]
    response = _put(**_rules_body(fake_db, ids))
    expected = (
        "- Ask before merging.\n"
        "- Reviews:\n"
        "  - run the tests\n"
        "  - read the diff\n"
        "- Delegate coding to a worktree subagent."
    )
    assert response.json() == {"status": "written", "version": 1, "chars": len(expected)}
    [row] = fake_db.profiles
    assert row["content"] == expected
    assert row["source_ids"] == ids
    assert row["model"] is None


@pytest.mark.parametrize("outside", ["note:default:p", "note:default:z", "note:default:nope"])
def test_write_rules_refuses_an_id_outside_the_eligible_set(consolidator, fake_db, outside):
    _seed_rules(fake_db)
    response = _put(**_rules_body(fake_db, ["note:default:a", outside]))
    assert response.status_code == 400
    assert outside in response.json()["error"]
    assert "note:default:a" not in response.json()["error"]
    assert fake_db.profiles == []


def test_write_rules_over_the_budget_is_refused_with_the_rendered_length(consolidator, fake_db):
    fake_db.add_note("note:default:a", "r" * 150)
    fake_db.add_note("note:default:b", "s" * 150)
    response = _put(**_rules_body(fake_db, ["note:default:a", "note:default:b"], max_chars=300))
    assert response.status_code == 400
    assert response.json()["chars"] == 305
    assert "305" in response.json()["error"]
    assert fake_db.profiles == []


def test_write_rules_with_an_empty_selection_stores_empty_content(consolidator, fake_db):
    _seed_rules(fake_db)
    response = _put(**_rules_body(fake_db, []))
    assert response.json() == {"status": "written", "version": 1, "chars": 0}
    assert fake_db.profiles[0]["content"] == ""
    assert fake_db.profiles[0]["source_ids"] == []


def test_an_unchanged_write_adds_no_version(consolidator, fake_db):
    _seed_rules(fake_db)
    assert _put(**_rules_body(fake_db, ["note:default:a"])).json()["version"] == 1
    again = _put(**_rules_body(fake_db, ["note:default:a"]))
    assert again.json() == {"status": "unchanged", "version": 1}
    planned = _put(**_rules_body(fake_db, ["note:default:a"], dry_run=True))
    assert planned.json() == {"status": "unchanged", "version": 1}
    assert len(fake_db.profiles) == 1


def test_a_changed_selection_writes_the_next_version(consolidator, fake_db):
    _seed_rules(fake_db)
    _put(**_rules_body(fake_db, ["note:default:a"]))
    response = _put(**_rules_body(fake_db, ["note:default:a", "note:default:c"]))
    assert response.json()["version"] == 2
    assert [r["version"] for r in fake_db.profiles] == [1, 2]


def test_a_dry_run_plans_without_writing(consolidator, fake_db):
    _seed_rules(fake_db)
    response = _put(**_rules_body(fake_db, ["note:default:c"], dry_run=True))
    assert response.status_code == 200
    assert response.json() == {
        "status": "planned",
        "content": "- Ask before merging.",
        "chars": 21,
    }
    assert fake_db.profiles == []
    assert fake_db.writes() == []


# ---- read -------------------------------------------------------------------


def _profile(namespace, slot, version, content):
    return {
        "namespace": namespace,
        "slot": slot,
        "version": version,
        "content": content,
        "source_ids": [],
        "source_hash": "h",
        "author": "consolidator",
        "model": None,
        "created_at": SAVED + version,
    }


def _seed_profiles(fake_db):
    fake_db.profiles = [
        _profile("work", "work-rules", 1, "- old rule"),
        _profile("work", "work-rules", 2, "- new rule"),
        _profile("default", "work-rules", 1, "- review every PR"),
        _profile("default", "user", 1, "Lives in Seoul."),
        _profile("personal", "user", 1, "Private fact."),
        _profile("emptied", "work-rules", 1, "- a rule"),
        _profile("emptied", "work-rules", 2, ""),
    ]


def test_profiles_returns_the_latest_non_empty_version_per_slot_in_order(fake_db):
    _seed_profiles(fake_db)
    response = client.get("/profiles")
    assert response.status_code == 200
    assert response.json() == [
        {
            "namespace": "default",
            "slot": "user",
            "version": 1,
            "content": "Lives in Seoul.",
            "created_at": "2023-11-14T22:13:21+00:00",
        },
        {
            "namespace": "default",
            "slot": "work-rules",
            "version": 1,
            "content": "- review every PR",
            "created_at": "2023-11-14T22:13:21+00:00",
        },
        {
            "namespace": "personal",
            "slot": "user",
            "version": 1,
            "content": "Private fact.",
            "created_at": "2023-11-14T22:13:21+00:00",
        },
        {
            "namespace": "work",
            "slot": "work-rules",
            "version": 2,
            "content": "- new rule",
            "created_at": "2023-11-14T22:13:22+00:00",
        },
    ]


def test_profiles_filters_by_repeated_namespace(fake_db):
    _seed_profiles(fake_db)
    response = client.get("/profiles", params=[("namespace", "work"), ("namespace", "default")])
    assert [(p["namespace"], p["slot"]) for p in response.json()] == [
        ("default", "user"),
        ("default", "work-rules"),
        ("work", "work-rules"),
    ]


def test_profiles_default_to_the_namespaces_a_member_key_can_read(monkeypatch, fake_db):
    _use_identity(monkeypatch, is_admin=False, authors=(), allowed=("default", "work"))
    _seed_profiles(fake_db)
    response = client.get("/profiles")
    assert response.status_code == 200
    assert {p["namespace"] for p in response.json()} == {"default", "work"}


def test_profiles_refuse_a_namespace_the_key_cannot_read(monkeypatch, fake_db):
    _use_identity(monkeypatch, is_admin=False, authors=(), allowed=("default",))
    response = client.get("/profiles", params=[("namespace", "default"), ("namespace", "personal")])
    assert response.status_code == 403
    assert fake_db.calls == []


# ---- versions ---------------------------------------------------------------


def test_versions_lists_every_version_newest_first(consolidator, fake_db):
    _seed_rules(fake_db)
    _put(**_rules_body(fake_db, ["note:default:a"], model="m1"))
    _put(**_rules_body(fake_db, ["note:default:c", "note:default:a"], model="m2"))
    response = client.get(
        "/admin/profiles/versions", params={"namespace": NS, "slot": "work-rules"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["namespace"] == NS
    assert body["slot"] == "work-rules"
    assert [v["version"] for v in body["versions"]] == [2, 1]
    newest = body["versions"][0]
    assert set(newest) == {
        "version",
        "content",
        "source_ids",
        "source_hash",
        "author",
        "model",
        "created_at",
    }
    assert newest["source_ids"] == ["note:default:c", "note:default:a"]
    assert newest["model"] == "m2"
    assert newest["content"] == ("- Ask before merging.\n- Delegate coding to a worktree subagent.")
    _, args = fake_db.calls[-1]
    assert args == (NS, "work-rules", 20)


@pytest.mark.parametrize(
    "params",
    [
        {"namespace": NS, "slot": "user", "limit": "0"},
        {"namespace": NS, "slot": "user", "limit": "201"},
        {"namespace": NS, "slot": "user", "limit": "many"},
        {"namespace": NS},
        {"slot": "user"},
        {"namespace": NS, "slot": "rules"},
        {"namespace": "nowhere", "slot": "user"},
    ],
)
def test_versions_rejects_a_bad_parameter(consolidator, fake_db, params):
    response = client.get("/admin/profiles/versions", params=params)
    assert response.status_code == 400
    assert fake_db.calls == []


def test_versions_takes_the_limit(consolidator, fake_db):
    client.get("/admin/profiles/versions", params={"namespace": NS, "slot": "user", "limit": "200"})
    _, args = fake_db.calls[-1]
    assert args == (NS, "user", 200)
