"""Unit tests for agent-owned profiles: request validation, authority, and the routes.

No DB, no network: the routes read and write through a fake connection behind
``db.acquire`` that answers each statement by its table and verb, emulates the
transaction rollback, and checks that every mutation reads under the owner lock.
"""

from __future__ import annotations

import copy
import json
from contextlib import asynccontextmanager

import pytest
from starlette.testclient import TestClient

from memory_base.serve import api
from memory_base.serve.access import auth
from memory_base.serve.profiles import routes, store

client = TestClient(api.app, headers={"X-API-Key": "test-key"})

OWNER = "claude-code"
NOW = 1_700_000_000.0
NOW_ISO = "2023-11-14T22:13:20+00:00"
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"


# ---- fake database ----------------------------------------------------------


class FakeTransaction:
    def __init__(self, conn, options):
        self.conn = conn
        self.options = options

    async def __aenter__(self):
        self.conn.transactions.append(self.options)
        self.conn.depth += 1
        self.conn.mutating = not self.options.get("readonly", False)
        self.saved = copy.deepcopy((self.conn.versions, self.conn.proposals))
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.conn.depth -= 1
        self.conn.locks.clear()
        if exc_type is not None:
            self.conn.versions, self.conn.proposals = self.saved
        return False


def _kind(query):
    q = " ".join(query.split())
    if "pg_advisory_xact_lock" in q:
        return "lock"
    if q.startswith("INSERT INTO") and "agent_profiles" in q:
        return "insert_version"
    if q.startswith("INSERT INTO") and "profile_proposals" in q:
        return "insert_proposal"
    if q.startswith("UPDATE") and "'superseded'" in q:
        return "supersede"
    if q.startswith("UPDATE") and "profile_proposals" in q:
        return "decide"
    if "agent_profiles" in q and "LIMIT 1" in q:
        return "latest"
    if "agent_profiles" in q:
        return "versions"
    if q.startswith("SELECT owner FROM"):
        return "proposal_owner"
    if "profile_proposals" in q and "WHERE id = $1" in q:
        return "proposal"
    if "profile_proposals" in q and "status = 'pending'" in q:
        return "pending"
    if "profile_proposals" in q:
        return "proposals"
    raise AssertionError(f"unexpected statement: {q}")


class FakeConnection:
    """Emulates the profile statements over two in-memory tables."""

    def __init__(self):
        self.versions: list[dict] = []
        self.proposals: list[dict] = []
        self.transactions: list[dict] = []
        self.calls: list[tuple[str, tuple]] = []
        self.locks: set[str] = set()
        self.depth = 0
        self.mutating = False

    def transaction(self, **options):
        return FakeTransaction(self, options)

    def _owner_of(self, kind, args):
        if kind in ("latest", "insert_version", "supersede", "insert_proposal"):
            return args[0]
        if kind in ("proposal", "decide"):
            return next((p["owner"] for p in self.proposals if p["id"] == args[0]), None)
        return None

    def _run(self, query, args):
        kind = _kind(query)
        self.calls.append((kind, args))
        if kind not in ("versions", "proposals"):
            assert self.depth, f"{kind} runs inside a transaction"
        owner = self._owner_of(kind, args)
        if self.mutating and owner is not None:
            assert owner in self.locks, f"{kind} runs under the owner lock"
        return kind

    def _latest(self, owner, part):
        rows = [v for v in self.versions if v["owner"] == owner and v["part"] == part]
        return max(rows, key=lambda v: v["version"]) if rows else None

    async def execute(self, query, *args):
        kind = self._run(query, args)
        if kind == "lock":
            assert self.mutating
            self.locks.add(args[0])
            return "SELECT 1"
        if kind == "decide":
            proposal_id, status, decided_at, note = args
            row = next(p for p in self.proposals if p["id"] == proposal_id)
            row.update(status=status, decided_at=decided_at, decision_note=note)
            return "UPDATE 1"
        raise AssertionError(f"unexpected execute: {kind}")

    async def fetchval(self, query, *args):
        kind = self._run(query, args)
        if kind == "proposal_owner":
            row = next((p for p in self.proposals if p["id"] == args[0]), None)
            return None if row is None else row["owner"]
        if kind == "insert_version":
            owner, part, version, content, author, proposal_id, created_at = args
            assert (owner, part, version) not in {
                (v["owner"], v["part"], v["version"]) for v in self.versions
            }
            self.versions.append(
                {
                    "owner": owner,
                    "part": part,
                    "version": version,
                    "content": content,
                    "author": author,
                    "proposal_id": proposal_id,
                    "created_at": created_at,
                }
            )
            return version
        if kind == "supersede":
            owner, decided_at = args
            pending = [
                p for p in self.proposals if p["owner"] == owner and p["status"] == "pending"
            ]
            assert len(pending) <= 1
            for row in pending:
                row.update(status="superseded", decided_at=decided_at)
            return pending[0]["id"] if pending else None
        if kind == "insert_proposal":
            owner, content, reason, base_version, created_at = args
            assert not [
                p for p in self.proposals if p["owner"] == owner and p["status"] == "pending"
            ]
            row = {
                "id": len(self.proposals) + 1,
                "owner": owner,
                "content": content,
                "reason": reason,
                "base_version": base_version,
                "status": "pending",
                "created_at": created_at,
                "decided_at": None,
                "decision_note": None,
            }
            self.proposals.append(row)
            return row["id"]
        raise AssertionError(f"unexpected fetchval: {kind}")

    async def fetchrow(self, query, *args):
        kind = self._run(query, args)
        if kind == "latest":
            return copy.deepcopy(self._latest(*args))
        if kind == "proposal":
            row = next((p for p in self.proposals if p["id"] == args[0]), None)
            return copy.deepcopy(row)
        if kind == "pending":
            rows = [p for p in self.proposals if p["owner"] == args[0] and p["status"] == "pending"]
            return copy.deepcopy(rows[0]) if rows else None
        raise AssertionError(f"unexpected fetchrow: {kind}")

    async def fetch(self, query, *args):
        kind = self._run(query, args)
        if kind == "versions":
            owner, part, limit = args
            rows = [v for v in self.versions if v["owner"] == owner and v["part"] == part]
            return copy.deepcopy(sorted(rows, key=lambda v: -v["version"])[:limit])
        if kind == "proposals":
            owner, status, limit = args
            rows = [
                p
                for p in self.proposals
                if (owner is None or p["owner"] == owner)
                and (status is None or p["status"] == status)
            ]
            rows = sorted(rows, key=lambda p: (-p["created_at"], -p["id"]))
            return copy.deepcopy(rows[:limit])
        raise AssertionError(f"unexpected fetch: {kind}")

    # ---- seeding helpers

    def add_version(self, owner, part, content, author=None, proposal_id=None, created_at=NOW):
        latest = self._latest(owner, part)
        version = 1 if latest is None else latest["version"] + 1
        self.versions.append(
            {
                "owner": owner,
                "part": part,
                "version": version,
                "content": content,
                "author": author or (owner if part == "self" else "user"),
                "proposal_id": proposal_id,
                "created_at": created_at,
            }
        )
        return version

    def add_proposal(
        self, owner=OWNER, content="The user lives in Seoul.", base_version=0, status="pending",
        created_at=NOW, reason="learned it", decided_at=None, decision_note=None,
    ):  # fmt: skip
        row = {
            "id": len(self.proposals) + 1,
            "owner": owner,
            "content": content,
            "reason": reason,
            "base_version": base_version,
            "status": status,
            "created_at": created_at,
            "decided_at": decided_at,
            "decision_note": decision_note,
        }
        self.proposals.append(row)
        return row["id"]

    def proposal(self, proposal_id):
        return next(p for p in self.proposals if p["id"] == proposal_id)

    def writes(self):
        return [
            kind
            for kind, _ in self.calls
            if kind in ("insert_version", "insert_proposal", "supersede", "decide")
        ]


@pytest.fixture()
def fake_db(monkeypatch):
    conn = FakeConnection()

    @asynccontextmanager
    async def acquire(timeout=None):
        yield conn

    async def noop(conn):
        return None

    monkeypatch.setattr(store.db, "acquire", acquire)
    monkeypatch.setattr(store, "ensure_schema_once", noop)
    monkeypatch.setattr(store, "_now", lambda: NOW)
    return conn


def _use_identity(monkeypatch, authors=(OWNER,), is_admin=False, label="agents", allowed=()):
    identity = auth.KeyIdentity(
        key_id=f"{label}-key-hash",
        label=label,
        home="default",
        is_admin=is_admin,
        allowed=frozenset(allowed),
        authors=frozenset(authors),
    )

    async def fake_authenticate_request(plaintext_key):
        return identity if plaintext_key == "test-key" else None

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)


@pytest.fixture()
def agent(monkeypatch):
    _use_identity(monkeypatch, authors=(OWNER,))


@pytest.fixture()
def user(monkeypatch):
    _use_identity(monkeypatch, authors=("user",), label="yw0nam")


def _put_self(content="I review every diff before merging.", **fields):
    return client.put("/profiles/self", json={"owner": OWNER, "content": content, **fields})


def _propose(content="The user lives in Seoul.", base_version=0, **fields):
    body = {"owner": OWNER, "content": content, "reason": "they said so", "base_version": 0}
    body.update(base_version=base_version, **fields)
    return client.post("/profiles/user/proposals", json=body)


def _approve(proposal_id, **body):
    return client.post(f"/profiles/user/proposals/{proposal_id}/approve", json=body)


def _reject(proposal_id, **body):
    return client.post(f"/profiles/user/proposals/{proposal_id}/reject", json=body)


def _send(method, path, body):
    kwargs = {"json": body} if body is not None else {}
    return client.request(method.upper(), path, **kwargs)


# ---- authentication ---------------------------------------------------------

PROPOSAL = {"owner": OWNER, "content": "x", "reason": "r", "base_version": 0}
AGENT_ROUTES = [
    ("put", "/profiles/self", {"owner": OWNER, "content": "x"}),
    ("post", "/profiles/user/proposals", PROPOSAL),
]
READ_ROUTES = [
    ("get", "/profiles?owner=claude-code", None),
    ("get", "/profiles/user/proposals?owner=claude-code", None),
    ("get", "/profiles/versions?owner=claude-code&part=self", None),
]
DECISION_ROUTES = [
    ("post", "/profiles/user/proposals/1/approve", {}),
    ("post", "/profiles/user/proposals/1/reject", {}),
]
ROUTES = [
    *AGENT_ROUTES,
    *READ_ROUTES,
    ("get", "/profiles/user/proposals/1", None),
    *DECISION_ROUTES,
]


@pytest.mark.parametrize(("method", "path", "body"), ROUTES)
@pytest.mark.parametrize("key", [None, "unknown-key"])
def test_every_profile_route_needs_a_known_key(fake_db, method, path, body, key):
    bare = TestClient(api.app)
    kwargs = {"headers": {"X-API-Key": key} if key else {}}
    if body is not None:
        kwargs["json"] = body
    response = bare.request(method.upper(), path, **kwargs)
    assert response.status_code == 401
    assert fake_db.calls == []


@pytest.mark.real_auth
def test_a_revoked_key_is_never_resolved(monkeypatch, fake_db):
    seen = []

    class KeyConn:
        async def fetchrow(self, query, *args):
            seen.append(query)
            return None

    @asynccontextmanager
    async def acquire(timeout=None):
        yield KeyConn()

    async def noop(conn):
        return None

    monkeypatch.setattr(auth.db, "acquire", acquire)
    monkeypatch.setattr(auth, "ensure_schema_once", noop)
    response = client.get("/profiles", params={"owner": OWNER})
    assert response.status_code == 401
    assert "revoked_at IS NULL" in seen[0]
    assert fake_db.calls == []


# ---- authority --------------------------------------------------------------


@pytest.mark.parametrize(("method", "path", "body"), AGENT_ROUTES + READ_ROUTES + DECISION_ROUTES)
@pytest.mark.parametrize(
    "identity",
    [
        {"authors": ("natsume", "consolidator"), "is_admin": True, "label": "claude-code"},
        {"authors": (), "is_admin": True, "label": "user", "allowed": ("default", "personal")},
        {"authors": ("natsume",), "is_admin": False, "label": "claude-code"},
    ],
)
def test_admin_label_home_and_namespaces_confer_no_profile_authority(
    monkeypatch, fake_db, method, path, body, identity
):
    _use_identity(monkeypatch, **identity)
    fake_db.add_proposal()
    response = _send(method, path, body)
    assert response.status_code == 403, response.json()
    assert "error" in response.json()
    assert fake_db.calls == []


@pytest.mark.parametrize(("method", "path", "body"), AGENT_ROUTES)
def test_the_user_author_cannot_write_an_agents_parts(monkeypatch, fake_db, method, path, body):
    _use_identity(monkeypatch, authors=("user",))
    response = _send(method, path, body)
    assert response.status_code == 403
    assert fake_db.writes() == []


@pytest.mark.parametrize(("method", "path", "body"), DECISION_ROUTES)
def test_an_owner_author_cannot_decide_its_own_proposal(monkeypatch, fake_db, method, path, body):
    _use_identity(monkeypatch, authors=(OWNER, "natsume", "consolidator"), is_admin=True)
    fake_db.add_proposal()
    response = _send(method, path, body)
    assert response.status_code == 403
    assert "user" in response.json()["error"]
    assert fake_db.proposal(1)["status"] == "pending"
    assert fake_db.writes() == []


def test_a_non_admin_owner_key_writes_and_a_user_key_reads_and_decides(monkeypatch, fake_db):
    _use_identity(monkeypatch, authors=(OWNER,), is_admin=False)
    assert _put_self().status_code == 200
    assert _propose().status_code == 201
    _use_identity(monkeypatch, authors=("user",), is_admin=False)
    assert client.get("/profiles", params={"owner": OWNER}).status_code == 200
    assert client.get("/profiles/user/proposals").status_code == 200
    assert client.get("/profiles/user/proposals/1").status_code == 200
    versions = client.get("/profiles/versions", params={"owner": OWNER, "part": "self"})
    assert versions.status_code == 200
    assert _approve(1).status_code == 200


def test_an_owner_key_reads_its_own_proposal_but_not_another_owners(monkeypatch, fake_db):
    fake_db.add_proposal(owner="natsume")
    fake_db.add_proposal(owner=OWNER)
    _use_identity(monkeypatch, authors=(OWNER,))
    assert client.get("/profiles/user/proposals/2").status_code == 200
    assert client.get("/profiles/user/proposals/1").status_code == 403


# ---- request validation -----------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"content": "x"},
        {"owner": OWNER},
        {"owner": OWNER, "content": "x", "extra": 1},
        {"owner": OWNER, "content": 5},
        {"owner": OWNER, "content": None},
        {"owner": 5, "content": "x"},
        {"owner": "Claude", "content": "x"},
        {"owner": "-bad", "content": "x"},
        {"owner": "a" * 41, "content": "x"},
        {"owner": "user", "content": "x"},
        {"owner": "consolidator", "content": "x"},
        {"owner": OWNER, "content": "x", "max_chars": 199},
        {"owner": OWNER, "content": "x", "max_chars": 20001},
        {"owner": OWNER, "content": "x", "max_chars": True},
        {"owner": OWNER, "content": "x", "max_chars": 500.0},
        {"owner": OWNER, "content": "x", "max_chars": "500"},
        [{"owner": OWNER, "content": "x"}],
    ],
)
def test_self_write_rejects_a_bad_body(monkeypatch, fake_db, body):
    _use_identity(monkeypatch, authors=(OWNER, "user", "consolidator"))
    response = client.put("/profiles/self", json=body)
    assert response.status_code == 400, response.json()
    assert "error" in response.json()
    assert fake_db.calls == []


@pytest.mark.parametrize("raw", [b"", b"{", b"not json", b'"text"'])
def test_mutations_reject_malformed_json(agent, fake_db, raw):
    for method, path in (("PUT", "/profiles/self"), ("POST", "/profiles/user/proposals")):
        assert client.request(method, path, content=raw).status_code == 400
    assert fake_db.calls == []


@pytest.mark.parametrize("raw", [b"", b"{", b"[]", b"null"])
def test_decisions_reject_malformed_json(user, fake_db, raw):
    fake_db.add_proposal()
    for action in ("approve", "reject"):
        response = client.post(f"/profiles/user/proposals/1/{action}", content=raw)
        assert response.status_code == 400
    assert fake_db.calls == []
    assert fake_db.proposal(1)["status"] == "pending"


@pytest.mark.parametrize(
    "change",
    [
        {"owner": None},
        {"content": None},
        {"reason": None},
        {"base_version": None},
        {"extra": True},
        {"reason": ""},
        {"reason": "   "},
        {"reason": "r" * 1001},
        {"reason": 5},
        {"content": 5},
        {"base_version": -1},
        {"base_version": 2147483648},
        {"base_version": True},
        {"base_version": 1.0},
        {"base_version": "0"},
        {"max_chars": 199},
        {"max_chars": 20001},
        {"max_chars": False},
        {"owner": "user"},
        {"owner": "consolidator"},
        {"owner": "Bad Owner"},
    ],
)
def test_proposal_rejects_a_bad_body(monkeypatch, fake_db, change):
    _use_identity(monkeypatch, authors=(OWNER, "user", "consolidator"))
    body = {k: v for k, v in {**PROPOSAL, **change}.items() if v is not None}
    response = client.post("/profiles/user/proposals", json=body)
    assert response.status_code == 400, response.json()
    assert fake_db.calls == []


def test_proposal_accepts_a_reason_of_exactly_1000_characters_after_stripping(agent, fake_db):
    response = _propose(reason="  " + "r" * 1000 + "  ")
    assert response.status_code == 201
    assert fake_db.proposal(1)["reason"] == "r" * 1000


def test_proposal_accepts_the_largest_base_version_and_reports_it_stale(agent, fake_db):
    response = _propose(base_version=2147483647)
    assert response.status_code == 409
    assert response.json() == {"error": "stale", "version": 0}


@pytest.mark.parametrize(
    "body",
    [
        {"note": 5},
        {"note": None},
        {"note": "n" * 1001},
        {"note": "x", "extra": 1},
        {"status": "approved"},
    ],
)
def test_decisions_reject_a_bad_body(user, fake_db, body):
    fake_db.add_proposal()
    for action in ("approve", "reject"):
        response = client.post(f"/profiles/user/proposals/1/{action}", json=body)
        assert response.status_code == 400, response.json()
    assert fake_db.calls == []
    assert fake_db.proposal(1)["status"] == "pending"


def test_decisions_accept_a_note_of_exactly_1000_characters_after_stripping(user, fake_db):
    fake_db.add_proposal()
    assert _reject(1, note=" " + "n" * 1000 + " ").status_code == 200
    assert fake_db.proposal(1)["decision_note"] == "n" * 1000


@pytest.mark.parametrize(
    "proposal_id", ["abc", "0", "-1", "1.5", "9223372036854775808", "1e3", "+1", "01"]
)
def test_proposal_routes_reject_a_malformed_id(user, fake_db, proposal_id):
    assert client.get(f"/profiles/user/proposals/{proposal_id}").status_code == 400
    assert _approve(proposal_id).status_code == 400
    assert _reject(proposal_id).status_code == 400
    assert fake_db.calls == []


def test_the_largest_proposal_id_is_well_formed(user, fake_db):
    assert client.get("/profiles/user/proposals/9223372036854775807").status_code == 404


@pytest.mark.parametrize(
    "query",
    [
        "",
        "owner=",
        "owner=user",
        "owner=consolidator",
        "owner=Bad",
        "owner=claude-code&owner=natsume",
        "owner=claude-code&part=self",
        "owner=claude-code&extra=1",
    ],
)
def test_profile_read_rejects_a_bad_query(monkeypatch, fake_db, query):
    _use_identity(monkeypatch, authors=(OWNER, "user"))
    assert client.get(f"/profiles?{query}").status_code == 400
    assert fake_db.calls == []


@pytest.mark.parametrize(
    "query",
    [
        "owner=claude-code",
        "part=self",
        "owner=claude-code&part=both",
        "owner=claude-code&part=self&part=user",
        "owner=claude-code&part=self&limit=0",
        "owner=claude-code&part=self&limit=201",
        "owner=claude-code&part=self&limit=x",
        "owner=claude-code&part=self&limit=5&limit=6",
        "owner=claude-code&part=self&status=pending",
        "owner=user&part=self",
    ],
)
def test_versions_reject_a_bad_query(monkeypatch, fake_db, query):
    _use_identity(monkeypatch, authors=(OWNER, "user"))
    assert client.get(f"/profiles/versions?{query}").status_code == 400
    assert fake_db.calls == []


@pytest.mark.parametrize(
    "query",
    [
        "status=open",
        "status=pending&status=approved",
        "limit=0",
        "limit=201",
        "limit=ten",
        "owner=user",
        "owner=claude-code&owner=natsume",
        "part=user",
    ],
)
def test_proposal_list_rejects_a_bad_query(monkeypatch, fake_db, query):
    _use_identity(monkeypatch, authors=(OWNER, "user"))
    assert client.get(f"/profiles/user/proposals?{query}").status_code == 400
    assert fake_db.calls == []


def test_proposal_by_id_and_decisions_reject_any_query(user, fake_db):
    fake_db.add_proposal()
    assert client.get("/profiles/user/proposals/1?owner=claude-code").status_code == 400
    assert client.post("/profiles/user/proposals/1/approve?x=1", json={}).status_code == 400
    assert fake_db.calls == []


# ---- self -------------------------------------------------------------------


def test_self_write_stores_the_stripped_content_as_the_next_version(agent, fake_db):
    response = _put_self("  I review every diff.  ")
    assert response.status_code == 200
    assert response.json() == {"status": "written", "version": 1}
    assert fake_db.versions == [
        {
            "owner": OWNER,
            "part": "self",
            "version": 1,
            "content": "I review every diff.",
            "author": OWNER,
            "proposal_id": None,
            "created_at": NOW,
        }
    ]
    assert _put_self("I review every diff twice.").json() == {"status": "written", "version": 2}
    assert fake_db.transactions[-1] == {"isolation": "read_committed"}
    kinds = [kind for kind, _ in fake_db.calls]
    assert kinds[:2] == ["lock", "latest"]
    assert fake_db.calls[0] == ("lock", (OWNER,))


def test_self_write_of_the_latest_content_is_unchanged(agent, fake_db):
    _put_self("Same text.")
    assert _put_self(" Same text.\n").json() == {"status": "unchanged", "version": 1}
    assert len(fake_db.versions) == 1


def test_self_write_of_empty_content_clears_the_part(agent, fake_db):
    _put_self("Something.")
    assert _put_self("   ").json() == {"status": "written", "version": 2}
    assert fake_db.versions[-1]["content"] == ""
    assert _put_self("").json() == {"status": "unchanged", "version": 2}
    body = client.get("/profiles", params={"owner": OWNER}).json()
    assert body["self_version"] == 2
    assert body["self"] is None


def test_self_write_over_the_budget_is_refused_with_its_length(agent, fake_db):
    response = _put_self("x" * 4001)
    assert response.status_code == 400
    assert response.json()["chars"] == 4001
    assert response.json()["max_chars"] == 4000
    assert _put_self("x" * 4000).status_code == 200
    response = _put_self("y" * 301, max_chars=300)
    assert response.status_code == 400
    assert response.json()["chars"] == 301
    assert _put_self("y" * 300, max_chars=300).status_code == 200
    assert _put_self("z" * 20000, max_chars=20000).status_code == 200


def test_self_write_with_a_credential_is_refused_naming_only_its_type(agent, fake_db):
    response = _put_self(f"My token is {GITHUB_TOKEN}.")
    assert response.status_code == 400
    assert "credential" in response.json()["error"]
    assert GITHUB_TOKEN not in json.dumps(response.json())
    assert fake_db.calls == []


def test_self_write_for_another_owner_is_forbidden(agent, fake_db):
    response = client.put("/profiles/self", json={"owner": "natsume", "content": "x"})
    assert response.status_code == 403
    assert fake_db.calls == []


# ---- proposals --------------------------------------------------------------


def test_a_proposal_starts_pending_with_no_decision(agent, fake_db):
    response = _propose("  The user lives in Seoul.  ", reason="  they said so  ")
    assert response.status_code == 201
    assert response.json() == {"id": 1, "status": "pending", "superseded": None}
    assert fake_db.proposals == [
        {
            "id": 1,
            "owner": OWNER,
            "content": "The user lives in Seoul.",
            "reason": "they said so",
            "base_version": 0,
            "status": "pending",
            "created_at": NOW,
            "decided_at": None,
            "decision_note": None,
        }
    ]
    assert fake_db.transactions[-1] == {"isolation": "read_committed"}
    assert fake_db.calls[0] == ("lock", (OWNER,))


def test_a_new_proposal_supersedes_the_pending_one(agent, fake_db, monkeypatch):
    _propose("first")
    monkeypatch.setattr(store, "_now", lambda: NOW + 5)
    response = _propose("second")
    assert response.json() == {"id": 2, "status": "pending", "superseded": 1}
    first, second = fake_db.proposals
    assert first["status"] == "superseded"
    assert first["decided_at"] == NOW + 5
    assert first["decision_note"] is None
    assert second["status"] == "pending"
    assert second["created_at"] == NOW + 5
    assert second["decided_at"] is None
    assert second["decision_note"] is None
    kinds = [kind for kind, _ in fake_db.calls]
    assert kinds[-2:] == ["supersede", "insert_proposal"]


def test_a_proposal_against_a_stale_base_is_refused_with_the_current_version(agent, fake_db):
    fake_db.add_version(OWNER, "user", "Lives in Seoul.")
    fake_db.add_proposal(base_version=1)
    before = dict(fake_db.proposal(1))
    response = _propose(base_version=0)
    assert response.status_code == 409
    assert response.json() == {"error": "stale", "version": 1}
    response = _propose(base_version=2)
    assert response.status_code == 409
    assert response.json() == {"error": "stale", "version": 1}
    assert fake_db.proposal(1) == before
    assert len(fake_db.proposals) == 1
    assert _propose(base_version=1).status_code == 201


def test_a_proposal_over_the_budget_or_with_a_credential_is_refused(agent, fake_db):
    response = _propose("x" * 3001)
    assert response.status_code == 400
    assert response.json()["chars"] == 3001
    assert response.json()["max_chars"] == 3000
    assert _propose("x" * 3001, max_chars=3001).status_code == 201
    before = dict(fake_db.proposal(1))
    for field in ("content", "reason"):
        response = _propose(**{field: f"token {GITHUB_TOKEN}"})
        assert response.status_code == 400
        assert "credential" in response.json()["error"]
        assert GITHUB_TOKEN not in json.dumps(response.json())
    assert fake_db.proposals == [before]


def test_a_proposal_for_another_owner_is_forbidden(agent, fake_db):
    body = dict(PROPOSAL, owner="natsume")
    assert client.post("/profiles/user/proposals", json=body).status_code == 403
    assert fake_db.calls == []


# ---- approve ----------------------------------------------------------------


def test_approve_writes_a_user_version_with_the_proposal_id(user, fake_db, monkeypatch):
    fake_db.add_proposal(content="The user lives in Seoul.")
    monkeypatch.setattr(store, "_now", lambda: NOW + 9)
    response = _approve(1, note="  looks right  ")
    assert response.status_code == 200
    assert response.json() == {"status": "approved", "version": 1}
    assert fake_db.versions == [
        {
            "owner": OWNER,
            "part": "user",
            "version": 1,
            "content": "The user lives in Seoul.",
            "author": "user",
            "proposal_id": 1,
            "created_at": NOW + 9,
        }
    ]
    row = fake_db.proposal(1)
    assert row["status"] == "approved"
    assert row["decided_at"] == NOW + 9
    assert row["decision_note"] == "looks right"
    kinds = [kind for kind, _ in fake_db.calls]
    assert kinds[:3] == ["proposal_owner", "lock", "proposal"]
    assert fake_db.transactions == [{"isolation": "read_committed"}]


def test_approve_without_a_note_stores_null_and_an_empty_note_stores_empty(user, fake_db):
    fake_db.add_proposal()
    _approve(1)
    assert fake_db.proposal(1)["decision_note"] is None
    fake_db.add_proposal(base_version=1)
    _approve(2, note="   ")
    assert fake_db.proposal(2)["decision_note"] == ""


def test_approve_of_a_stale_proposal_leaves_it_pending_and_unchanged(user, fake_db):
    fake_db.add_version(OWNER, "user", "Lives in Seoul.")
    fake_db.add_proposal(base_version=0)
    before = dict(fake_db.proposal(1))
    response = _approve(1, note="ok")
    assert response.status_code == 409
    assert response.json() == {"error": "stale", "version": 1}
    assert fake_db.proposal(1) == before
    assert len(fake_db.versions) == 1


@pytest.mark.parametrize("status", ["approved", "rejected", "superseded"])
@pytest.mark.parametrize("action", ["approve", "reject"])
def test_deciding_a_non_pending_proposal_is_refused_with_its_status(user, fake_db, status, action):
    fake_db.add_proposal(status=status, decided_at=NOW - 1, decision_note="earlier")
    before = dict(fake_db.proposal(1))
    response = client.post(f"/profiles/user/proposals/1/{action}", json={"note": "again"})
    assert response.status_code == 409
    assert response.json() == {"error": "not_pending", "status": status}
    assert fake_db.proposal(1) == before
    assert fake_db.versions == []
    assert fake_db.writes() == []


def test_a_non_pending_status_is_reported_before_a_stale_base(user, fake_db):
    fake_db.add_version(OWNER, "user", "v1")
    fake_db.add_proposal(status="superseded", base_version=0, decided_at=NOW)
    response = _approve(1)
    assert response.json() == {"error": "not_pending", "status": "superseded"}


@pytest.mark.parametrize("action", ["approve", "reject"])
def test_deciding_an_unknown_proposal_is_404(user, fake_db, action):
    response = client.post(f"/profiles/user/proposals/7/{action}", json={})
    assert response.status_code == 404
    assert fake_db.writes() == []


def test_approve_with_a_credential_in_the_note_is_refused(user, fake_db):
    fake_db.add_proposal()
    response = _approve(1, note=f"key {GITHUB_TOKEN}")
    assert response.status_code == 400
    assert GITHUB_TOKEN not in json.dumps(response.json())
    assert fake_db.proposal(1)["status"] == "pending"
    assert fake_db.calls == []


def test_approve_over_an_existing_user_version_increments_it(user, fake_db):
    fake_db.add_version(OWNER, "user", "v1")
    fake_db.add_version(OWNER, "user", "")
    fake_db.add_proposal(content="v3", base_version=2)
    assert _approve(1).json() == {"status": "approved", "version": 3}


# ---- reject -----------------------------------------------------------------


def test_reject_records_the_decision_and_writes_no_version(user, fake_db, monkeypatch):
    fake_db.add_version(OWNER, "user", "v1")
    fake_db.add_proposal(base_version=0)
    monkeypatch.setattr(store, "_now", lambda: NOW + 3)
    response = _reject(1, note=" not true ")
    assert response.status_code == 200
    assert response.json() == {"status": "rejected"}
    row = fake_db.proposal(1)
    assert row["status"] == "rejected"
    assert row["decided_at"] == NOW + 3
    assert row["decision_note"] == "not true"
    assert len(fake_db.versions) == 1
    assert [kind for kind, _ in fake_db.calls][:3] == ["proposal_owner", "lock", "proposal"]
    fake_db.add_proposal(base_version=1)
    _reject(2)
    assert fake_db.proposal(2)["decision_note"] is None


# ---- reads ------------------------------------------------------------------


def test_profile_read_of_an_owner_without_versions(agent, fake_db):
    response = client.get("/profiles", params={"owner": OWNER})
    assert response.status_code == 200
    assert response.json() == {
        "owner": OWNER,
        "self_version": 0,
        "self": None,
        "user_version": 0,
        "user": None,
        "pending_proposal": None,
    }
    assert fake_db.transactions == [{"isolation": "repeatable_read", "readonly": True}]


def test_profile_read_serves_the_latest_parts_and_the_pending_proposal(user, fake_db):
    fake_db.add_version(OWNER, "self", "old self")
    fake_db.add_version(OWNER, "self", "Self text.", created_at=NOW + 1)
    fake_db.add_version(OWNER, "user", "Lives in Seoul.", proposal_id=1)
    fake_db.add_version("natsume", "user", "Other owner.")
    fake_db.add_proposal(status="approved", decided_at=NOW)
    fake_db.add_proposal(base_version=1, reason="moved", created_at=NOW + 2)
    fake_db.add_proposal(owner="natsume")
    body = client.get("/profiles", params={"owner": OWNER}).json()
    assert body == {
        "owner": OWNER,
        "self_version": 2,
        "self": {"content": "Self text.", "created_at": "2023-11-14T22:13:21+00:00"},
        "user_version": 1,
        "user": {"content": "Lives in Seoul.", "created_at": NOW_ISO},
        "pending_proposal": {
            "id": 2,
            "created_at": "2023-11-14T22:13:22+00:00",
            "reason": "moved",
            "base_version": 1,
        },
    }


def test_profile_read_keeps_the_version_of_a_cleared_part_and_never_serves_an_older_one(
    agent, fake_db
):
    fake_db.add_version(OWNER, "user", "Lives in Seoul.")
    fake_db.add_version(OWNER, "user", "")
    fake_db.add_version(OWNER, "self", "Self.")
    fake_db.add_version(OWNER, "self", "")
    body = client.get("/profiles", params={"owner": OWNER}).json()
    assert body["user_version"] == 2
    assert body["user"] is None
    assert body["self_version"] == 2
    assert body["self"] is None


def test_profile_read_with_only_a_pending_proposal(agent, fake_db):
    fake_db.add_proposal()
    body = client.get("/profiles", params={"owner": OWNER}).json()
    assert body["user"] is None and body["self"] is None
    assert body["pending_proposal"]["id"] == 1


def test_versions_are_listed_newest_first_within_the_limit(agent, fake_db):
    for text in ("one", "two", "three"):
        fake_db.add_version(OWNER, "self", text)
    fake_db.add_version(OWNER, "user", "user one", proposal_id=4)
    response = client.get("/profiles/versions", params={"owner": OWNER, "part": "self", "limit": 2})
    assert response.status_code == 200
    assert response.json() == [
        {"version": 3, "content": "three", "author": OWNER, "proposal_id": None, "created_at": NOW_ISO},
        {"version": 2, "content": "two", "author": OWNER, "proposal_id": None, "created_at": NOW_ISO},
    ]  # fmt: skip
    users = client.get("/profiles/versions", params={"owner": OWNER, "part": "user"}).json()
    assert users == [
        {"version": 1, "content": "user one", "author": "user", "proposal_id": 4, "created_at": NOW_ISO}
    ]  # fmt: skip
    default = client.get("/profiles/versions", params={"owner": OWNER, "part": "self"})
    assert [row["version"] for row in default.json()] == [3, 2, 1]
    assert ("versions", (OWNER, "self", 20)) in fake_db.calls
    maximum = client.get(
        "/profiles/versions", params={"owner": OWNER, "part": "self", "limit": 200}
    )
    assert maximum.status_code == 200


def test_proposal_list_is_newest_first_with_every_field(agent, fake_db):
    fake_db.add_proposal(status="superseded", decided_at=NOW + 1, created_at=NOW)
    fake_db.add_proposal(owner="natsume", created_at=NOW + 1)
    fake_db.add_proposal(created_at=NOW + 1, reason="newer")
    response = client.get("/profiles/user/proposals", params={"owner": OWNER})
    assert response.status_code == 200
    rows = response.json()
    assert [row["id"] for row in rows] == [3, 1]
    assert rows[1] == {
        "id": 1,
        "owner": OWNER,
        "status": "superseded",
        "base_version": 0,
        "reason": "learned it",
        "content": "The user lives in Seoul.",
        "created_at": NOW_ISO,
        "decided_at": "2023-11-14T22:13:21+00:00",
        "decision_note": None,
    }
    assert rows[0]["decided_at"] is None
    pending = client.get("/profiles/user/proposals", params={"owner": OWNER, "status": "pending"})
    assert [row["id"] for row in pending.json()] == [3]
    limited = client.get("/profiles/user/proposals", params={"owner": OWNER, "limit": 1})
    assert [row["id"] for row in limited.json()] == [3]
    assert ("proposals", (OWNER, None, 20)) in fake_db.calls


def test_the_ownerless_proposal_list_is_for_user_author_keys_only(monkeypatch, fake_db):
    fake_db.add_proposal(owner="natsume")
    fake_db.add_proposal(owner=OWNER)
    _use_identity(monkeypatch, authors=(OWNER, "natsume", "consolidator"), is_admin=True)
    assert client.get("/profiles/user/proposals").status_code == 403
    assert fake_db.calls == []
    _use_identity(monkeypatch, authors=("user",))
    response = client.get("/profiles/user/proposals", params={"status": "pending"})
    assert response.status_code == 200
    assert [row["owner"] for row in response.json()] == [OWNER, "natsume"]
    assert ("proposals", (None, "pending", 20)) in fake_db.calls


def test_a_proposal_by_id_carries_the_current_user_content_from_one_snapshot(user, fake_db):
    fake_db.add_version(OWNER, "user", "Lives in Busan.")
    fake_db.add_proposal(content="Lives in Seoul.", base_version=1, reason="moved")
    response = client.get("/profiles/user/proposals/1")
    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "owner": OWNER,
        "status": "pending",
        "base_version": 1,
        "reason": "moved",
        "content": "Lives in Seoul.",
        "created_at": NOW_ISO,
        "decided_at": None,
        "decision_note": None,
        "current_user_version": 1,
        "current_user_content": "Lives in Busan.",
    }
    assert fake_db.transactions == [{"isolation": "repeatable_read", "readonly": True}]


def test_a_proposal_by_id_reports_a_cleared_or_absent_user_part_as_null(user, fake_db):
    fake_db.add_proposal(status="rejected", decided_at=NOW, decision_note="no")
    body = client.get("/profiles/user/proposals/1").json()
    assert body["current_user_version"] == 0
    assert body["current_user_content"] is None
    assert body["decision_note"] == "no"
    fake_db.add_version(OWNER, "user", "")
    body = client.get("/profiles/user/proposals/1").json()
    assert body["current_user_version"] == 1
    assert body["current_user_content"] is None


def test_an_unknown_proposal_by_id_is_404(user, fake_db):
    assert client.get("/profiles/user/proposals/3").status_code == 404


def test_the_old_profile_routes_and_slot_machinery_are_gone(monkeypatch, fake_db):
    _use_identity(monkeypatch, authors=("consolidator", OWNER, "user"), is_admin=True)
    for method, path in (
        ("GET", "/admin/profiles/sources?namespace=default&slot=user"),
        ("PUT", "/admin/profiles"),
        ("GET", "/admin/profiles/versions?namespace=default&slot=user"),
    ):
        assert client.request(method, path).status_code in (404, 405)
    assert client.get("/profiles").status_code == 400
    assert client.get("/profiles", params={"namespace": "default"}).status_code == 400
    for name in ("PROFILE_VERSION", "source_hash", "render_rules", "SLOT_KINDS", "SLOT_FIELDS"):
        assert not hasattr(store, name)
        assert not hasattr(routes, name)
