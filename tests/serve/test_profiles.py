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
from memory_base.serve.profiles import store

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
        self.saved = copy.deepcopy(self.conn.versions)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.conn.depth -= 1
        self.conn.locks.clear()
        if exc_type is not None:
            self.conn.versions = self.saved
        return False


def _kind(query):
    q = " ".join(query.split())
    if "pg_advisory_xact_lock" in q:
        return "lock"
    if q.startswith("INSERT INTO") and "agent_profiles" in q:
        return "insert_version"
    if "agent_profiles" in q and "LIMIT 1" in q:
        return "latest"
    if "agent_profiles" in q:
        return "versions"
    raise AssertionError(f"unexpected statement: {q}")


class FakeConnection:
    """Emulates the profile statements over one in-memory table."""

    def __init__(self):
        self.versions: list[dict] = []
        self.transactions: list[dict] = []
        self.calls: list[tuple[str, tuple]] = []
        self.locks: set[str] = set()
        self.depth = 0
        self.mutating = False

    def transaction(self, **options):
        return FakeTransaction(self, options)

    def _run(self, query, args):
        kind = _kind(query)
        self.calls.append((kind, args))
        if kind != "versions":
            assert self.depth, f"{kind} runs inside a transaction"
        if self.mutating and kind in ("latest", "insert_version"):
            assert args[0] in self.locks, f"{kind} runs under the owner lock"
        return kind

    def _latest(self, owner, part):
        rows = [v for v in self.versions if v["owner"] == owner and v["part"] == part]
        return max(rows, key=lambda v: v["version"]) if rows else None

    async def execute(self, query, *args):
        kind = self._run(query, args)
        assert kind == "lock" and self.mutating
        self.locks.add(args[0])
        return "SELECT 1"

    async def fetchval(self, query, *args):
        kind = self._run(query, args)
        assert kind == "insert_version"
        owner, part, version, content, author, reason, method, consented_at, created_at = args
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
                "reason": reason,
                "consent_method": method,
                "consented_at": consented_at,
                "created_at": created_at,
            }
        )
        return version

    async def fetchrow(self, query, *args):
        kind = self._run(query, args)
        assert kind == "latest"
        return copy.deepcopy(self._latest(*args))

    async def fetch(self, query, *args):
        kind = self._run(query, args)
        assert kind == "versions"
        owner, part, limit = args
        rows = [v for v in self.versions if v["owner"] == owner and v["part"] == part]
        return copy.deepcopy(sorted(rows, key=lambda v: -v["version"])[:limit])

    def add_version(
        self, owner, part, content, author=None, reason=None, consent_method=None,
        consented_at=None, created_at=NOW,
    ):  # fmt: skip
        latest = self._latest(owner, part)
        version = 1 if latest is None else latest["version"] + 1
        self.versions.append(
            {
                "owner": owner,
                "part": part,
                "version": version,
                "content": content,
                "author": author or owner,
                "reason": reason,
                "consent_method": consent_method,
                "consented_at": consented_at,
                "created_at": created_at,
            }
        )
        return version

    def writes(self):
        return [kind for kind, _ in self.calls if kind == "insert_version"]


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


def _put_user(content="The user lives in Seoul.", base_version=0, **fields):
    body = {
        "owner": OWNER,
        "content": content,
        "reason": "they said so",
        "base_version": base_version,
        "consent_method": "conversation",
    }
    return client.put("/profiles/user", json={**body, **fields})


def _send(method, path, body):
    kwargs = {"json": body} if body is not None else {}
    return client.request(method.upper(), path, **kwargs)


# ---- authentication ---------------------------------------------------------

USER_BODY = {
    "owner": OWNER, "content": "x", "reason": "r", "base_version": 0,
    "consent_method": "conversation",
}  # fmt: skip
AGENT_ROUTES = [
    ("put", "/profiles/self", {"owner": OWNER, "content": "x"}),
    ("put", "/profiles/user", USER_BODY),
]
READ_ROUTES = [
    ("get", "/profiles?owner=claude-code", None),
    ("get", "/profiles/versions?owner=claude-code&part=self", None),
]
ROUTES = [*AGENT_ROUTES, *READ_ROUTES]


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


@pytest.mark.parametrize(("method", "path", "body"), AGENT_ROUTES + READ_ROUTES)
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


def test_a_user_key_reads_every_owner(user, fake_db):
    fake_db.add_version(OWNER, "self", "Self.")
    assert client.get("/profiles", params={"owner": OWNER}).status_code == 200
    versions = client.get("/profiles/versions", params={"owner": OWNER, "part": "self"})
    assert versions.status_code == 200


def test_the_import_method_needs_an_admin_key_carrying_the_user_author(monkeypatch, fake_db):
    for authors, admin in (((OWNER,), True), ((OWNER, "user"), False)):
        _use_identity(monkeypatch, authors=authors, is_admin=admin)
        response = _put_user(consent_method="import")
        assert response.status_code == 403
        assert "import" in response.json()["error"]
    assert fake_db.calls == []
    _use_identity(monkeypatch, authors=(OWNER, "user"), is_admin=True)
    assert _put_user(consent_method="import").json() == {"status": "written", "version": 1}
    assert fake_db.versions[0]["consent_method"] == "import"


# ---- request validation -----------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"content": "x"},
        {"owner": OWNER, "content": "x", "extra": 1},
        {"owner": OWNER, "content": 5},
        {"owner": "-bad", "content": "x"},
        {"owner": "a" * 41, "content": "x"},
        {"owner": "user", "content": "x"},
        {"owner": "consolidator", "content": "x"},
        {"owner": OWNER, "content": "x", "max_chars": 199},
        {"owner": OWNER, "content": "x", "max_chars": 20001},
        {"owner": OWNER, "content": "x", "max_chars": True},
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


@pytest.mark.parametrize("raw", [b"{", b'"text"'])
def test_mutations_reject_malformed_json(agent, fake_db, raw):
    for path in ("/profiles/self", "/profiles/user"):
        assert client.put(path, content=raw).status_code == 400
    assert fake_db.calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"owner": None},
        {"extra": True},
        {"reason": "   "},
        {"reason": "r" * 1001},
        {"reason": 5},
        {"content": 5},
        {"base_version": -1},
        {"base_version": 2147483648},
        {"base_version": True},
        {"max_chars": 199},
        {"max_chars": 20001},
        {"owner": "user"},
        {"owner": "Bad Owner"},
        {"consent_method": None},
        {"consent_method": "approval"},
        {"dry_run": "yes"},
    ],
)
def test_user_write_rejects_a_bad_body(monkeypatch, fake_db, change):
    _use_identity(monkeypatch, authors=(OWNER, "user", "consolidator"))
    body = {k: v for k, v in {**USER_BODY, **change}.items() if v is not None}
    response = client.put("/profiles/user", json=body)
    assert response.status_code == 400, response.json()
    assert fake_db.calls == []


def test_user_write_accepts_a_reason_of_exactly_1000_characters_after_stripping(agent, fake_db):
    response = _put_user(reason="  " + "r" * 1000 + "  ")
    assert response.status_code == 200
    assert fake_db.versions[0]["reason"] == "r" * 1000


def test_user_write_accepts_the_largest_base_version_and_reports_it_stale(agent, fake_db):
    response = _put_user(base_version=2147483647)
    assert response.status_code == 409
    assert response.json() == {"error": "stale", "version": 0}


@pytest.mark.parametrize(
    "query",
    [
        "",
        "owner=user",
        "owner=Bad",
        "owner=claude-code&owner=natsume",
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
        "owner=claude-code&part=both",
        "owner=claude-code&part=self&part=user",
        "owner=claude-code&part=self&limit=0",
        "owner=claude-code&part=self&limit=201",
        "owner=claude-code&part=self&status=pending",
        "owner=user&part=self",
    ],
)
def test_versions_reject_a_bad_query(monkeypatch, fake_db, query):
    _use_identity(monkeypatch, authors=(OWNER, "user"))
    assert client.get(f"/profiles/versions?{query}").status_code == 400
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
            "reason": None,
            "consent_method": None,
            "consented_at": None,
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


# ---- user -------------------------------------------------------------------


def test_user_write_stores_the_next_version_with_the_consent_record(agent, fake_db):
    response = _put_user("  The user lives in Seoul.  ", reason="  they said so  ")
    assert response.status_code == 200
    assert response.json() == {"status": "written", "version": 1}
    assert fake_db.versions == [
        {
            "owner": OWNER,
            "part": "user",
            "version": 1,
            "content": "The user lives in Seoul.",
            "author": OWNER,
            "reason": "they said so",
            "consent_method": "conversation",
            "consented_at": NOW,
            "created_at": NOW,
        }
    ]
    assert _put_user("Lives in Busan.", base_version=1, consent_method="elicitation").json() == {
        "status": "written",
        "version": 2,
    }
    assert fake_db.versions[1]["consent_method"] == "elicitation"
    assert fake_db.calls[0] == ("lock", (OWNER,))


def test_a_dry_run_returns_the_current_content_and_writes_nothing(agent, fake_db):
    assert _put_user("Lives in Seoul.", dry_run=True).json() == {
        "status": "valid",
        "version": 0,
        "current": "",
    }
    fake_db.add_version(OWNER, "user", "Lives in Busan.")
    response = _put_user("Lives in Seoul.", base_version=1, dry_run=True)
    assert response.json() == {"status": "valid", "version": 1, "current": "Lives in Busan."}
    assert fake_db.writes() == []


def test_content_equal_to_the_latest_is_unchanged_even_as_a_dry_run(agent, fake_db):
    fake_db.add_version(OWNER, "user", "Lives in Seoul.")
    for dry_run in (True, False):
        response = _put_user(" Lives in Seoul. ", base_version=1, dry_run=dry_run)
        assert response.json() == {"status": "unchanged", "version": 1}
    assert fake_db.writes() == []


@pytest.mark.parametrize("dry_run", [True, False])
def test_a_stale_base_is_refused_with_the_current_version_and_writes_nothing(
    agent, fake_db, dry_run
):
    fake_db.add_version(OWNER, "user", "Lives in Busan.")
    response = _put_user("Lives in Seoul.", base_version=0, dry_run=dry_run)
    assert response.status_code == 409
    assert response.json() == {"error": "stale", "version": 1}
    assert fake_db.writes() == []


def test_user_write_over_the_budget_or_with_a_credential_is_refused(agent, fake_db):
    over = _put_user("x" * 3001)
    assert over.status_code == 400
    assert over.json()["chars"] == 3001
    assert over.json()["max_chars"] == 3000
    secret = _put_user(f"token {GITHUB_TOKEN}")
    assert secret.status_code == 400
    assert GITHUB_TOKEN not in json.dumps(secret.json())
    assert fake_db.calls == []


def test_user_write_for_another_owner_is_forbidden(agent, fake_db):
    assert _put_user(owner="natsume").status_code == 403
    assert fake_db.calls == []


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
    }
    assert fake_db.transactions == [{"isolation": "repeatable_read", "readonly": True}]


def test_profile_read_serves_the_latest_parts(user, fake_db):
    fake_db.add_version(OWNER, "self", "old self")
    fake_db.add_version(OWNER, "self", "Self text.", created_at=NOW + 1)
    fake_db.add_version(OWNER, "user", "Lives in Seoul.")
    fake_db.add_version("natsume", "user", "Other owner.")
    body = client.get("/profiles", params={"owner": OWNER}).json()
    assert body == {
        "owner": OWNER,
        "self_version": 2,
        "self": {"content": "Self text.", "created_at": "2023-11-14T22:13:21+00:00"},
        "user_version": 1,
        "user": {"content": "Lives in Seoul.", "created_at": NOW_ISO},
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


def test_versions_are_listed_newest_first_with_their_consent_record(agent, fake_db):
    for text in ("one", "two", "three"):
        fake_db.add_version(OWNER, "self", text)
    fake_db.add_version(
        OWNER, "user", "user one", reason="they said so", consent_method="elicitation",
        consented_at=NOW + 5,
    )  # fmt: skip
    response = client.get("/profiles/versions", params={"owner": OWNER, "part": "self", "limit": 2})
    assert response.status_code == 200
    assert response.json() == [
        {
            "version": version, "content": text, "author": OWNER, "reason": None,
            "consent_method": None, "consented_at": None, "created_at": NOW_ISO,
        }
        for version, text in ((3, "three"), (2, "two"))
    ]  # fmt: skip
    users = client.get("/profiles/versions", params={"owner": OWNER, "part": "user"}).json()
    assert users == [
        {
            "version": 1, "content": "user one", "author": OWNER, "reason": "they said so",
            "consent_method": "elicitation", "consented_at": "2023-11-14T22:13:25+00:00",
            "created_at": NOW_ISO,
        }
    ]  # fmt: skip
    default = client.get("/profiles/versions", params={"owner": OWNER, "part": "self"})
    assert [row["version"] for row in default.json()] == [3, 2, 1]
    assert ("versions", (OWNER, "self", 20)) in fake_db.calls
