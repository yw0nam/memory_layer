"""Unit contracts for POST /tables/query with fake query-role connections."""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from decimal import Decimal

import asyncpg
import httpx
import pytest

from memory_base.serve import api, auth, tables


def _post(body, *, api_key="test-key"):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": api_key},
        ) as client:
            return await client.post("/tables/query", json=body)

    return asyncio.run(request())


class FakeAttribute:
    def __init__(self, name):
        self.name = name


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.fetch_sizes = []

    async def fetch(self, size):
        self.fetch_sizes.append(size)
        return self.rows[:size]


class FakeStatement:
    def __init__(self, columns, rows):
        self.attributes = [FakeAttribute(column) for column in columns]
        self.cursor_instance = FakeCursor(rows)

    def get_attributes(self):
        return self.attributes

    async def cursor(self):
        return self.cursor_instance


class FakeTransaction:
    def __init__(self, connection, readonly):
        self.connection = connection
        self.readonly = readonly

    async def __aenter__(self):
        self.connection.events.append(("begin", self.readonly))

    async def __aexit__(self, *args):
        self.connection.events.append(("end", args[0]))


class FakeConnection:
    def __init__(self, columns, rows):
        self.statement = FakeStatement(columns, rows)
        self.events = []

    def transaction(self, *, readonly=False):
        return FakeTransaction(self, readonly)

    async def execute(self, sql):
        self.events.append(("execute", sql))

    async def prepare(self, sql):
        self.events.append(("prepare", sql))
        return self.statement


def _patch_connection(monkeypatch, connection):
    @asynccontextmanager
    async def acquire():
        yield connection

    monkeypatch.setattr(tables.db, "acquire_table_query", acquire)


def test_execute_table_query_uses_read_only_extended_protocol_and_normalizes_values(monkeypatch):
    identifier = uuid.uuid4()
    instant = datetime(2026, 8, 11, 3, 4, 5, tzinfo=timezone.utc)
    connection = FakeConnection(
        ["numeric", "day", "instant", "identifier", "payload"],
        [[Decimal("2.5"), date(2026, 8, 11), instant, identifier, {"ok": True}]],
    )
    _patch_connection(monkeypatch, connection)
    sql = "SELECT 2.5::numeric, current_date, now(), gen_random_uuid(), '{}'::jsonb"

    result = asyncio.run(tables.execute_table_query(sql, "team-a"))

    assert result == {
        "columns": ["numeric", "day", "instant", "identifier", "payload"],
        "rows": [[2.5, "2026-08-11", instant.isoformat(), str(identifier), {"ok": True}]],
        "row_count": 1,
        "truncated": False,
    }
    assert connection.events == [
        ("begin", True),
        ("execute", "SET LOCAL app.namespace = 'team-a'"),
        ("prepare", sql),
        ("end", None),
    ]
    assert connection.statement.cursor_instance.fetch_sizes == [tables.ROW_CAP + 1]


def test_execute_table_query_fetches_one_extra_row_to_report_truncation(monkeypatch):
    rows = [[index] for index in range(tables.ROW_CAP + 1)]
    connection = FakeConnection(["value"], rows)
    _patch_connection(monkeypatch, connection)

    result = asyncio.run(tables.execute_table_query("SELECT value FROM t", "default"))

    assert result["row_count"] == tables.ROW_CAP
    assert len(result["rows"]) == tables.ROW_CAP
    assert result["truncated"] is True


def test_execute_table_query_rejects_binary_values(monkeypatch):
    connection = FakeConnection(["payload"], [[memoryview(b"secret")]])
    _patch_connection(monkeypatch, connection)

    with pytest.raises(tables.UnsupportedTableValueError, match="binary"):
        asyncio.run(tables.execute_table_query("SELECT payload FROM t", "default"))


@pytest.mark.parametrize("sql", ["INSERT INTO t VALUES (1)", "DELETE FROM t", "", "  "])
def test_query_route_rejects_non_select_prefix_before_opening_pool(monkeypatch, sql):
    async def forbidden(*args):
        raise AssertionError("rejected SQL must not reach the query pool")

    monkeypatch.setattr(tables, "execute_table_query", forbidden)
    response = _post({"sql": sql})
    assert response.status_code == 400


@pytest.mark.parametrize("sql", ["SELECT 1", "  select 1", "WITH x AS (SELECT 1) SELECT * FROM x"])
def test_query_route_accepts_select_and_with_prefixes(monkeypatch, sql):
    captured = {}

    async def execute(query, namespace):
        captured.update(sql=query, namespace=namespace)
        return {"columns": [], "rows": [], "row_count": 0, "truncated": False}

    monkeypatch.setattr(tables, "execute_table_query", execute)
    response = _post({"sql": sql})
    assert response.status_code == 200
    assert captured == {"sql": sql.strip(), "namespace": "default"}


def test_query_route_defaults_namespace_to_key_home(monkeypatch):
    identity = auth.KeyIdentity(
        key_id="member-hash",
        label="member",
        home="team-a",
        is_admin=False,
        allowed=frozenset({"team-a"}),
    )

    async def authenticate(plaintext):
        return identity if plaintext == "member-key" else None

    captured = {}

    async def execute(sql, namespace):
        captured["namespace"] = namespace
        return {"columns": [], "rows": [], "row_count": 0, "truncated": False}

    monkeypatch.setattr(auth, "authenticate_request", authenticate)
    monkeypatch.setattr(tables, "execute_table_query", execute)
    response = _post({"sql": "SELECT 1"}, api_key="member-key")
    assert response.status_code == 200
    assert captured["namespace"] == "team-a"


def test_query_route_rejects_namespace_outside_key_scope(monkeypatch):
    identity = auth.KeyIdentity(
        key_id="member-hash",
        label="member",
        home="default",
        is_admin=False,
        allowed=frozenset({"default"}),
    )

    async def authenticate(plaintext):
        return identity if plaintext == "member-key" else None

    monkeypatch.setattr(auth, "authenticate_request", authenticate)
    response = _post({"sql": "SELECT 1", "namespace": "team-a"}, api_key="member-key")
    assert response.status_code == 403


def test_query_route_revalidates_namespace_slug(monkeypatch):
    response = _post({"sql": "SELECT 1", "namespace": "Bad Namespace"})
    assert response.status_code == 400
    assert "namespace must match" in response.json()["error"]


@pytest.mark.parametrize(
    ("exception", "status"),
    [
        (asyncpg.QueryCanceledError("statement timeout"), 408),
        (asyncpg.PostgresError("permission denied"), 400),
        (tables.UnsupportedTableValueError("unsupported binary value"), 400),
    ],
)
def test_query_route_maps_execution_errors(monkeypatch, exception, status):
    async def execute(sql, namespace):
        raise exception

    monkeypatch.setattr(tables, "execute_table_query", execute)
    response = _post({"sql": "SELECT 1"})
    assert response.status_code == status
    assert set(response.json()) == {"error"}


def test_query_route_rejects_serialized_payload_over_five_mb(monkeypatch):
    async def execute(sql, namespace):
        return {
            "columns": ["payload"],
            "rows": [["x" * (tables.RESPONSE_MAX_BYTES + 1)]],
            "row_count": 1,
            "truncated": False,
        }

    monkeypatch.setattr(tables, "execute_table_query", execute)
    response = _post({"sql": "SELECT payload"})
    assert response.status_code == 413


def test_query_route_serialization_is_compact_utf8(monkeypatch):
    payload = {"columns": ["value"], "rows": [["한글"]], "row_count": 1, "truncated": False}

    async def execute(sql, namespace):
        return payload

    monkeypatch.setattr(tables, "execute_table_query", execute)
    response = _post({"sql": "SELECT value"})
    assert response.status_code == 200
    assert (
        response.content
        == json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    )
