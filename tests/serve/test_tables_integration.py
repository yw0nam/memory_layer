"""Integration coverage for CSV row storage and the restricted SQL lane."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import asyncpg
import httpx
import pytest

from memory_base.core import db
from memory_base.core.config import EMB_DIM, PG_SCHEMA, db_url
from memory_base.core.schema import ensure_schema
from memory_base.retrieval.search import search
from memory_base.serve import api, ingest_api, mcp_server, namespaces

NAMESPACE_A = "zzz-tables-a"
NAMESPACE_B = "zzz-tables-b"
DELETE_NAMESPACE = "zzz-tables-delete"
DOCUMENT_ID = "zzz-sql-lane.csv"
REPLACE_DOCUMENT_ID = "zzz-sql-lane-replace.csv"
SUMMARY = "SQL lane integration grouped values table marker."
ROW_SECRET = "row-secret-never-embedded"


def _db_reachable() -> bool:
    async def check():
        conn = await asyncpg.connect(db_url(), timeout=5)
        await conn.close()

    try:
        asyncio.run(check())
        return True
    except Exception:
        return False


if not _db_reachable():
    pytest.skip("DB is not configured or not reachable", allow_module_level=True)

pytestmark = pytest.mark.integration


async def _cleanup(conn):
    await conn.execute(
        f'DELETE FROM "{PG_SCHEMA}".memory_chunks WHERE source_ref = ANY($1::text[])',
        [DOCUMENT_ID, REPLACE_DOCUMENT_ID],
    )
    await conn.execute(
        f'DELETE FROM "{PG_SCHEMA}".doc_rows WHERE document_id = ANY($1::text[])',
        [DOCUMENT_ID, REPLACE_DOCUMENT_ID, "delete-only.csv"],
    )
    await conn.execute(
        f'DELETE FROM "{PG_SCHEMA}".namespaces WHERE name = ANY($1::text[])',
        [NAMESPACE_A, NAMESPACE_B, DELETE_NAMESPACE],
    )


@pytest.fixture(scope="module", autouse=True)
def provision_sql_lane():
    async def setup():
        conn = await asyncpg.connect(db_url())
        try:
            await ensure_schema(conn)
            await _cleanup(conn)
            await conn.executemany(
                f'INSERT INTO "{PG_SCHEMA}".namespaces (name, created_at) VALUES ($1, $2)',
                [
                    (NAMESPACE_A, time.time()),
                    (NAMESPACE_B, time.time()),
                    (DELETE_NAMESPACE, time.time()),
                ],
            )
        finally:
            await conn.close()
        await db.close_table_query_pool()

    asyncio.run(setup())
    yield

    async def teardown():
        await db.close_table_query_pool()
        conn = await asyncpg.connect(db_url())
        try:
            await _cleanup(conn)
        finally:
            await conn.close()

    asyncio.run(teardown())


def _post(sql: str, namespace: str = NAMESPACE_A):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": "test-key"},
        ) as client:
            return await client.post("/tables/query", json={"sql": sql, "namespace": namespace})

    return asyncio.run(request())


def _zero_embedding(rows):
    vector = "[" + ",".join("0" for _ in range(EMB_DIM)) + "]"
    for row in rows:
        row["embedding"] = vector
        row.pop("embedding_text")


def _job(document_id: str) -> ingest_api.IngestJob:
    now = time.time()
    return ingest_api.IngestJob(
        job_id=f"job-{document_id}",
        document_id=document_id,
        namespace=NAMESPACE_A,
        filename=document_id,
        key_label="integration",
        status="running",
        created_at=now,
        updated_at=now,
    )


def test_two_thousand_row_csv_ingest_queries_aggregate_search_and_truncation(
    monkeypatch, tmp_path: Path
):
    path = tmp_path / DOCUMENT_ID
    source_rows = [
        ("alpha" if index % 2 == 0 else "beta", str(index % 17), f"{ROW_SECRET}-{index}")
        for index in range(2_000)
    ]
    path.write_text("group,value,secret\n" + "\n".join(",".join(row) for row in source_rows) + "\n")

    async def summarize(*args, **kwargs):
        return {"summary": SUMMARY, "tags": ["sql lane integration"]}

    async def embed(rows):
        _zero_embedding(rows)

    monkeypatch.setattr(ingest_api, "summarize_and_tag", summarize)
    monkeypatch.setattr(ingest_api, "_embed_rows", embed)
    job = _job(DOCUMENT_ID)
    asyncio.run(
        ingest_api.run_document_job(
            job,
            path,
            DOCUMENT_ID,
            "force",
            "integration:test",
            NAMESPACE_A,
        )
    )

    assert job.status == "succeeded"
    count = _post(
        f"SELECT COUNT(*) AS count FROM memory.doc_rows WHERE document_id = '{DOCUMENT_ID}'"
    )
    assert count.status_code == 200
    assert count.json() == {
        "columns": ["count"],
        "rows": [[2_000]],
        "row_count": 1,
        "truncated": False,
    }

    first = _post(
        "SELECT row_index, data FROM memory.doc_rows "
        f"WHERE document_id = '{DOCUMENT_ID}' AND row_index < 1000 ORDER BY row_index"
    ).json()
    second = _post(
        "SELECT row_index, data FROM memory.doc_rows "
        f"WHERE document_id = '{DOCUMENT_ID}' AND row_index >= 1000 ORDER BY row_index"
    ).json()
    stitched = first["rows"] + second["rows"]
    assert first["truncated"] is False
    assert second["truncated"] is False
    assert [row[0] for row in stitched] == list(range(2_000))
    assert stitched[0][1] == {
        "group": source_rows[0][0],
        "value": source_rows[0][1],
        "secret": source_rows[0][2],
    }

    expected = {
        group: sum(int(value) for item_group, value, _ in source_rows if item_group == group)
        / sum(item_group == group for item_group, _, _ in source_rows)
        for group in ("alpha", "beta")
    }
    aggregate = _post(
        "SELECT data->>'group' AS group_name, "
        "AVG((data->>'value')::numeric) AS mean_value "
        "FROM memory.doc_rows "
        f"WHERE document_id = '{DOCUMENT_ID}' GROUP BY 1 ORDER BY 1"
    ).json()
    assert aggregate["rows"] == [
        ["alpha", pytest.approx(expected["alpha"])],
        ["beta", pytest.approx(expected["beta"])],
    ]

    truncated = _post(
        "SELECT row_index FROM memory.doc_rows "
        f"WHERE document_id = '{DOCUMENT_ID}' ORDER BY row_index"
    ).json()
    assert truncated["row_count"] == 1_000
    assert len(truncated["rows"]) == 1_000
    assert truncated["truncated"] is True

    async def stored_card_and_search():
        conn = await asyncpg.connect(db_url())
        try:
            card = await conn.fetchrow(
                f'SELECT metadata, embedding IS NOT NULL AS embedded FROM "{PG_SCHEMA}".memory_chunks '
                "WHERE namespace = $1 AND source_ref = $2",
                NAMESPACE_A,
                DOCUMENT_ID,
            )
        finally:
            await conn.close()
        hits = await search(
            SUMMARY,
            source="memory",
            rerank=False,
            namespaces=[NAMESPACE_A],
        )
        secret_hits = await search(
            f"{ROW_SECRET}-1999",
            source="memory",
            rerank=False,
            namespaces=[NAMESPACE_A],
        )
        return card, hits, secret_hits

    card, hits, secret_hits = asyncio.run(stored_card_and_search())
    metadata = (
        json.loads(card["metadata"]) if isinstance(card["metadata"], str) else card["metadata"]
    )
    assert card["embedded"] is True
    assert metadata["columns"] == ["group", "value", "secret"]
    assert metadata["table_rows_loaded"] is True
    assert any(hit.meta.get("columns") == ["group", "value", "secret"] for hit in hits)
    assert all(ROW_SECRET not in hit.text for hit in secret_hits)


def test_rls_isolates_same_document_id_across_namespaces():
    async def seed_other_namespace():
        conn = await asyncpg.connect(db_url())
        try:
            await conn.execute(
                f'INSERT INTO "{PG_SCHEMA}".doc_rows(namespace, document_id, row_index, data) '
                "VALUES ($1, $2, 0, $3::jsonb) ON CONFLICT (namespace, document_id, row_index) "
                "DO UPDATE SET data = EXCLUDED.data",
                NAMESPACE_B,
                DOCUMENT_ID,
                json.dumps({"scope": "namespace-b"}),
            )
        finally:
            await conn.close()

    asyncio.run(seed_other_namespace())
    a = _post(
        f"SELECT COUNT(*) FROM memory.doc_rows WHERE document_id = '{DOCUMENT_ID}'",
        NAMESPACE_A,
    )
    b = _post(
        f"SELECT data FROM memory.doc_rows WHERE document_id = '{DOCUMENT_ID}'",
        NAMESPACE_B,
    )
    assert a.json()["rows"] == [[2_000]]
    assert b.json()["rows"] == [[{"scope": "namespace-b"}]]


def test_insert_prefix_is_rejected():
    response = _post("INSERT INTO memory.doc_rows VALUES ('x', 'x', 0, '{}')")
    assert response.status_code == 400


def test_data_modifying_cte_is_rejected_in_read_only_transaction():
    response = _post(
        "WITH x AS (UPDATE memory.doc_rows SET data = '{}'::jsonb RETURNING *) SELECT * FROM x"
    )
    assert response.status_code == 400


def test_multiple_statements_are_rejected_by_extended_protocol():
    response = _post("SELECT 1; DROP TABLE memory.doc_rows")
    assert response.status_code == 400


def test_statement_timeout_maps_to_408():
    response = _post("SELECT pg_sleep(30)")
    assert response.status_code == 408


def test_set_config_namespace_jump_is_revoked():
    response = _post(f"SELECT set_config('app.namespace', '{NAMESPACE_B}', true)")
    assert response.status_code == 400
    assert "permission denied" in response.json()["error"].lower()


@pytest.mark.parametrize("table", ["memory_chunks", "api_keys", "jobs"])
def test_query_role_cannot_read_other_memory_tables(table):
    response = _post(f"SELECT * FROM memory.{table} LIMIT 1")
    assert response.status_code == 400
    assert "permission denied" in response.json()["error"].lower()


def test_query_role_has_only_the_reviewed_attributes_memberships_and_table_grant():
    async def inspect_role():
        conn = await asyncpg.connect(db_url())
        try:
            role = await conn.fetchrow(
                """
                SELECT oid, rolsuper, rolinherit, rolcreaterole, rolcreatedb, rolcanlogin,
                       rolreplication, rolbypassrls, rolconfig
                FROM pg_roles WHERE rolname = 'memory_tables_query'
                """
            )
            memberships = await conn.fetchval(
                "SELECT count(*) FROM pg_auth_members WHERE member = $1", role["oid"]
            )
            grants = await conn.fetch(
                """
                SELECT table_schema, table_name, privilege_type
                FROM information_schema.role_table_grants
                WHERE grantee = 'memory_tables_query'
                ORDER BY table_schema, table_name, privilege_type
                """
            )
            owned = await conn.fetchval(
                "SELECT count(*) FROM pg_class WHERE relowner = $1", role["oid"]
            )
            return role, memberships, [tuple(row) for row in grants], owned
        finally:
            await conn.close()

    role, memberships, grants, owned = asyncio.run(inspect_role())
    assert role["rolcanlogin"] is True
    assert role["rolsuper"] is False
    assert role["rolinherit"] is False
    assert role["rolcreaterole"] is False
    assert role["rolcreatedb"] is False
    assert role["rolreplication"] is False
    assert role["rolbypassrls"] is False
    assert memberships == 0
    assert grants == [("memory", "doc_rows", "SELECT")]
    assert owned == 0
    assert set(role["rolconfig"]) == {
        "lock_timeout=2s",
        "max_parallel_workers_per_gather=0",
        "statement_timeout=10s",
        "temp_file_limit=256MB",
        "track_activities=off",
        "work_mem=16MB",
    }


def test_namespace_document_query_uses_primary_key_index():
    async def explain():
        async with db.acquire_table_query() as conn:
            async with conn.transaction(readonly=True):
                await conn.execute(f"SET LOCAL app.namespace = '{NAMESPACE_A}'")
                return await conn.fetch(
                    "EXPLAIN SELECT row_index FROM memory.doc_rows "
                    f"WHERE namespace = '{NAMESPACE_A}' AND document_id = '{DOCUMENT_ID}' "
                    "AND row_index BETWEEN 0 AND 10"
                )

    plan = "\n".join(row[0] for row in asyncio.run(explain()))
    assert "doc_rows_pkey" in plan


def test_upsert_changed_csv_replaces_rows_and_loaded_same_hash_no_ops(monkeypatch, tmp_path):
    path = tmp_path / REPLACE_DOCUMENT_ID
    path.write_text("name,value\none,1\ntwo,2\n")

    async def summarize(*args, **kwargs):
        return {"summary": "Replace integration table.", "tags": []}

    async def embed(rows):
        _zero_embedding(rows)

    monkeypatch.setattr(ingest_api, "summarize_and_tag", summarize)
    monkeypatch.setattr(ingest_api, "_embed_rows", embed)
    first = _job(REPLACE_DOCUMENT_ID)
    asyncio.run(
        ingest_api.run_document_job(first, path, REPLACE_DOCUMENT_ID, "upsert", None, NAMESPACE_A)
    )
    assert first.status == "succeeded"

    path.write_text("name,value\nthree,3\n")
    changed = _job(REPLACE_DOCUMENT_ID)
    asyncio.run(
        ingest_api.run_document_job(changed, path, REPLACE_DOCUMENT_ID, "upsert", None, NAMESPACE_A)
    )
    assert changed.status == "succeeded"
    response = _post(
        f"SELECT data FROM memory.doc_rows WHERE document_id = '{REPLACE_DOCUMENT_ID}'"
    )
    assert response.json()["rows"] == [[{"name": "three", "value": "3"}]]

    unchanged = _job(REPLACE_DOCUMENT_ID)
    asyncio.run(
        ingest_api.run_document_job(
            unchanged, path, REPLACE_DOCUMENT_ID, "upsert", None, NAMESPACE_A
        )
    )
    assert unchanged.status == "no_op"


def test_document_deletion_removes_table_rows_and_namespace_check_sees_them():
    async def scenario():
        deleted = await ingest_api.delete_document_rows(REPLACE_DOCUMENT_ID, NAMESPACE_A)
        conn = await asyncpg.connect(db_url())
        try:
            remaining = await conn.fetchval(
                f'SELECT count(*) FROM "{PG_SCHEMA}".doc_rows '
                "WHERE namespace = $1 AND document_id = $2",
                NAMESPACE_A,
                REPLACE_DOCUMENT_ID,
            )
            await conn.execute(
                f'INSERT INTO "{PG_SCHEMA}".doc_rows(namespace, document_id, row_index, data) '
                "VALUES ($1, 'delete-only.csv', 0, '{}'::jsonb)",
                DELETE_NAMESPACE,
            )
        finally:
            await conn.close()
        with pytest.raises(namespaces.NamespaceNotEmptyError):
            await namespaces.delete_namespace(DELETE_NAMESPACE)
        return deleted, remaining

    deleted, remaining = asyncio.run(scenario())
    assert deleted == 1
    assert remaining == 0


def test_mcp_query_table_end_to_end(rest_in_process):
    from mcp.shared.memory import create_connected_server_and_client_session

    async def call():
        async with create_connected_server_and_client_session(mcp_server.mcp._mcp_server) as client:
            return await client.call_tool(
                "query_table",
                {
                    "sql": (
                        "SELECT COUNT(*) AS count FROM memory.doc_rows "
                        f"WHERE document_id = '{DOCUMENT_ID}'"
                    ),
                    "namespace": NAMESPACE_A,
                },
            )

    result = asyncio.run(call())
    assert not result.isError
    assert result.structuredContent["rows"] == [[2_000]]
