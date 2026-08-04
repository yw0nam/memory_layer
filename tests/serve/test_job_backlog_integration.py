"""Race-sensitive Postgres integration tests for job admission and dispatch."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager

import asyncpg
import httpx
import pytest

from memory_base.core import db
from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.core.schema import ensure_schema
from memory_base.serve import api, auth, ingest_api, job_store

pytestmark = pytest.mark.integration


@asynccontextmanager
async def connections(count=2):
    opened = [await asyncpg.connect(db_url()) for _ in range(count)]
    try:
        await ensure_schema(opened[0])
        yield opened
    finally:
        for connection in opened:
            await connection.close()


async def insert_document(
    connection,
    marker,
    key_id,
    created,
    *,
    status="queued",
    namespace="default",
    document_id=None,
):
    job_id = f"it-{marker}-{uuid.uuid4().hex}"
    await connection.execute(
        f'''INSERT INTO "{PG_SCHEMA}".jobs
        (job_id, kind, status, key_id, key_label, created_at, updated_at,
         namespace, document_id, mode, filename, spool_path, stage)
        VALUES ($1, 'document', $2, $3, 'label', to_timestamp($4), to_timestamp($4),
                $5, $6, 'force', 'doc.md', $7, 'queued')''',
        job_id,
        status,
        key_id,
        created,
        namespace,
        document_id or job_id,
        f"/tmp/{job_id}.md",
    )
    return job_id


async def cleanup(connection, marker):
    await connection.execute(
        f'DELETE FROM "{PG_SCHEMA}".jobs WHERE job_id LIKE $1', f"it-{marker}-%"
    )


def test_fair_claim_prefers_the_key_with_fewer_active_jobs_and_then_oldest_key():
    async def scenario():
        marker = uuid.uuid4().hex
        async with connections() as (first, second):
            try:
                await insert_document(first, marker, "key-a", 1, status="running")
                await insert_document(first, marker, "key-a", 2, status="running")
                await insert_document(first, marker, "key-a", 3)
                key_b = await insert_document(first, marker, "key-b", 4)
                claimed = await job_store.claim_job("document", connection=second)
                assert claimed.job_id == key_b

                await first.execute(
                    f'''UPDATE "{PG_SCHEMA}".jobs SET status = 'succeeded'
                    WHERE job_id LIKE $1''',
                    f"it-{marker}-%",
                )
                oldest = await insert_document(first, marker, "key-c", 10)
                await insert_document(first, marker, "key-d", 11)
                claimed = await job_store.claim_job("document", connection=second)
                assert claimed.job_id == oldest
            finally:
                await cleanup(first, marker)

    asyncio.run(scenario())


def test_same_document_claims_on_independent_connections_never_overlap():
    async def scenario():
        marker = uuid.uuid4().hex
        async with connections() as (first, second):
            try:
                await insert_document(first, marker, "key-a", 1, document_id="same-document")
                await insert_document(first, marker, "key-b", 2, document_id="same-document")
                claims = await asyncio.gather(
                    job_store.claim_job("document", connection=first),
                    job_store.claim_job("document", connection=second),
                )
                assert sum(claim is not None for claim in claims) == 1
            finally:
                await cleanup(first, marker)

    asyncio.run(scenario())


def test_repo_claim_stays_serial_even_with_two_independent_claimers():
    async def scenario():
        marker = uuid.uuid4().hex
        async with connections() as (first, second):
            try:
                for created in (1, 2):
                    await first.execute(
                        f'''INSERT INTO "{PG_SCHEMA}".jobs
                        (job_id, kind, status, key_id, key_label, created_at, updated_at,
                         name, action)
                        VALUES ($1, 'repo', 'queued', 'key', 'label', to_timestamp($2),
                                to_timestamp($2), $3, 'remove')''',
                        f"it-{marker}-{created}",
                        created,
                        f"repo-{created}",
                    )
                claims = await asyncio.gather(
                    job_store.claim_job("repo", connection=first),
                    job_store.claim_job("repo", connection=second),
                )
                assert sum(claim is not None for claim in claims) == 1
                assert await job_store.claim_job("repo", connection=second) is None
            finally:
                await cleanup(first, marker)

    asyncio.run(scenario())


def test_concurrent_document_admission_stops_exactly_at_the_cap(tmp_path, monkeypatch):
    async def scenario():
        marker = uuid.uuid4().hex
        monkeypatch.setattr(job_store, "INGEST_BACKLOG_PER_KEY", 3)
        monkeypatch.setattr(job_store, "INGEST_BACKLOG_MAX", 100)
        async with connections() as (first, second):
            try:
                await insert_document(first, marker, marker, 1)
                paths = [tmp_path / f"{index}.md" for index in range(2)]
                for path in paths:
                    path.write_text("content")

                async def admit(connection, index):
                    return await job_store.admit_document(
                        connection=connection,
                        job_id=f"it-{marker}-admit-{index}",
                        key_id=marker,
                        key_label="same-label",
                        namespace="default",
                        document_id=f"doc-{index}",
                        origin=None,
                        mode="force",
                        filename=f"{index}.md",
                        spool_path=str(paths[index]),
                    )

                results = await asyncio.gather(
                    admit(first, 0), admit(second, 1), return_exceptions=True
                )
                assert sum(not isinstance(result, Exception) for result in results) == 2

                extra = tmp_path / "extra.md"
                extra.write_text("content")
                with pytest.raises(job_store.BacklogFullError, match="per-key"):
                    await job_store.admit_document(
                        connection=first,
                        job_id=f"it-{marker}-extra",
                        key_id=marker,
                        key_label="same-label",
                        namespace="default",
                        document_id="extra",
                        origin=None,
                        mode="force",
                        filename="extra.md",
                        spool_path=str(extra),
                    )
            finally:
                await cleanup(first, marker)

    asyncio.run(scenario())


def test_repo_cap_counts_one_running_plus_nine_queued(monkeypatch):
    async def scenario():
        marker = uuid.uuid4().hex
        monkeypatch.setattr(job_store, "REPO_MAX_QUEUED", 10)
        async with connections(1) as (connection,):
            try:
                for index in range(10):
                    await connection.execute(
                        f'''INSERT INTO "{PG_SCHEMA}".jobs
                        (job_id, kind, status, key_id, key_label, created_at, updated_at,
                         name, action)
                        VALUES ($1, 'repo', $2, $3, 'label', now(), now(), $4, 'remove')''',
                        f"it-{marker}-{index}",
                        "running" if index == 0 else "queued",
                        marker,
                        f"repo-{index}",
                    )
                with pytest.raises(job_store.BacklogFullError, match="repo job queue is full"):
                    await job_store.admit_repo(
                        connection=connection,
                        job_id=f"it-{marker}-extra",
                        key_id=marker,
                        key_label="label",
                        name="extra",
                        action="remove",
                        url=None,
                        branch=None,
                    )
            finally:
                await cleanup(connection, marker)

    asyncio.run(scenario())


def test_global_document_cap_has_a_distinct_error(monkeypatch, tmp_path):
    async def scenario():
        marker = uuid.uuid4().hex
        monkeypatch.setattr(job_store, "INGEST_BACKLOG_PER_KEY", 100)
        monkeypatch.setattr(job_store, "INGEST_BACKLOG_MAX", 2)
        async with connections(1) as (connection,):
            try:
                await insert_document(connection, marker, f"{marker}-a", 1)
                await insert_document(connection, marker, f"{marker}-b", 2)
                spool = tmp_path / "global.md"
                spool.write_text("content")
                with pytest.raises(job_store.BacklogFullError, match="global"):
                    await job_store.admit_document(
                        connection=connection,
                        job_id=f"it-{marker}-global",
                        key_id=f"{marker}-c",
                        key_label="label",
                        namespace="default",
                        document_id="global",
                        origin=None,
                        mode="force",
                        filename="global.md",
                        spool_path=str(spool),
                    )
            finally:
                await cleanup(connection, marker)

    asyncio.run(scenario())


def test_concurrent_repo_admission_at_the_boundary_admits_only_one(monkeypatch):
    async def scenario():
        marker = uuid.uuid4().hex
        monkeypatch.setattr(job_store, "REPO_MAX_QUEUED", 10)
        async with connections() as (first, second):
            try:
                for index in range(9):
                    await first.execute(
                        f'''INSERT INTO "{PG_SCHEMA}".jobs
                        (job_id, kind, status, key_id, key_label, created_at, updated_at,
                         name, action)
                        VALUES ($1, 'repo', 'queued', $2, 'label', now(), now(), $3, 'remove')''',
                        f"it-{marker}-{index}",
                        marker,
                        f"repo-{index}",
                    )

                async def admit(connection, suffix):
                    return await job_store.admit_repo(
                        connection=connection,
                        job_id=f"it-{marker}-{suffix}",
                        key_id=marker,
                        key_label="label",
                        name=f"repo-{suffix}",
                        action="remove",
                        url=None,
                        branch=None,
                    )

                results = await asyncio.gather(
                    admit(first, "a"), admit(second, "b"), return_exceptions=True
                )
                assert sum(not isinstance(result, Exception) for result in results) == 1
                assert (
                    sum(isinstance(result, job_store.BacklogFullError) for result in results) == 1
                )
            finally:
                await cleanup(first, marker)

    asyncio.run(scenario())


def test_fifty_sequential_uploads_are_durably_accepted_and_reach_terminal(monkeypatch, tmp_path):
    async def scenario():
        key_id = f"it-fifty-{uuid.uuid4().hex}"
        identity = auth.KeyIdentity(
            key_id=key_id,
            label="duplicate-label",
            home="default",
            is_admin=False,
            allowed=frozenset({"default"}),
        )

        async def authenticate(plaintext_key):
            return identity

        async def namespace_exists(name):
            return name == "default"

        spool = tmp_path / "spool"
        monkeypatch.setattr(auth, "authenticate_request", authenticate)
        monkeypatch.setattr(ingest_api.namespaces, "namespace_exists", namespace_exists)
        monkeypatch.setattr(ingest_api, "INGEST_SPOOL", spool)
        job_ids = []
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": "key"},
        ) as client:
            for index in range(50):
                response = await client.post(
                    "/ingest/document",
                    data={"document_id": f"doc-{index}", "mode": "force"},
                    files={"file": (f"original-{index}.md", b"small document")},
                )
                assert response.status_code == 202
                job_ids.append(response.json()["job_id"])

        connection = await asyncpg.connect(db_url())
        try:
            for _ in range(50):
                claimed = await job_store.claim_job("document", connection=connection)
                assert claimed is not None
                await connection.execute(
                    f'''UPDATE "{PG_SCHEMA}".jobs
                    SET status = 'succeeded', stage = 'done', updated_at = now()
                    WHERE job_id = $1''',
                    claimed.job_id,
                )
            rows = await connection.fetch(
                f'''SELECT status, filename FROM "{PG_SCHEMA}".jobs
                WHERE job_id = ANY($1::text[])''',
                job_ids,
            )
            assert len(rows) == 50
            assert {row["status"] for row in rows} == {"succeeded"}
            assert {row["filename"] for row in rows} == {
                f"original-{index}.md" for index in range(50)
            }
            await connection.execute(
                f'DELETE FROM "{PG_SCHEMA}".jobs WHERE job_id = ANY($1::text[])', job_ids
            )
        finally:
            await connection.close()
            await db.close_pool()

    asyncio.run(scenario())


def test_startup_recovery_requeues_valid_spool_and_fails_missing_spool(tmp_path):
    async def scenario():
        marker = uuid.uuid4().hex
        spool = tmp_path / "spool"
        spool.mkdir()
        valid_path = spool / "valid.md"
        valid_path.write_text("content")
        missing_path = spool / "missing.md"
        async with connections() as (connection, second):
            try:
                valid = await insert_document(
                    connection, marker, marker, 1, status="running", document_id="valid"
                )
                missing = await insert_document(
                    connection, marker, marker, 2, status="running", document_id="missing"
                )
                await connection.execute(
                    f'''UPDATE "{PG_SCHEMA}".jobs
                    SET spool_path = $2, filename = 'Original.MD', stage = 'enriching',
                        chunks_total = 1, chunks_done = 9
                    WHERE job_id = $1''',
                    valid,
                    str(valid_path),
                )
                await connection.execute(
                    f'UPDATE "{PG_SCHEMA}".jobs SET spool_path = $2 WHERE job_id = $1',
                    missing,
                    str(missing_path),
                )
                await job_store.recover_and_prune(spool)
                recovered = await connection.fetchrow(
                    f'SELECT status, stage, filename FROM "{PG_SCHEMA}".jobs WHERE job_id = $1',
                    valid,
                )
                failed = await connection.fetchrow(
                    f'SELECT status, error FROM "{PG_SCHEMA}".jobs WHERE job_id = $1', missing
                )
                assert dict(recovered) == {
                    "status": "queued",
                    "stage": "queued",
                    "filename": "Original.MD",
                }
                assert failed["status"] == "failed"
                assert "spool file is missing" in failed["error"]
                claimed = await job_store.claim_job("document", connection=second)
                assert claimed.job_id == valid
                assert claimed.filename == "Original.MD"
                assert claimed.chunks_done == 0
                assert claimed.chunks_total == 0
            finally:
                await cleanup(connection, marker)
                await db.close_pool()

    asyncio.run(scenario())
