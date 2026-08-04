"""Race-sensitive Postgres integration tests for job admission and dispatch."""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import pytest

from memory_base.core.config import PG_SCHEMA, db_url
from memory_base.core.schema import ensure_schema
from memory_base.serve import job_store

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
