"""Unit contract for the durable Postgres job backlog."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest

from memory_base.serve import api, auth, ingest_api, job_store, repos


def test_document_row_keeps_internal_fields_out_of_the_response():
    now = time.time()
    job = ingest_api.IngestJob(
        job_id="job-1",
        document_id="guide.md",
        namespace="default",
        origin="manual",
        mode="upsert",
        filename="Guide.MD",
        spool_path="/spool/job-1.md",
        key_id="hash",
        key_label="duplicate-label",
        created_at=now,
        updated_at=now,
    )

    assert set(job.response()) == {
        "job_id",
        "document_id",
        "status",
        "stage",
        "chunks_total",
        "chunks_done",
        "chunks_dropped",
        "rows_written",
        "enrichment_retries",
        "content_hash",
        "error",
        "created_at",
        "updated_at",
    }


def test_repo_row_keeps_persisted_runner_fields_out_of_the_response():
    now = time.time()
    job = repos.RepoJob(
        job_id="job-1",
        name="repo",
        action="ingest",
        url="https://example.com/repo.git",
        branch="main",
        key_id="hash",
        key_label="label",
        created_at=now,
        updated_at=now,
    )

    assert set(job.response()) == {
        "job_id",
        "name",
        "action",
        "status",
        "error",
        "created_at",
        "updated_at",
    }


def test_worker_loop_retries_after_a_claim_error(monkeypatch):
    events = []
    stop = asyncio.Event()

    async def claim(kind):
        events.append(kind)
        if len(events) == 1:
            raise RuntimeError("temporary database fault")
        stop.set()
        return None

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(job_store.asyncio, "sleep", no_sleep)
    asyncio.run(job_store.worker_loop("document", stop=stop, claim=claim))
    assert events == ["document", "document"]


def test_recovered_remove_tolerates_a_missing_checkout(monkeypatch, tmp_path):
    indexed = []
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)

    async def fake_index():
        indexed.append(True)

    monkeypatch.setattr(repos, "run_index", fake_index)
    asyncio.run(repos._run_remove_job(tmp_path / "missing"))
    assert indexed == [True]


def test_recovered_ingest_reclones_a_partial_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    destination = tmp_path / "partial"
    destination.mkdir()
    (destination / "incomplete").write_text("partial")
    calls = []

    async def fake_clone(url, dest, branch):
        calls.append(("clone", url, dest, branch))

    async def forbidden_pull(dest):
        raise AssertionError("partial checkout must not be pulled")

    async def fake_index():
        calls.append(("index",))

    monkeypatch.setattr(repos, "clone", fake_clone)
    monkeypatch.setattr(repos, "pull", forbidden_pull)
    monkeypatch.setattr(repos, "run_index", fake_index)
    asyncio.run(repos._run_ingest_job("https://example.com/repo.git", destination, "main", "label"))
    assert calls == [
        ("clone", "https://example.com/repo.git", destination, "main"),
        ("index",),
    ]
    assert not destination.exists()


def test_document_runner_revalidates_namespace(monkeypatch, tmp_path):
    upload = tmp_path / "job.md"
    upload.write_text("content")

    async def missing(namespace):
        return False

    monkeypatch.setattr(ingest_api.namespaces, "namespace_exists", missing)
    job = ingest_api.IngestJob.for_document(
        job_id="job-1",
        document_id="guide.md",
        namespace="deleted",
        origin=None,
        mode="force",
        filename="guide.md",
        spool_path=str(upload),
        key_id="hash",
        key_label="label",
    )
    with pytest.raises(RuntimeError, match="namespace.*deleted"):
        asyncio.run(ingest_api.run_document_job(job))


def test_startup_prune_removes_terminal_and_orphan_spool_files(monkeypatch, tmp_path):
    terminal = tmp_path / "terminal.md"
    orphan = tmp_path / "orphan.md"
    active = tmp_path / "active.md"
    for path in (terminal, orphan, active):
        path.write_text(path.name)

    async def fake_rows():
        return [
            {"spool_path": str(terminal), "status": "succeeded"},
            {"spool_path": str(active), "status": "queued"},
        ]

    monkeypatch.setattr(job_store, "document_spool_rows", fake_rows)
    asyncio.run(job_store.prune_spool(Path(tmp_path)))
    assert not terminal.exists()
    assert not orphan.exists()
    assert active.exists()


def test_listing_scopes_non_admin_and_leaves_admin_unrestricted(monkeypatch):
    calls = []

    async def fake_list(*, namespaces, origin, status):
        calls.append((namespaces, origin, status))
        return []

    monkeypatch.setattr(job_store, "list_document_jobs", fake_list)

    async def request(identity, query=""):
        async def authenticate(plaintext_key):
            return identity

        monkeypatch.setattr(auth, "authenticate_request", authenticate)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": "key"},
        ) as client:
            return await client.get(f"/ingest/jobs{query}")

    member = auth.KeyIdentity(
        key_id="member-hash",
        label="same-label",
        home="default",
        is_admin=False,
        allowed=frozenset({"default", "team"}),
    )
    admin = auth.KeyIdentity(
        key_id="admin-hash",
        label="same-label",
        home="default",
        is_admin=True,
        allowed=frozenset(),
    )
    assert asyncio.run(request(member, "?origin=manual&status=failed")).json() == {"jobs": []}
    assert asyncio.run(request(admin)).json() == {"jobs": []}
    assert calls == [(["default", "team"], "manual", "failed"), (None, None, None)]


def test_lifespan_initializes_then_starts_and_stops_workers_before_pool_close(monkeypatch):
    events = []

    async def initialize():
        events.append("initialize")

    def start():
        events.append("start")
        return ["workers"]

    async def stop(tasks):
        events.append(("stop", tasks))

    async def close():
        events.append("close")

    monkeypatch.setattr(job_store, "initialize", initialize)
    monkeypatch.setattr(job_store, "start_workers", start)
    monkeypatch.setattr(job_store, "stop_workers", stop)
    monkeypatch.setattr(api.db, "close_pool", close)

    async def scenario():
        async with api.lifespan(api.app):
            events.append("serving")

    asyncio.run(scenario())
    assert events == ["initialize", "start", "serving", ("stop", ["workers"]), "close"]
