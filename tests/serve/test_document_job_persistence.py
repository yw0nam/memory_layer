"""Document ingest jobs must outlive the process, at a bounded write cost.

Progress counters advance per chunk; only status transitions are durable, so a
large document must not turn into a write per chunk.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from memory_base.serve import ingest_api, job_store


class Recorder:
    """Stands in for the transport only, so the real serialisation still runs.

    Payloads are stored as JSON, exactly as Redis would hold them, and reads go
    through the production `load` — a fake that returned the live object would
    never cross the asdict/from-dict round trip.
    """

    def __init__(self):
        self.stored: dict[str, str] = {}
        self.writes: list[str] = []

    async def set(self, key, value, ex=None):
        self.writes.append(json.loads(value)["status"])
        self.stored[key] = value

    async def get(self, key):
        return self.stored.get(key)


@pytest.fixture()
def store(monkeypatch):
    recorder = Recorder()

    async def get_client():
        return recorder

    monkeypatch.setattr(job_store, "get_client", get_client)
    return recorder


async def _drain(registry):
    if registry._tasks:
        await asyncio.gather(*list(registry._tasks))


def test_completed_job_is_readable_after_a_restart(store):
    async def scenario():
        first = ingest_api.JobRegistry()
        job = first.create("doc-1")
        first.start(job, lambda: _succeed(job))
        await _drain(first)

        revived = await ingest_api.JobRegistry().get(job.job_id)
        assert revived is not None
        assert revived.status == "succeeded"
        assert revived.document_id == "doc-1"

    async def _succeed(job):
        job.touch(status="succeeded", stage="done")

    asyncio.run(scenario())


def test_no_op_result_survives_too(store):
    """`start()` has no success branch — the runner sets no_op itself."""

    async def scenario():
        first = ingest_api.JobRegistry()
        job = first.create("doc-1")
        first.start(job, lambda: _no_op(job))
        await _drain(first)

        revived = await ingest_api.JobRegistry().get(job.job_id)
        assert revived.status == "no_op"

    async def _no_op(job):
        job.touch(status="no_op", stage="done")

    asyncio.run(scenario())


def test_interrupted_job_is_reported_failed_not_missing(store):
    async def scenario():
        first = ingest_api.JobRegistry()
        job = first.create("doc-1")
        job.touch(status="running")
        await job_store.save("document", job)

        revived = await ingest_api.JobRegistry().get(job.job_id)
        assert revived.status == "failed"
        assert "restart" in (revived.error or "")

    asyncio.run(scenario())


def test_progress_updates_do_not_write(store):
    """A 2000-chunk document must not become 2000 round trips."""

    async def scenario():
        registry = ingest_api.JobRegistry()
        job = registry.create("doc-1")
        store.writes.clear()
        for _ in range(10):
            job.chunks_done += 1
            job.touch(stage="enriching")
        assert store.writes == []

    asyncio.run(scenario())


def test_a_whole_job_costs_three_writes(store):
    async def scenario():
        registry = ingest_api.JobRegistry()
        job = registry.create("doc-1")
        registry.start(job, lambda: _succeed(job))
        await _drain(registry)
        assert store.writes == ["queued", "running", "succeeded"]

    async def _succeed(job):
        job.chunks_done += 1
        job.touch(stage="embedding")
        job.touch(status="succeeded", stage="done")

    asyncio.run(scenario())


def test_failure_is_persisted(store):
    async def scenario():
        registry = ingest_api.JobRegistry()
        job = registry.create("doc-1")
        registry.start(job, _boom)
        await _drain(registry)

        revived = await ingest_api.JobRegistry().get(job.job_id)
        assert revived.status == "failed"
        assert "boom" in (revived.error or "")

    async def _boom():
        raise RuntimeError("boom")

    asyncio.run(scenario())


def test_unknown_job_is_still_a_miss(store):
    assert asyncio.run(ingest_api.JobRegistry().get("ghost")) is None


def test_store_outage_does_not_break_the_job(monkeypatch):
    """Runs through the real job_store against an unreachable client."""

    class BrokenRedis:
        async def set(self, *args, **kwargs):
            raise OSError("connection refused")

        async def get(self, *args, **kwargs):
            raise OSError("connection refused")

    async def get_client():
        return BrokenRedis()

    monkeypatch.setattr(job_store, "get_client", get_client)

    async def scenario():
        registry = ingest_api.JobRegistry()
        job = registry.create("doc-1")
        registry.start(job, lambda: _succeed(job))
        await _drain(registry)
        assert job.status == "succeeded"
        assert await registry.get(job.job_id) is job

    async def _succeed(job):
        job.touch(status="succeeded", stage="done")

    asyncio.run(scenario())


def test_a_misconfigured_store_cannot_swallow_the_document(monkeypatch):
    """A scheme-less REDIS_URL must not stop the pipeline from running.

    The first store call happens before the runner; if it raises there, the
    document is never converted while the status endpoint still says queued.
    """
    monkeypatch.setattr(job_store, "_client", None)
    monkeypatch.setattr(job_store, "_client_initialized", False)
    monkeypatch.setenv("REDIS_URL", "localhost:6379/0")

    async def scenario():
        ran = []
        registry = ingest_api.JobRegistry()
        job = registry.create("doc-1")

        async def runner():
            ran.append(job.job_id)
            job.touch(status="succeeded", stage="done")

        registry.start(job, runner)
        await _drain(registry)
        assert ran == [job.job_id]
        assert job.status == "succeeded"

    asyncio.run(scenario())
