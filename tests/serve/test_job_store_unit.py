"""Contract for the shared Redis-backed job store.

The store is a durability mirror, not a source of truth: it must never be able
to take a job down, and a store outage must degrade to today's behaviour.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import pytest

from memory_base.serve import job_store

TERMINAL = frozenset({"succeeded", "failed", "no_op"})


@dataclass
class SampleJob:
    job_id: str
    status: str = "queued"
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


class FakeRedis:
    """Minimal stand-in recording writes and expiries."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.expiries: dict[str, int] = {}
        self.writes = 0

    async def set(self, key, value, ex=None):
        self.store[key] = value
        self.expiries[key] = ex
        self.writes += 1

    async def get(self, key):
        return self.store.get(key)


class BrokenRedis:
    async def set(self, *args, **kwargs):
        raise OSError("connection refused")

    async def get(self, *args, **kwargs):
        raise OSError("connection refused")


@pytest.fixture()
def fake(monkeypatch):
    client = FakeRedis()

    async def get_client():
        return client

    monkeypatch.setattr(job_store, "get_client", get_client)
    return client


@pytest.fixture()
def broken(monkeypatch):
    async def get_client():
        return BrokenRedis()

    monkeypatch.setattr(job_store, "get_client", get_client)


def test_round_trips_a_job(fake):
    job = SampleJob("j1", status="succeeded", created_at=1.0, updated_at=2.0)
    asyncio.run(job_store.save("document", job))

    revived = asyncio.run(job_store.load("document", "j1", SampleJob, TERMINAL))
    assert revived == job


def test_kinds_do_not_collide(fake):
    asyncio.run(job_store.save("repo", SampleJob("same", status="succeeded")))
    asyncio.run(job_store.save("document", SampleJob("same", status="failed")))

    repo_job = asyncio.run(job_store.load("repo", "same", SampleJob, TERMINAL))
    doc_job = asyncio.run(job_store.load("document", "same", SampleJob, TERMINAL))
    assert (repo_job.status, doc_job.status) == ("succeeded", "failed")


def test_unknown_job_is_a_miss(fake):
    assert asyncio.run(job_store.load("document", "ghost", SampleJob, TERMINAL)) is None


def test_saved_jobs_expire_rather_than_accumulate(fake):
    asyncio.run(job_store.save("repo", SampleJob("j1")))
    assert set(fake.expiries.values()) == {job_store.JOB_TTL_SECONDS}


def test_non_terminal_job_is_reported_failed_not_running(fake):
    asyncio.run(job_store.save("document", SampleJob("j1", status="running")))

    revived = asyncio.run(job_store.load("document", "j1", SampleJob, TERMINAL))
    assert revived.status == "failed"
    assert "restart" in (revived.error or "")


def test_no_op_counts_as_terminal_and_is_preserved(fake):
    asyncio.run(job_store.save("document", SampleJob("j1", status="no_op")))

    revived = asyncio.run(job_store.load("document", "j1", SampleJob, TERMINAL))
    assert revived.status == "no_op"


def test_save_never_raises_when_the_store_is_down(broken):
    asyncio.run(job_store.save("repo", SampleJob("j1")))


def test_load_degrades_to_a_miss_when_the_store_is_down(broken):
    assert asyncio.run(job_store.load("repo", "j1", SampleJob, TERMINAL)) is None


def test_absent_client_is_tolerated(monkeypatch):
    async def no_client():
        return None

    monkeypatch.setattr(job_store, "get_client", no_client)
    asyncio.run(job_store.save("repo", SampleJob("j1")))
    assert asyncio.run(job_store.load("repo", "j1", SampleJob, TERMINAL)) is None


def test_corrupt_payload_is_a_miss_not_a_crash(fake):
    fake.store[job_store.job_key("repo", "j1")] = "{ not json"
    assert asyncio.run(job_store.load("repo", "j1", SampleJob, TERMINAL)) is None


# ---- client construction ---------------------------------------------------


@pytest.fixture()
def fresh_client(monkeypatch):
    """Reset the module-level client cache so get_client() actually runs."""
    monkeypatch.setattr(job_store, "_client", None)
    monkeypatch.setattr(job_store, "_client_initialized", False)


def test_unusable_redis_url_yields_no_client_instead_of_raising(fresh_client, monkeypatch):
    """A scheme-less URL is a plausible config typo; it must not raise."""
    monkeypatch.setenv("REDIS_URL", "localhost:6379/0")
    assert asyncio.run(job_store.get_client()) is None


def test_save_and_load_stay_silent_on_an_unusable_url(fresh_client, monkeypatch):
    monkeypatch.setenv("REDIS_URL", "localhost:6379/0")
    asyncio.run(job_store.save("document", SampleJob("j1")))
    assert asyncio.run(job_store.load("document", "j1", SampleJob, TERMINAL)) is None


def test_client_cannot_wait_forever(fresh_client, monkeypatch):
    """load() runs inside the HTTP status route; an unbounded wait hangs it."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    client = asyncio.run(job_store.get_client())
    kwargs = client.connection_pool.connection_kwargs
    assert kwargs.get("socket_connect_timeout")
    assert kwargs.get("socket_timeout")


# ---- honesty about an unknown outcome --------------------------------------


def test_missing_terminal_record_does_not_assert_a_cause(fake):
    """A dropped terminal write looks identical to a restart — say so."""
    asyncio.run(job_store.save("document", SampleJob("j1", status="running")))

    revived = asyncio.run(job_store.load("document", "j1", SampleJob, TERMINAL))
    assert revived.status == "failed"
    assert "no terminal state" in (revived.error or "")


# ---- the shared JobRegistry -------------------------------------------------


@dataclass
class RegistryJob:
    job_id: str
    status: str = "queued"
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def touch(self, *, status: str | None = None) -> None:
        if status is not None:
            self.status = status
        self.updated_at = time.time()

    def mark_running(self) -> None:
        self.touch(status="running")

    def mark_succeeded(self) -> None:
        self.touch(status="succeeded")

    def mark_failed(self) -> None:
        self.touch(status="failed")


def _registry(**overrides):
    kwargs = {
        "kind": "sample",
        "job_cls": RegistryJob,
        "terminal_statuses": frozenset({"succeeded"}),
        "queue_full_message": "sample queue is full",
        "max_queued": 10,
        "max_concurrent": 1,
    }
    kwargs.update(overrides)
    return job_store.JobRegistry(**kwargs)


def test_register_enforces_max_queued_and_a_finished_job_frees_capacity():
    registry = _registry(max_queued=1)
    first = RegistryJob("j1")
    registry.register(first)
    with pytest.raises(OverflowError, match="sample queue is full"):
        registry.register(RegistryJob("j2"))

    first.touch(status="succeeded")  # updated_at=now, so it survives the TTL check below
    second = RegistryJob("j2")
    registry.register(second)
    assert set(registry.jobs) == {"j1", "j2"}


def test_cleanup_drops_expired_terminal_jobs_and_trims_beyond_max_completed():
    registry = _registry(ttl_seconds=10, max_completed=2)
    expired = RegistryJob("expired", status="succeeded", updated_at=0.0)
    registry.jobs[expired.job_id] = expired
    now = 1_000.0
    for index in range(3):
        job = RegistryJob(f"j{index}", status="succeeded", updated_at=now + index)
        registry.jobs[job.job_id] = job

    registry.cleanup(now + 5)

    assert "expired" not in registry.jobs
    assert list(registry.jobs) == ["j1", "j2"]


def test_max_concurrent_one_holds_the_second_job_until_the_first_finishes(fake):
    async def scenario():
        registry = _registry(max_concurrent=1)
        release = asyncio.Event()
        order: list[str] = []

        async def first_runner():
            order.append("first-start")
            await release.wait()
            order.append("first-end")

        async def second_runner():
            order.append("second")

        job1 = RegistryJob("j1")
        registry.register(job1)
        registry.start(job1, first_runner)
        await asyncio.sleep(0.01)
        assert job1.status == "running"

        job2 = RegistryJob("j2")
        registry.register(job2)
        registry.start(job2, second_runner)
        await asyncio.sleep(0.01)
        assert job2.status == "queued"
        assert order == ["first-start"]

        release.set()
        await asyncio.gather(*registry._tasks)
        assert order == ["first-start", "first-end", "second"]

    asyncio.run(scenario())


def test_a_raising_runner_leaves_the_job_failed_with_the_exception_text(fake):
    async def scenario():
        registry = _registry()
        job = RegistryJob("j1")
        registry.register(job)

        async def boom():
            raise RuntimeError("kaboom")

        registry.start(job, boom)
        await asyncio.gather(*registry._tasks)
        assert job.status == "failed"
        assert job.error == "kaboom"

    asyncio.run(scenario())


def test_get_falls_back_to_the_redis_mirror_for_a_job_it_no_longer_holds(fake):
    registry = _registry()
    mirrored = RegistryJob("ghost", status="succeeded", created_at=1.0, updated_at=2.0)
    asyncio.run(job_store.save("sample", mirrored))

    assert "ghost" not in registry.jobs
    revived = asyncio.run(registry.get("ghost"))
    assert revived == mirrored


def test_document_registry_treats_no_op_as_terminal():
    from memory_base.serve import ingest_api

    registry = ingest_api.JobRegistry(
        max_queued=5, max_concurrent=1, ttl_seconds=10, max_completed=5
    )
    job = registry.create("doc-1")
    job.status = "no_op"
    job.updated_at = 0.0

    registry.cleanup(100)

    assert job.job_id not in registry.jobs


def test_repo_registry_does_not_treat_no_op_as_terminal():
    from memory_base.serve import repos

    registry = repos.RepoJobRegistry(
        max_queued=5, max_concurrent=1, ttl_seconds=10, max_completed=5
    )
    job = registry.create("repo-1", "ingest")
    job.status = "no_op"
    job.updated_at = 0.0

    registry.cleanup(100)

    assert job.job_id in registry.jobs
