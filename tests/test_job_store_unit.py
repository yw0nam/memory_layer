"""Contract for the shared Redis-backed job store.

The store is a durability mirror, not a source of truth: it must never be able
to take a job down, and a store outage must degrade to today's behaviour.
"""

from __future__ import annotations

import asyncio
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
