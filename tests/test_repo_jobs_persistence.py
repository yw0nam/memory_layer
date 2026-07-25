"""Repo job state must outlive the process that created it."""

from __future__ import annotations

import json

import pytest

from memory_base.serve import repos


@pytest.fixture()
def store(tmp_path, monkeypatch):
    path = tmp_path / "jobs.json"
    monkeypatch.setattr(repos, "JOBS_FILE", path)
    return path


def test_terminal_job_survives_a_restart(store):
    first = repos.RepoJobRegistry()
    job = first.create("repo", "ingest")
    job.touch(status="succeeded")
    first.persist()

    revived = repos.RepoJobRegistry().get(job.job_id)
    assert revived is not None
    assert revived.status == "succeeded"
    assert revived.name == "repo"
    assert revived.action == "ingest"


def test_interrupted_jobs_are_reported_as_failed_not_lost(store):
    first = repos.RepoJobRegistry()
    queued = first.create("a", "ingest")
    running = first.create("b", "remove")
    running.touch(status="running")
    first.persist()

    revived = repos.RepoJobRegistry()
    for job_id in (queued.job_id, running.job_id):
        job = revived.get(job_id)
        assert job is not None
        assert job.status == "failed"
        assert "restart" in (job.error or "")


def test_interrupted_jobs_do_not_hold_queue_capacity(store):
    first = repos.RepoJobRegistry(max_queued=1)
    first.create("a", "ingest")
    first.persist()

    assert repos.RepoJobRegistry(max_queued=1).has_capacity()


def test_unreadable_store_does_not_break_startup(store):
    store.write_text("{ not json")
    assert repos.RepoJobRegistry().get("anything") is None


def test_store_is_written_atomically_on_status_change(store):
    registry = repos.RepoJobRegistry()
    job = registry.create("repo", "ingest")
    job.touch(status="running")
    registry.persist()

    saved = {entry["job_id"]: entry for entry in json.loads(store.read_text())}
    assert saved[job.job_id]["status"] == "running"
    assert not list(store.parent.glob("*.tmp")), "temp file left behind"
