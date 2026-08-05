"""Unit tests for repo ownership: recording, exposure in listing, and the
admin-or-owner authorization gate on DELETE /repos/{name}."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from memory_base.serve import api, auth, repos


def _delete(path, api_key="test-key"):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": api_key},
        ) as client:
            return await client.delete(path)

    return asyncio.run(request())


def _identity(label, *, is_admin=False):
    return auth.KeyIdentity(
        key_id=f"{label}-hash", label=label, home="default", is_admin=is_admin, allowed=frozenset()
    )


def _auth_map(monkeypatch, mapping):
    async def fake_authenticate_request(plaintext_key):
        return mapping.get(plaintext_key)

    monkeypatch.setattr(auth, "authenticate_request", fake_authenticate_request)


class _CapturingAdmit:
    """Records whether a repo job was admitted, without touching the DB."""

    def __init__(self):
        self.called = False

    async def admit(self, **kwargs):
        self.called = True
        now = time.time()
        return repos.RepoJob(
            job_id="job-1",
            name=kwargs["name"],
            action=kwargs["action"],
            url=kwargs["url"],
            branch=kwargs["branch"],
            key_id=kwargs["key_id"],
            key_label=kwargs["key_label"],
            created_at=now,
            updated_at=now,
        )


def _own(cache, name, label):
    owners = cache / ".owners"
    owners.mkdir(exist_ok=True)
    (owners / name).write_text(label)


# ---- DELETE /repos/{name} authorization -----------------------------------


def test_owner_key_removes_its_repo_returns_202(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    (tmp_path / "repo").mkdir()
    _own(tmp_path, "repo", "owner-label")
    _auth_map(monkeypatch, {"owner-key": _identity("owner-label")})
    admitted = _CapturingAdmit()
    monkeypatch.setattr(repos.job_store, "admit_repo", admitted.admit)

    response = _delete("/repos/repo", api_key="owner-key")

    assert response.status_code == 202
    assert admitted.called


def test_different_non_admin_key_gets_403_and_no_job(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    (tmp_path / "repo").mkdir()
    _own(tmp_path, "repo", "owner-label")
    _auth_map(monkeypatch, {"intruder-key": _identity("intruder-label")})
    admitted = _CapturingAdmit()
    monkeypatch.setattr(repos.job_store, "admit_repo", admitted.admit)

    response = _delete("/repos/repo", api_key="intruder-key")

    assert response.status_code == 403
    assert not admitted.called


def test_admin_key_removes_an_owned_repo_returns_202(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    (tmp_path / "repo").mkdir()
    _own(tmp_path, "repo", "someone-else")
    admitted = _CapturingAdmit()
    monkeypatch.setattr(repos.job_store, "admit_repo", admitted.admit)

    response = _delete("/repos/repo")  # default test-key (conftest) is an admin

    assert response.status_code == 202
    assert admitted.called


def test_admin_key_removes_an_orphaned_repo_returns_202(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    (tmp_path / "repo").mkdir()
    admitted = _CapturingAdmit()
    monkeypatch.setattr(repos.job_store, "admit_repo", admitted.admit)

    response = _delete("/repos/repo")

    assert response.status_code == 202
    assert admitted.called


def test_non_admin_removal_of_an_unowned_repo_gets_403(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    (tmp_path / "repo").mkdir()
    _auth_map(monkeypatch, {"member-key": _identity("member-label")})
    admitted = _CapturingAdmit()
    monkeypatch.setattr(repos.job_store, "admit_repo", admitted.admit)

    response = _delete("/repos/repo", api_key="member-key")

    assert response.status_code == 403
    assert not admitted.called


# ---- ownership recording on ingest -----------------------------------------


def test_first_ingest_records_owner_and_reingest_does_not_transfer(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(repos, "CACHE_ROOT", cache)
    dest = cache / "repo"

    async def fake_clone(url, d, branch):
        d.mkdir(parents=True)
        (d / ".git").mkdir()

    async def fake_run_git(*args, cwd=None):
        return "true"

    async def fake_pull(d):
        return None

    async def fake_index():
        return None

    monkeypatch.setattr(repos, "clone", fake_clone)
    monkeypatch.setattr(repos, "_run_git", fake_run_git)
    monkeypatch.setattr(repos, "pull", fake_pull)
    monkeypatch.setattr(repos, "run_index", fake_index)

    asyncio.run(repos._run_ingest_job("https://example.com/repo.git", dest, None, "first-owner"))
    owner_file = cache / ".owners" / "repo"
    assert owner_file.read_text() == "first-owner"

    asyncio.run(repos._run_ingest_job("https://example.com/repo.git", dest, None, "second-owner"))
    assert owner_file.read_text() == "first-owner"


def test_failed_clone_records_no_owner(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(repos, "CACHE_ROOT", cache)
    dest = cache / "repo"

    async def failing_clone(url, d, branch):
        raise repos.RepoError("boom")

    monkeypatch.setattr(repos, "clone", failing_clone)

    with pytest.raises(repos.RepoError):
        asyncio.run(repos._run_ingest_job("https://example.com/repo.git", dest, None, "someone"))

    assert not (cache / ".owners" / "repo").exists()


def test_remove_job_deletes_the_owner_record(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    dest = cache / "repo"
    dest.mkdir(parents=True)
    _own(cache, "repo", "alice")
    monkeypatch.setattr(repos, "CACHE_ROOT", cache)

    async def fake_index():
        return None

    monkeypatch.setattr(repos, "run_index", fake_index)
    asyncio.run(repos._run_remove_job(dest))

    assert not (cache / ".owners" / "repo").exists()
    assert not dest.exists()


# ---- GET /repos exposes ownership ------------------------------------------


def test_list_repos_includes_owner_field(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    (cache / "repo").mkdir(parents=True)
    (cache / "repo2").mkdir(parents=True)
    _own(cache, "repo", "alice")
    monkeypatch.setattr(repos, "CACHE_ROOT", cache)

    async def no_counts():
        return {}

    monkeypatch.setattr(repos, "_repo_chunk_counts", no_counts)

    listed = {entry["name"]: entry for entry in asyncio.run(repos.list_repos())}
    assert listed["repo"]["owner"] == "alice"
    assert listed["repo2"]["owner"] is None
