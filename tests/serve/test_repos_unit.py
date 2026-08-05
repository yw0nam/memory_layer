"""Unit tests for URL-driven multi-repo ingestion: validation, naming,
job registry, list parsing (offline local git repos), and REST routes."""

from __future__ import annotations

import asyncio
import subprocess
import time
from contextlib import asynccontextmanager

import httpx
import pytest

from memory_base.serve import api, repos


def _post(path, **kwargs):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": "test-key"},
        ) as client:
            return await client.post(path, **kwargs)

    return asyncio.run(request())


def _delete(path):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": "test-key"},
        ) as client:
            return await client.delete(path)

    return asyncio.run(request())


# ---- URL validation (trust boundary) --------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/owner/repo.git",
        "http://example.com/owner/repo",
    ],
)
def test_validate_repo_url_accepts_supported_forms(url):
    assert repos.validate_repo_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "ftp://example.com/repo.git",
        "javascript:alert(1)",
        "not a url",
        "https://github.com/owner/repo.git\n--upload-pack=x",
        "--upload-pack=/bin/sh",
        "https://exa mple.com/repo.git",
        "ssh://git@github.com/owner/repo.git",
        "git@github.com:owner/repo.git",
        "https://user:token@github.com/owner/repo.git",
        "https://token@github.com/owner/repo.git",
    ],
)
def test_validate_repo_url_rejects_bad_forms(url):
    with pytest.raises(ValueError):
        repos.validate_repo_url(url)


# ---- name derivation + path traversal -------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/owner/repo.git", "repo"),
        ("https://github.com/owner/repo", "repo"),
        ("https://github.com/owner/my-repo.git", "my-repo"),
        ("https://host/owner/repo.git/", "repo"),
    ],
)
def test_derive_repo_name_from_url(url, expected):
    assert repos.derive_repo_name(url, None) == expected


def test_derive_repo_name_uses_explicit_name():
    assert repos.derive_repo_name("https://x/y.git", "custom_name") == "custom_name"


@pytest.mark.parametrize("name", ["..", "../etc", "a/b", "a\\b", "bad:name", "with space"])
def test_derive_repo_name_rejects_traversal_and_bad_chars(name):
    with pytest.raises(ValueError):
        repos.derive_repo_name("https://x/y.git", name)


# ---- list_repos parsing against local temp git repos (offline) ------------


def _make_git_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    (path / "main.py").write_text("print('hi')\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True, env={**env})


def test_list_repos_parses_local_clones(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    _make_git_repo(origin)
    cache = tmp_path / "cache"
    cache.mkdir()
    subprocess.run(["git", "clone", "-q", str(origin), str(cache / "repo")], check=True)

    monkeypatch.setattr(repos, "CACHE_ROOT", cache)

    async def no_counts():
        return {"repo": 7}

    monkeypatch.setattr(repos, "_repo_chunk_counts", no_counts)

    listed = asyncio.run(repos.list_repos())
    assert len(listed) == 1
    entry = listed[0]
    assert entry["name"] == "repo"
    assert entry["url"] == str(origin)
    assert entry["branch"] == "main"
    assert len(entry["head"]) >= 7
    assert entry["chunks"] == 7


def test_list_repos_survives_db_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", "postgres://fake/db")
    origin = tmp_path / "origin"
    _make_git_repo(origin)
    cache = tmp_path / "cache"
    cache.mkdir()
    subprocess.run(["git", "clone", "-q", str(origin), str(cache / "repo")], check=True)

    monkeypatch.setattr(repos, "CACHE_ROOT", cache)

    @asynccontextmanager
    async def unreachable():
        raise OSError("connection refused")
        yield

    monkeypatch.setattr(repos.db, "acquire", unreachable)

    listed = asyncio.run(repos.list_repos())
    assert [entry["name"] for entry in listed] == ["repo"]
    assert listed[0]["chunks"] == 0


# ---- index command ---------------------------------------------------------


def test_index_command_skips_dependency_sync():
    command = repos._index_command()
    assert command[:2] == ["uv", "run"]
    assert "--no-sync" in command
    assert command[-1] == repos.CODE_APP


# ---- RepoJob.response() shape ---------------------------------------------


def test_repo_job_response_shape():
    now = time.time()
    job = repos.RepoJob(
        job_id="id",
        name="repo",
        action="ingest",
        url="https://example.com/repo.git",
        branch=None,
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


# ---- REST routes -----------------------------------------------------------


class AcceptingBacklog:
    def __init__(self):
        self.job = None

    async def admit(self, **kwargs):
        now = time.time()
        self.job = repos.RepoJob(
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
        return self.job


def test_post_repos_rejects_bad_url_with_400(monkeypatch):
    response = _post("/repos", json={"url": "ftp://bad/repo.git"})
    assert response.status_code == 400
    assert set(response.json()) == {"error"}


def test_post_repos_returns_202_and_shape(monkeypatch):
    fake = AcceptingBacklog()
    monkeypatch.setattr(repos.job_store, "admit_repo", fake.admit)
    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})
    assert response.status_code == 202
    body = response.json()
    assert body == {
        "job_id": "job-1",
        "name": "repo",
        "status": "queued",
        "status_url": "/repos/jobs/job-1",
    }
    assert fake.job.url == "https://github.com/owner/repo.git"


def test_post_repos_returns_429_when_full(monkeypatch):
    async def full(**kwargs):
        raise repos.job_store.BacklogFullError("repo job queue is full")

    monkeypatch.setattr(repos.job_store, "admit_repo", full)
    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})
    assert response.status_code == 429


def test_delete_unknown_repo_returns_404(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    response = _delete("/repos/ghost")
    assert response.status_code == 404


def test_delete_existing_repo_returns_202(monkeypatch, tmp_path):
    (tmp_path / "repo").mkdir()
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    fake = AcceptingBacklog()
    monkeypatch.setattr(repos.job_store, "admit_repo", fake.admit)
    response = _delete("/repos/repo")
    assert response.status_code == 202
    assert response.json()["name"] == "repo"
    assert fake.job.action == "remove"
