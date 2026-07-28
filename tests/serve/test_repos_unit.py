"""Unit tests for URL-driven multi-repo ingestion: validation, naming,
job registry, list parsing (offline local git repos), and REST routes."""

from __future__ import annotations

import asyncio
import subprocess
import time

import httpx
import pytest

from memory_base.serve import api, repos


def _post(path, **kwargs):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.post(path, **kwargs)

    return asyncio.run(request())


def _delete(path):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
        ) as client:
            return await client.delete(path)

    return asyncio.run(request())


# ---- URL validation (trust boundary) --------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/owner/repo.git",
        "http://example.com/owner/repo",
        "ssh://git@github.com/owner/repo.git",
        "git@github.com:owner/repo.git",
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
        ("git@github.com:owner/my-repo.git", "my-repo"),
        ("ssh://git@host/owner/repo.git/", "repo"),
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

    async def unreachable(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(repos.asyncpg, "connect", unreachable)

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
    job = repos.RepoJob("id", "repo", "ingest", created_at=now, updated_at=now)
    assert set(job.response()) == {
        "job_id",
        "name",
        "action",
        "status",
        "error",
        "created_at",
        "updated_at",
    }


def test_repo_job_registry_queue_bound_and_ttl():
    registry = repos.RepoJobRegistry(max_queued=1, ttl_seconds=10, max_completed=2)
    first = registry.create("a", "ingest")
    with pytest.raises(OverflowError):
        registry.create("b", "ingest")
    first.status = "succeeded"
    first.updated_at = 100
    assert asyncio.run(registry.get(first.job_id)) is None


# ---- REST routes -----------------------------------------------------------


class AcceptingRegistry:
    def __init__(self):
        self.job = None
        self.runner = None
        self.capacity = True

    def has_capacity(self):
        return self.capacity

    def create(self, name, action):
        now = time.time()
        self.job = repos.RepoJob("job-1", name, action, created_at=now, updated_at=now)
        return self.job

    def start(self, job, runner):
        self.runner = runner

    async def get(self, job_id):
        return self.job if self.job and self.job.job_id == job_id else None


def test_post_repos_rejects_bad_url_with_400(monkeypatch):
    monkeypatch.setattr(repos, "registry", AcceptingRegistry())
    response = _post("/repos", json={"url": "ftp://bad/repo.git"})
    assert response.status_code == 400
    assert set(response.json()) == {"error"}


def test_post_repos_returns_202_and_shape(monkeypatch):
    fake = AcceptingRegistry()
    monkeypatch.setattr(repos, "registry", fake)
    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})
    assert response.status_code == 202
    body = response.json()
    assert body == {
        "job_id": "job-1",
        "name": "repo",
        "status": "queued",
        "status_url": "/repos/jobs/job-1",
    }
    assert fake.runner is not None


def test_post_repos_returns_429_when_full(monkeypatch):
    fake = AcceptingRegistry()
    fake.capacity = False
    monkeypatch.setattr(repos, "registry", fake)
    response = _post("/repos", json={"url": "https://github.com/owner/repo.git"})
    assert response.status_code == 429


def test_delete_unknown_repo_returns_404(monkeypatch, tmp_path):
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(repos, "registry", AcceptingRegistry())
    response = _delete("/repos/ghost")
    assert response.status_code == 404


def test_delete_existing_repo_returns_202(monkeypatch, tmp_path):
    (tmp_path / "repo").mkdir()
    monkeypatch.setattr(repos, "CACHE_ROOT", tmp_path)
    fake = AcceptingRegistry()
    monkeypatch.setattr(repos, "registry", fake)
    response = _delete("/repos/repo")
    assert response.status_code == 202
    assert response.json()["name"] == "repo"
    assert fake.runner is not None
