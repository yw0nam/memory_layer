"""Integration tests for URL-driven multi-repo ingestion against Postgres and CocoIndex.

The test creates its own database on the session's throwaway Postgres container and
drops it afterwards; CocoIndex's LMDB state points at a temp dir. Both env vars are
redirected for the test, and the `run_index()` subprocess inherits them
(config.load_dotenv uses override=False). Skipped when the embedder is unreachable.
Index/teardown run through the job-runner coroutines directly so completion is
deterministic in-process.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from urllib.parse import urlsplit, urlunsplit

import httpx
import pytest

import asyncpg

IT_DB_NAME = "memory_base_it"


def _with_db(url: str, db_name: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{db_name}"))


def _admin_db() -> str:
    return _with_db(os.environ["DB_URL"], "postgres")


def _it_db() -> str:
    return _with_db(os.environ["DB_URL"], IT_DB_NAME)


def _emb_reachable() -> bool:
    emb = os.getenv("EMB_URL")
    if not emb:
        return False
    parts = urlsplit(emb)
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False


if not _emb_reachable():
    pytest.skip("EMB_URL unset or unreachable; skipping integration tests", allow_module_level=True)

pytestmark = pytest.mark.integration

from memory_base.retrieval.search import search  # noqa: E402
from memory_base.serve import api, repos  # noqa: E402

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _make_repo(path, marker: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    (path / "lib.py").write_text(f"def {marker}():\n    return '{marker}'\n")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "init"],
        check=True,
        env={**os.environ, **_GIT_ENV},
    )


def _get(path):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app),
            base_url="http://testserver",
            headers={"X-API-Key": "test-key"},
        ) as client:
            return await client.get(path)

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


async def _repo_row_count(repo: str) -> int:
    conn = await asyncpg.connect(_it_db())
    try:
        return await conn.fetchval(
            'SELECT COUNT(*) FROM "memory"."code_chunks" WHERE repo = $1', repo
        )
    finally:
        await conn.close()


class _CapturingBacklog:
    """Records durable route admission without running a background index."""

    def __init__(self):
        self.job = None

    async def admit(self, **kwargs):
        self.job = repos.RepoJob(**kwargs)
        return self.job

    async def get(self, job_id, *, kind):
        return self.job if self.job and self.job.job_id == job_id else None


@pytest.fixture()
def isolated_stack(tmp_path, monkeypatch):
    """Create a throwaway DB + LMDB state; redirect all DB access to them."""

    async def _create() -> None:
        admin = await asyncpg.connect(_admin_db())
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{IT_DB_NAME}" WITH (FORCE)')
            await admin.execute(f'CREATE DATABASE "{IT_DB_NAME}"')
        finally:
            await admin.close()
        conn = await asyncpg.connect(_it_db())
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute('CREATE SCHEMA IF NOT EXISTS "memory"')
        finally:
            await conn.close()

    async def _drop() -> None:
        admin = await asyncpg.connect(_admin_db())
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{IT_DB_NAME}" WITH (FORCE)')
        finally:
            await admin.close()

    it_db = _it_db()
    asyncio.run(_create())

    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("DB_URL", it_db)
    monkeypatch.setenv("COCOINDEX_DB", str(tmp_path / "cocoindex_state"))
    monkeypatch.setenv("REPO_CACHE", str(cache))
    monkeypatch.setattr(repos, "CACHE_ROOT", cache)
    backlog = _CapturingBacklog()
    monkeypatch.setattr(repos.job_store, "admit_repo", backlog.admit)
    monkeypatch.setattr(repos.job_store, "get_job", backlog.get)
    try:
        yield cache
    finally:
        asyncio.run(_drop())


def test_multi_repo_index_search_and_teardown(isolated_stack, tmp_path):
    cache = isolated_stack
    origin_a = tmp_path / "origin_a"
    origin_b = tmp_path / "origin_b"
    _make_repo(origin_a, "alpha_marker_fn")
    _make_repo(origin_b, "beta_marker_fn")

    # Local paths are intentionally rejected by the URL trust boundary, so drive
    # the ingest runner directly (clone + index) rather than POST /repos.
    asyncio.run(repos._run_ingest_job(str(origin_a), cache / "repo_a", None, "test"))
    asyncio.run(repos._run_ingest_job(str(origin_b), cache / "repo_b", None, "test"))

    listed = {r["name"]: r for r in _get("/repos").json()}
    assert {"repo_a", "repo_b"} <= set(listed)
    assert listed["repo_a"]["chunks"] > 0
    assert listed["repo_b"]["chunks"] > 0

    hits_a = asyncio.run(search("alpha_marker_fn", source="code", rerank=False))
    assert any(h.ref.startswith("repo_a/") for h in hits_a)
    hits_b = asyncio.run(search("beta_marker_fn", source="code", rerank=False))
    assert any(h.ref.startswith("repo_b/") for h in hits_b)

    scoped = asyncio.run(search("alpha_marker_fn", source="code", rerank=False, repo=["repo_a"]))
    assert scoped
    assert {h.meta["repo"] for h in scoped} == {"repo_a"}
    elsewhere = asyncio.run(search("alpha_marker_fn", source="code", rerank=False, repo=["repo_b"]))
    assert not any("alpha_marker_fn" in h.text for h in elsewhere)

    response = _delete("/repos/repo_a")
    assert response.status_code == 202
    body = response.json()
    assert body["name"] == "repo_a"
    assert _get(f"/repos/jobs/{body['job_id']}").status_code == 200

    # Deterministic teardown: run the remove runner, then assert rows are gone.
    asyncio.run(repos._run_remove_job(cache / "repo_a"))
    assert asyncio.run(_repo_row_count("repo_a")) == 0
    assert asyncio.run(_repo_row_count("repo_b")) > 0
    assert "repo_a" not in {r["name"] for r in _get("/repos").json()}
