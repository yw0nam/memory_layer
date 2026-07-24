"""Integration tests for URL-driven multi-repo ingestion against the live DB
and CocoIndex indexer (configured via .env).

Destructive: rebuilds the code_chunks table from throwaway local git repos, so
it is gated behind the `integration` marker and skipped when the DB is
unreachable (keeping CI safe). Index/teardown are driven through the job-runner
coroutines directly so completion is deterministic in-process; the REST surface
(GET /repos, DELETE, job polling) is exercised through the real Starlette app.
"""

from __future__ import annotations

import asyncio
import os
import subprocess

import httpx
import pytest

import asyncpg

from memory_base.common import DB_URL, PG_SCHEMA


def _db_reachable() -> bool:
    async def _check() -> None:
        conn = await asyncpg.connect(DB_URL, timeout=5)
        await conn.close()

    try:
        asyncio.run(_check())
        return True
    except Exception:
        return False


if not _db_reachable():
    pytest.skip(
        f"DB not reachable at {DB_URL}; skipping integration tests", allow_module_level=True
    )

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
            transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
        ) as client:
            return await client.get(path)

    return asyncio.run(request())


def _delete(path):
    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
        ) as client:
            return await client.delete(path)

    return asyncio.run(request())


async def _repo_row_count(repo: str) -> int:
    conn = await asyncpg.connect(DB_URL)
    try:
        return await conn.fetchval(
            f'SELECT COUNT(*) FROM "{PG_SCHEMA}"."code_chunks" WHERE repo = $1', repo
        )
    finally:
        await conn.close()


def test_multi_repo_index_search_and_teardown(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(repos, "CACHE_ROOT", cache)
    monkeypatch.setenv("REPO_CACHE", str(cache))

    origin_a = tmp_path / "origin_a"
    origin_b = tmp_path / "origin_b"
    _make_repo(origin_a, "alpha_marker_fn")
    _make_repo(origin_b, "beta_marker_fn")

    # Local paths are intentionally rejected by the URL trust boundary, so drive
    # the ingest runner directly (clone + index) rather than POST /repos.
    asyncio.run(repos._run_ingest_job(str(origin_a), cache / "repo_a", None))
    asyncio.run(repos._run_ingest_job(str(origin_b), cache / "repo_b", None))

    listed = {r["name"]: r for r in _get("/repos").json()}
    assert {"repo_a", "repo_b"} <= set(listed)
    assert listed["repo_a"]["chunks"] > 0
    assert listed["repo_b"]["chunks"] > 0

    hits_a = asyncio.run(search("alpha_marker_fn", source="code", rerank=False))
    assert any(h.ref.startswith("repo_a/") for h in hits_a)
    hits_b = asyncio.run(search("beta_marker_fn", source="code", rerank=False))
    assert any(h.ref.startswith("repo_b/") for h in hits_b)

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
