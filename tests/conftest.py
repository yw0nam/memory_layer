"""Shared fixtures for the test suite.

No test reaches the deployment database. Unit tests see an unreachable DB_URL;
integration tests get a throwaway Postgres container that this session starts on
tmpfs and removes at the end, and every on-disk path points at a session tempdir.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE_LABEL = "memory-base-live-test"
OFFLINE_DB_URL = "postgresql://offline:offline@127.0.0.1:9/offline"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_orphaned_state_dirs() -> None:
    """A killed session cannot remove its tempdir; the next one does."""
    for path in Path(tempfile.gettempdir()).glob("memory-base-tests-*"):
        pid = path.name.split("-")[3]
        if pid.isdigit() and not _alive(int(pid)):
            shutil.rmtree(path, ignore_errors=True)


_remove_orphaned_state_dirs()
# Set before any memory_base import: these paths are read at module import time.
_STATE_DIR = Path(tempfile.mkdtemp(prefix=f"memory-base-tests-{os.getpid()}-"))
os.environ["DB_URL"] = OFFLINE_DB_URL
os.environ["INGEST_SPOOL"] = str(_STATE_DIR / "ingest-spool")
os.environ["REPO_CACHE"] = str(_STATE_DIR / "repos-cache")
os.environ["LOG_DIR"] = str(_STATE_DIR / "logs")
os.environ["COCOINDEX_DB"] = str(_STATE_DIR / "cocoindex")


_live_container: str | None = None


def _docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _remove_orphaned_live_databases() -> None:
    """A killed session cannot tear down its container; the next one does."""
    listing = _docker(
        "ps", "-a", "--filter", f"label={LIVE_LABEL}", "--format", '{{.ID}} {{.Label "pid"}}'
    )
    for line in listing.splitlines():
        container, _, pid = line.partition(" ")
        if not pid.isdigit() or not _alive(int(pid)):
            _docker("rm", "-f", container)


def _prepare_schema(url: str, deadline_seconds: float = 90) -> None:
    """Wait for the fresh server to accept connections, then create the schema."""
    import asyncpg

    from memory_base.core.schema import ensure_schema

    async def _connect_and_prepare() -> None:
        conn = await asyncpg.connect(url, timeout=3)
        try:
            await ensure_schema(conn)
        finally:
            await conn.close()

    deadline = time.monotonic() + deadline_seconds
    while True:
        try:
            asyncio.run(_connect_and_prepare())
            return
        except (OSError, asyncpg.CannotConnectNowError, asyncpg.ConnectionDoesNotExistError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def _start_live_database() -> None:
    """A fresh Postgres from db.Dockerfile on tmpfs, private to this session."""
    global _live_container
    _remove_orphaned_live_databases()
    image = _docker("build", "-q", "-f", str(ROOT / "db.Dockerfile"), str(ROOT))
    password = secrets.token_hex(16)
    _live_container = _docker(
        "run", "-d", "--rm",
        "--label", LIVE_LABEL, "--label", f"pid={os.getpid()}",
        "--tmpfs", "/var/lib/postgresql/data",
        "-e", "POSTGRES_USER=memory", "-e", f"POSTGRES_PASSWORD={password}",
        "-e", "POSTGRES_DB=memory_base",
        "-p", "127.0.0.1::5432",
        image, "postgres", "-c", "shared_preload_libraries=pg_textsearch",
    )  # fmt: skip
    port = _docker("port", _live_container, "5432/tcp").splitlines()[0].rsplit(":", 1)[1]
    url = f"postgresql://memory:{password}@127.0.0.1:{port}/memory_base"
    os.environ["DB_URL"] = url
    os.environ["TABLES_QUERY_PASSWORD"] = secrets.token_hex(16)
    _prepare_schema(url)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Start the throwaway database only when an integration test survived selection."""
    live = [item for item in items if item.get_closest_marker("integration") is not None]
    if not live:
        return
    if shutil.which("docker") is None:
        skip = pytest.mark.skip(reason="integration tests need docker for their database")
        for item in live:
            item.add_marker(skip)
        return
    _start_live_database()


def pytest_sessionfinish(session, exitstatus):
    if _live_container is not None:
        _docker("rm", "-f", _live_container)
    shutil.rmtree(_STATE_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def only_integration_tests_reach_the_database(request, monkeypatch):
    """Unit tests see an unreachable DB_URL; integration tests get the chat-model gate pinned open."""
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setenv("DB_URL", OFFLINE_DB_URL)
        return

    from memory_base.serve import notes

    async def accept(content, kind):
        return notes.ContentVerdict(accepted=True, reason="integration tests pin the gate open")

    monkeypatch.setattr(notes, "judge_note_content", accept)


@pytest.fixture(autouse=True)
def model_names(monkeypatch):
    """Tests that fake the backends still need the endpoints and names to resolve; real ones win."""
    for name in ("VLLM_URL", "EMB_URL", "RERANK_URL"):
        monkeypatch.setenv(name, os.getenv(name) or "http://vllm.test")
    for name in ("VLLM_MODEL", "EMB_MODEL", "RERANK_MODEL"):
        monkeypatch.setenv(name, os.getenv(name) or "test-model")


@pytest.fixture(autouse=True)
def table_query_password_for_unit_tests(request, monkeypatch):
    """Unit schema fakes receive a password without requiring deployment configuration."""
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setenv(
            "TABLES_QUERY_PASSWORD", os.getenv("TABLES_QUERY_PASSWORD") or "test-only"
        )


@pytest.fixture(autouse=True)
def isolated_access_log_buffer():
    """Rows buffered by one test's /search calls must never be flushed into the
    real retrieval_log by a later integration test in the same run."""
    from memory_base.serve import access_log

    access_log._pending_logs.clear()
    access_log._pending_hits.clear()
    yield
    access_log._pending_logs.clear()
    access_log._pending_hits.clear()


@pytest.fixture(autouse=True)
def isolate_unit_app_lifespan(request, monkeypatch):
    """Keep ordinary tests independent of Postgres-backed worker startup."""
    if request.node.get_closest_marker("integration") is not None:
        return

    from memory_base.serve import access_log, job_store

    async def initialize():
        return None

    def start_workers():
        return []

    async def stop_workers(tasks):
        assert tasks == []

    def start_flusher():
        return None

    async def stop_flusher(task):
        assert task is None

    monkeypatch.setattr(job_store, "initialize", initialize)
    monkeypatch.setattr(job_store, "start_workers", start_workers)
    monkeypatch.setattr(job_store, "stop_workers", stop_workers)
    monkeypatch.setattr(access_log, "start_flusher", start_flusher)
    monkeypatch.setattr(access_log, "stop_flusher", stop_flusher)


@pytest.fixture()
def rest_in_process(monkeypatch):
    """Route the MCP proxy at the REST app in-process (no server on :8010).

    Integration tests that invoke MCP tools exercise the full stack —
    proxy -> Starlette app -> notes/search -> real DB/embedder — without
    binding a port.
    """
    from memory_base.serve import api, mcp_server

    def _client() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://testserver"
        )

    monkeypatch.setattr(mcp_server, "_client", _client)
    # mcp_server has no request context here (ctx=None), so it falls back to this env var;
    # tests/serve/conftest.py's auth stub resolves "test-key" to an admin identity.
    monkeypatch.setenv("MEMORY_API_KEY", "test-key")
