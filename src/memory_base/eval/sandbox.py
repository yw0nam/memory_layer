"""The real REST API on a throwaway Postgres, for a PR's runtime evidence.

The API subprocess inherits the environment, so DB_URL points at the throwaway DB while the
embedder and reranker settings come from .env, which load_dotenv reads without overriding set
variables.

CLI: uv run python -m memory_base.eval.sandbox
"""

from __future__ import annotations

import asyncio
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from memory_base.core import db
from memory_base.serve.access import keys

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_LABEL = "memory-base-sandbox"
API_BOOT_SECONDS = 60.0
AUTHOR = "claude-code"


def _docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@contextmanager
def throwaway_postgres():
    """A fresh Postgres from db.Dockerfile on tmpfs; DB_URL points at it until removal."""
    image = _docker("build", "-q", "-f", str(REPO_ROOT / "db.Dockerfile"), str(REPO_ROOT))
    password = secrets.token_hex(16)
    container = _docker(
        "run", "-d", "--rm", "--label", DB_LABEL,
        "--tmpfs", "/var/lib/postgresql/data",
        "-e", "POSTGRES_USER=memory", "-e", f"POSTGRES_PASSWORD={password}",
        "-e", "POSTGRES_DB=memory_base",
        "-p", "127.0.0.1::5432",
        image, "postgres", "-c", "shared_preload_libraries=pg_textsearch",
    )  # fmt: skip
    try:
        port = _docker("port", container, "5432/tcp").splitlines()[0].rsplit(":", 1)[1]
        if not port.isdigit():
            raise ValueError(f"unexpected docker port: {port!r}")
        url = f"postgresql://memory:{password}@127.0.0.1:{port}/memory_base"
        os.environ["DB_URL"] = url
        yield {"image": image, "url": url}
    finally:
        _docker("rm", "-f", container)


async def _prepare_schema(url: str, deadline_seconds: float = 90) -> None:
    """Wait for the fresh server, then create the schema."""
    import asyncpg

    from memory_base.core.schema import ensure_schema

    deadline = time.monotonic() + deadline_seconds
    while True:
        try:
            conn = await asyncpg.connect(url, timeout=3)
            break
        except (OSError, asyncpg.CannotConnectNowError, asyncpg.ConnectionDoesNotExistError):
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.5)
    try:
        await ensure_schema(conn)
    finally:
        await conn.close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class SandboxApi:
    """The REST API as a subprocess on a free loopback port, against the throwaway DB_URL."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self.state = tempfile.TemporaryDirectory()

    def __enter__(self) -> SandboxApi:
        argv = [
            "uv", "run", "--no-sync", "uvicorn", "memory_base.serve.api:app",
            "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "warning",
        ]  # fmt: skip
        env = {**os.environ, "LOG_DIR": str(Path(self.state.name) / "logs")}
        try:
            self.proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=env)
            deadline = time.monotonic() + API_BOOT_SECONDS
            while True:
                try:
                    if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                        return self
                except httpx.HTTPError:
                    pass
                if self.proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"sandbox REST API did not come up on {self.url}")
                time.sleep(0.5)
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc: Any) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.state.cleanup()


async def _prepare(url: str) -> str:
    await _prepare_schema(url)
    try:
        return await keys.new_key("sandbox")
    finally:
        await db.close_pool()


def _wait() -> None:
    threading.Event().wait()


def _stop(signum, frame) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt


def main(wait: Callable[[], None] = _wait) -> None:
    # SIGINT is installed explicitly because a background start inherits it as ignored.
    previous = {
        signal.SIGINT: signal.signal(signal.SIGINT, _stop),
        signal.SIGTERM: signal.signal(signal.SIGTERM, _stop),
    }
    try:
        with throwaway_postgres() as database:
            key = asyncio.run(_prepare(database["url"]))
            with SandboxApi() as api:
                print(f"URL {api.url}")
                print(f"KEY {key}")
                print(f"AUTHOR {AUTHOR}")
                print(f"curl -s -H 'X-API-Key: {key}' -H 'X-Author: {AUTHOR}' {api.url}/notes")
                sys.stdout.flush()
                wait()
    except KeyboardInterrupt:
        print("sandbox stopped", file=sys.stderr)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    main()
