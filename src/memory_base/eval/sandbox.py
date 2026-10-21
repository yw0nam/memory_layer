"""The real REST API on a throwaway Postgres, for a PR's runtime evidence.

The API subprocess inherits the environment, so DB_URL points at the throwaway DB while the
embedder, reranker and chat settings come from .env, which load_dotenv reads without
overriding set variables.

CLI: uv run python -m memory_base.eval.sandbox
"""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
from collections.abc import Callable, Sequence

from memory_base.core import db
from memory_base.eval.longmemeval import _prepare_schema, throwaway_postgres
from memory_base.eval.mcp_writer import EvalApi
from memory_base.serve.access import keys

AUTHOR = "claude-code"


async def _prepare(url: str) -> str:
    await _prepare_schema(url)
    try:
        return await keys.new_key("sandbox")
    finally:
        await db.close_pool()


def _wait() -> None:
    threading.Event().wait()


def _terminate(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None, wait: Callable[[], None] = _wait) -> None:
    previous = {
        signal.SIGINT: signal.signal(signal.SIGINT, signal.default_int_handler),
        signal.SIGTERM: signal.signal(signal.SIGTERM, _terminate),
    }
    try:
        with throwaway_postgres() as database:
            key = asyncio.run(_prepare(database["url"]))
            with EvalApi() as api:
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
