"""Multi-repo incremental code indexing (CocoIndex + tree-sitter + pgvector).

Scans each subdirectory of the repo cache as an independent codebase, so a repo
that appears or disappears is mounted or torn down on the next update.

Run once / incremental:  uv run cocoindex update src/memory_base/ingest/code.py
Live watch:              uv run cocoindex update -L src/memory_base/ingest/code.py
"""

from __future__ import annotations

import asyncio
import os
import pathlib
from dataclasses import dataclass
from typing import Annotated, AsyncIterator

import asyncpg
from loguru import logger
from numpy.typing import NDArray

import cocoindex as coco
from cocoindex.connectors import localfs, postgres
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.chunk import Chunk
from cocoindex.resources.file import FileLike, PatternFilePathMatcher
from cocoindex.resources.id import IdGenerator

from memory_base.core.config import PG_SCHEMA, VllmEmbedder, db_url

TABLE_NAME = "code_chunks"
CACHE_ROOT = pathlib.Path(
    os.getenv("REPO_CACHE", pathlib.Path(__file__).parent.parent.parent.parent / ".repos_cache")
).resolve()
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

INCLUDED_PATTERNS = [
    "**/*.py",
    "**/*.js",
    "**/*.ts",
    "**/*.tsx",
    "**/*.go",
    "**/*.rs",
    "**/*.java",
    "**/*.c",
    "**/*.cpp",
    "**/*.h",
    "**/*.rb",
    "**/*.php",
    "**/*.sh",
    "**/*.md",
    "**/*.toml",
    "**/*.yml",
    "**/*.yaml",
    "**/*.sql",
]
EXCLUDED_PATTERNS = [
    "**/.*",
    "**/.venv",
    "**/__pycache__",
    "**/node_modules",
    "**/dist",
    "**/build",
    "**/vendor",
    "**/*.min.js",
    "**/*-bundle.js",
    "**/pnpm-lock.yaml",
]
# Minified/generated files pack far more characters per line than hand-written
# source; average line length is a size-independent proxy that large honest
# source files (e.g. a 500KB unminified lodash.js, avg ~32 chars/line) pass.
_MAX_AVG_LINE_LENGTH = 200

PG_DB = coco.ContextKey[asyncpg.Pool]("memory_base_db")
EMBEDDER = coco.ContextKey[VllmEmbedder]("embedder")

_splitter = RecursiveSplitter()


def _cache_rel(path: pathlib.PurePath) -> str:
    """Cache-relative path (repo/sub/file.py): unique per repo, ref-friendly."""
    return str(pathlib.Path(path).relative_to(CACHE_ROOT))


def _avg_line_length(text: str) -> float:
    lines = text.splitlines() or [""]
    return len(text) / len(lines)


def _is_minified(text: str, filename: str) -> bool:
    """True when average line length looks minified/bundled; markdown prose is exempt."""
    return not filename.endswith(".md") and _avg_line_length(text) > _MAX_AVG_LINE_LENGTH


async def _commit_time(path: pathlib.Path) -> float:
    """Last commit time for `path`, falling back to filesystem mtime."""
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(path.parent),
        "log",
        "-1",
        "--format=%ct",
        "--",
        path.name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    if proc.returncode == 0 and out.strip():
        return float(out.strip())
    return path.stat().st_mtime


@dataclass
class CodeChunk:
    id: int
    repo: str
    filename: str
    code: str
    embedding: Annotated[NDArray, VllmEmbedder()]
    start_line: int
    end_line: int
    mtime: float  # commit time (epoch sec), used for time-decay scoring at query time


@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    async with asyncpg.create_pool(db_url()) as pool:
        builder.provide(PG_DB, pool)
        builder.provide(EMBEDDER, VllmEmbedder())
        yield


_EMBED_BATCH_SIZE = 64


@coco.fn.as_async(batching=True, max_batch_size=_EMBED_BATCH_SIZE)
async def _batched_embed(texts: list[str], embedder: VllmEmbedder) -> list[NDArray]:
    """Groups concurrent process_chunk embed calls into one vLLM request per batch."""
    try:
        return await embedder.embed_many(texts)
    except Exception as err:
        # One bad text must not fail every other text sharing this batch; halve and retry.
        raise coco.RetryWithSmallerBatch() from err


@coco.fn
async def process_chunk(
    chunk: Chunk,
    repo: str,
    filename: str,
    mtime: float,
    id_gen: IdGenerator,
    table: postgres.TableTarget[CodeChunk],
) -> None:
    embedding = await _batched_embed(chunk.text, coco.use_context(EMBEDDER))
    table.declare_row(
        row=CodeChunk(
            id=await id_gen.next_id(chunk.text),
            repo=repo,
            filename=filename,
            code=chunk.text,
            embedding=embedding,
            start_line=chunk.start.line,
            end_line=chunk.end.line,
            mtime=mtime,
        ),
    )


@coco.fn(memo=True)
async def process_file(
    file: FileLike,
    repo: str,
    table: postgres.TableTarget[CodeChunk],
) -> None:
    text = await file.read_text()
    filename = str(file.file_path.path.name)
    if _is_minified(text, filename):
        logger.info(
            "skip minified file {}: avg line length {:.0f}",
            _cache_rel(file.file_path.path),
            _avg_line_length(text),
        )
        return
    language = detect_code_language(filename=filename)
    chunks = _splitter.split(
        text,
        chunk_size=1000,
        min_chunk_size=300,
        chunk_overlap=300,
        language=language,
    )
    mtime = await _commit_time(pathlib.Path(file.file_path.path))
    id_gen = IdGenerator()
    await coco.map(
        process_chunk, chunks, repo, _cache_rel(file.file_path.path), mtime, id_gen, table
    )


@coco.fn
async def process_repo(
    repo: str,
    repo_dir: pathlib.Path,
    table: postgres.TableTarget[CodeChunk],
) -> None:
    files = localfs.walk_dir(
        repo_dir,
        recursive=True,
        path_matcher=PatternFilePathMatcher(
            included_patterns=INCLUDED_PATTERNS,
            excluded_patterns=EXCLUDED_PATTERNS,
        ),
        live=True,
    )
    await coco.mount_each(process_file, files.items(), repo, table)


@coco.fn
async def app_main(root_dir: pathlib.Path) -> None:
    table = await postgres.mount_table_target(
        PG_DB,
        table_name=TABLE_NAME,
        table_schema=await postgres.TableSchema.from_class(CodeChunk, primary_key=["id"]),
        pg_schema_name=PG_SCHEMA,
    )
    table.declare_vector_index(column="embedding", metric="cosine", method="hnsw")
    table.declare_sql_command_attachment(
        name="bm25",
        setup_sql=(
            "CREATE EXTENSION IF NOT EXISTS pg_textsearch; "
            f'DROP INDEX IF EXISTS "{PG_SCHEMA}".{TABLE_NAME}__fts; '
            f'CREATE INDEX IF NOT EXISTS code_chunks_bm25 ON "{PG_SCHEMA}"."{TABLE_NAME}" '
            "USING bm25(code) WITH (text_config='simple')"
        ),
        teardown_sql=f'DROP INDEX IF EXISTS "{PG_SCHEMA}".code_chunks_bm25',
    )

    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        await coco.mount(
            coco.component_subpath("repo", entry.name),
            process_repo,
            entry.name,
            entry,
            table,
        )


app = coco.App(
    coco.AppConfig(name="MemoryBaseCode"),
    app_main,
    root_dir=CACHE_ROOT,
)
