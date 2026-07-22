"""Phase 0: incremental code repo indexing (CocoIndex + tree-sitter + pgvector).

Run once / incremental:  uv run cocoindex update src/code_index.py
Live watch:              uv run cocoindex update -L src/code_index.py
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Annotated, AsyncIterator

import asyncpg
from numpy.typing import NDArray

import cocoindex as coco
from cocoindex.connectors import localfs, postgres
from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from cocoindex.resources.chunk import Chunk
from cocoindex.resources.file import FileLike, PatternFilePathMatcher
from cocoindex.resources.id import IdGenerator

from common import DB_URL, PG_SCHEMA, VllmEmbedder

TABLE_NAME = "code_chunks"
# ponytail: single repo for now; add paths here to index more repos.
REPO_ROOT = pathlib.Path(os.getenv("INDEX_REPO", pathlib.Path(__file__).parent.parent))

PG_DB = coco.ContextKey[asyncpg.Pool]("memory_base_db")
EMBEDDER = coco.ContextKey[VllmEmbedder]("embedder")

_splitter = RecursiveSplitter()


@dataclass
class CodeChunk:
    id: int
    repo: str
    filename: str
    code: str
    embedding: Annotated[NDArray, VllmEmbedder()]
    start_line: int
    end_line: int
    mtime: float  # file mtime (epoch sec), used for time-decay scoring at query time


@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    async with asyncpg.create_pool(DB_URL) as pool:
        builder.provide(PG_DB, pool)
        builder.provide(EMBEDDER, VllmEmbedder())
        yield


@coco.fn
async def process_chunk(
    chunk: Chunk,
    filename: pathlib.PurePath,
    mtime: float,
    id_gen: IdGenerator,
    table: postgres.TableTarget[CodeChunk],
) -> None:
    embedding = await coco.use_context(EMBEDDER).embed(chunk.text)
    table.declare_row(
        row=CodeChunk(
            id=await id_gen.next_id(chunk.text),
            repo=REPO_ROOT.name,
            filename=str(filename),
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
    table: postgres.TableTarget[CodeChunk],
) -> None:
    text = await file.read_text()
    language = detect_code_language(filename=str(file.file_path.path.name))
    chunks = _splitter.split(
        text,
        chunk_size=1000,
        min_chunk_size=300,
        chunk_overlap=300,
        language=language,
    )
    mtime = pathlib.Path(file.file_path.path).stat().st_mtime
    id_gen = IdGenerator()
    await coco.map(process_chunk, chunks, file.file_path.path, mtime, id_gen, table)


@coco.fn
async def app_main(sourcedir: pathlib.Path) -> None:
    table = await postgres.mount_table_target(
        PG_DB,
        table_name=TABLE_NAME,
        table_schema=await postgres.TableSchema.from_class(CodeChunk, primary_key=["id"]),
        pg_schema_name=PG_SCHEMA,
    )
    table.declare_vector_index(column="embedding", metric="cosine", method="hnsw")
    # FTS over raw code: exact tokens (error strings, flags, identifiers).
    # 'simple' config: no stemming — right for code identifiers.
    table.declare_sql_command_attachment(
        name="fts_gin",
        setup_sql=(
            f'CREATE INDEX IF NOT EXISTS {TABLE_NAME}__fts ON "{PG_SCHEMA}"."{TABLE_NAME}" '
            "USING GIN (to_tsvector('simple', code))"
        ),
        teardown_sql=f'DROP INDEX IF EXISTS "{PG_SCHEMA}".{TABLE_NAME}__fts',
    )

    files = localfs.walk_dir(
        sourcedir,
        recursive=True,
        path_matcher=PatternFilePathMatcher(
            included_patterns=["**/*.py", "**/*.md", "**/*.toml", "**/*.yml", "**/*.sql"],
            excluded_patterns=["**/.*", "**/.venv", "**/__pycache__", "**/node_modules"],
        ),
        live=True,
    )
    await coco.mount_each(process_file, files.items(), table)


app = coco.App(
    coco.AppConfig(name="MemoryBaseCode"),
    app_main,
    sourcedir=REPO_ROOT,
)
