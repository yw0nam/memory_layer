"""Document conversion, chunking, CSV sampling, and storage-row mapping."""

from __future__ import annotations

import asyncio
import csv
import importlib.metadata
import os
import re
import sys
import tempfile
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

SUPPORTED_EXTENSIONS = frozenset(
    {".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".pdf", ".docx", ".pptx", ".csv"}
)
MCP_TEXT_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".csv"})
CSV_MAX_BYTES = 5 * 1024 * 1024
CSV_MAX_ROWS = 100_000
CONVERSION_MAX_BYTES = 16 * 1024 * 1024
CONVERSION_TIMEOUT_SECONDS = 120
EXTRACTED_MAX_CHARS = 2_000_000
SOFT_CHUNK_CHARS = 1_500
HARD_CHUNK_CHARS = 2_000
MIN_CHUNK_CHARS = 200
CONVERSION_LIMIT_EXIT_CODE = 23

_DOCUMENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,120}$")
_HEADING_RE = re.compile(r"^(#{1,6}) +(.*)$")
_MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\([^)]+\)")
_URL_RE = re.compile(r"https?://[^\s<>()]+")


class DocumentError(ValueError):
    """Base error for deterministic document processing failures."""


class UnsupportedDocumentError(DocumentError):
    """Raised when a filename has an unsupported extension."""


class ConversionError(DocumentError):
    """Raised when bounded conversion fails."""


@dataclass(frozen=True)
class Chunk:
    text: str
    heading_path: tuple[str, ...]
    ordinal: int | None = None


@dataclass(frozen=True)
class ChunkingResult:
    chunks: list[Chunk]
    dropped: int


@dataclass(frozen=True)
class CSVSample:
    header: list[str]
    rows: list[list[str]]
    row_count: int
    column_count: int


@dataclass(frozen=True)
class ConversionResult:
    text: str
    converter: str


def extension_for(filename: str) -> str:
    """Return a validated lowercase document extension."""
    extension = PurePath(filename.replace("\\", "/")).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(f"unsupported document extension: {extension or '(none)'}")
    return extension


def normalize_document_id(value: str) -> str:
    """Normalize and validate a stable document identity."""
    normalized = PurePath(value.replace("\\", "/")).name.lower()
    if not _DOCUMENT_ID_RE.fullmatch(normalized):
        raise DocumentError(
            "document_id must match ^[a-z0-9][a-z0-9._-]{0,120}$ after normalization"
        )
    return normalized


def _flush_paragraph(
    blocks: list[Chunk], paragraph: list[str], heading_path: tuple[str, ...]
) -> None:
    if paragraph:
        blocks.append(Chunk("\n".join(paragraph).strip(), heading_path))
        paragraph.clear()


def _blocks(text: str) -> list[Chunk]:
    blocks: list[Chunk] = []
    headings: list[str] = []
    paragraph: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        heading = _HEADING_RE.match(line)
        if heading:
            _flush_paragraph(blocks, paragraph, tuple(headings))
            level = len(heading.group(1))
            headings = headings[: level - 1]
            headings.append(heading.group(2).strip())
            index += 1
            continue
        if line.startswith("```"):
            _flush_paragraph(blocks, paragraph, tuple(headings))
            fenced = [line]
            index += 1
            while index < len(lines):
                fenced.append(lines[index])
                closing = lines[index].startswith("```")
                index += 1
                if closing:
                    break
            blocks.append(Chunk("\n".join(fenced), tuple(headings)))
            continue
        if not line.strip():
            _flush_paragraph(blocks, paragraph, tuple(headings))
        else:
            paragraph.append(line)
        index += 1
    _flush_paragraph(blocks, paragraph, tuple(headings))
    return [block for block in blocks if block.text]


def _pack(blocks: Sequence[Chunk]) -> list[Chunk]:
    packed: list[Chunk] = []
    current: Chunk | None = None
    for block in blocks:
        if current is None:
            current = block
            continue
        combined = f"{current.text}\n\n{block.text}"
        if current.heading_path == block.heading_path and len(combined) <= SOFT_CHUNK_CHARS:
            current = Chunk(combined, current.heading_path)
        else:
            packed.append(current)
            current = block
    if current is not None:
        packed.append(current)
    return packed


def _split_paragraph(text: str) -> list[str]:
    pieces: list[str] = []
    remaining = text
    while len(remaining) > HARD_CHUNK_CHARS:
        boundary = HARD_CHUNK_CHARS
        whitespace = max(
            (
                index
                for index, char in enumerate(remaining[: HARD_CHUNK_CHARS + 1])
                if char.isspace()
            ),
            default=-1,
        )
        if whitespace > 0:
            boundary = whitespace
        pieces.append(remaining[:boundary].rstrip())
        remaining = remaining[boundary:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _split_fence(text: str) -> list[str]:
    pieces: list[str] = []
    remaining = text
    while len(remaining) > HARD_CHUNK_CHARS:
        boundary = remaining.rfind("\n", 0, HARD_CHUNK_CHARS + 1)
        if boundary <= 0:
            boundary = HARD_CHUNK_CHARS
        pieces.append(remaining[:boundary])
        remaining = remaining[boundary:].lstrip("\n")
    if remaining:
        pieces.append(remaining)
    return pieces


def _split_oversized(chunks: Sequence[Chunk]) -> list[Chunk]:
    split: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.text) <= HARD_CHUNK_CHARS:
            split.append(chunk)
            continue
        splitter = _split_fence if chunk.text.startswith("```") else _split_paragraph
        split.extend(Chunk(piece, chunk.heading_path) for piece in splitter(chunk.text))
    return split


def _residue_pass(chunks: Sequence[Chunk]) -> tuple[list[Chunk], int]:
    result = list(chunks)
    dropped = 0
    index = 0
    while index < len(result):
        current = result[index]
        if len(current.text) >= MIN_CHUNK_CHARS:
            index += 1
            continue
        if index > 0:
            predecessor = result[index - 1]
            merged = f"{predecessor.text}\n\n{current.text}"
            if predecessor.heading_path == current.heading_path and len(merged) <= HARD_CHUNK_CHARS:
                result[index - 1] = Chunk(merged, current.heading_path)
                result.pop(index)
                index -= 1
                continue
        if index + 1 < len(result):
            successor = result[index + 1]
            merged = f"{current.text}\n\n{successor.text}"
            if successor.heading_path == current.heading_path and len(merged) <= HARD_CHUNK_CHARS:
                result[index] = Chunk(merged, current.heading_path)
                result.pop(index + 1)
                continue
        result.pop(index)
        dropped += 1
    return result, dropped


def _covered_characters(line: str) -> int:
    spans = [
        match.span() for regex in (_MARKDOWN_LINK_RE, _URL_RE) for match in regex.finditer(line)
    ]
    if not spans:
        return 0
    spans.sort()
    covered = 0
    start, end = spans[0]
    for next_start, next_end in spans[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            covered += end - start
            start, end = next_start, next_end
    return covered + end - start


def is_junk(text: str) -> bool:
    """Return whether a chunk fails the deterministic signal gate."""
    non_whitespace = [char for char in text if not char.isspace()]
    if not non_whitespace:
        return True
    letters = sum(unicodedata.category(char).startswith("L") for char in non_whitespace)
    if letters / len(non_whitespace) < 0.3:
        return True
    lines = [line for line in text.splitlines() if line.strip()]
    link_lines = sum(_covered_characters(line) > len(line) / 2 for line in lines)
    return bool(lines) and link_lines > len(lines) / 2


def chunk_markdown(text: str) -> ChunkingResult:
    """Apply the ordered deterministic document chunking pipeline."""
    packed = _pack(_blocks(text))
    split = _split_oversized(packed)
    residues, dropped = _residue_pass(split)
    accepted: list[Chunk] = []
    for chunk in residues:
        if is_junk(chunk.text):
            dropped += 1
        else:
            accepted.append(Chunk(chunk.text, chunk.heading_path, len(accepted)))
    return ChunkingResult(accepted, dropped)


async def convert_to_markdown(input_path: Path) -> ConversionResult:
    """Convert a document in a bounded killable worker process."""
    descriptor, output_name = tempfile.mkstemp(prefix="memory-base-convert-", suffix=".md")
    os.close(descriptor)
    output_path = Path(output_name)
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "memory_base.adapters.document_worker",
            str(input_path),
            str(output_path),
        )
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=CONVERSION_TIMEOUT_SECONDS)
        except TimeoutError:
            timed_out = True
            process.kill()
            await process.wait()

        output_size = output_path.stat().st_size
        if output_size > CONVERSION_MAX_BYTES:
            raise ConversionError("converted output exceeds 16 MB")
        if timed_out:
            raise ConversionError("document conversion timed out after 120 seconds")
        if process.returncode == CONVERSION_LIMIT_EXIT_CODE:
            raise ConversionError("converted output exceeds 16 MB")
        if process.returncode != 0:
            raise ConversionError(f"document conversion failed with exit code {process.returncode}")

        text = output_path.read_text(encoding="utf-8")
        if len(text) > EXTRACTED_MAX_CHARS:
            raise ConversionError("extracted text exceeds 2000000 chars")
        if not text.strip():
            raise ConversionError("document extraction produced no text")
        version = importlib.metadata.version("markitdown")
        return ConversionResult(text=text, converter=f"markitdown:{version}")
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        output_path.unlink(missing_ok=True)


def read_csv_sample(path: Path) -> CSVSample:
    """Read and validate a complete CSV while retaining its verbatim cell strings."""
    if path.stat().st_size > CSV_MAX_BYTES:
        raise DocumentError("CSV exceeds 5 MB")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise DocumentError("CSV header must not be empty") from exc
            if not header:
                raise DocumentError("CSV header must not be empty")
            if any(not name.strip() for name in header):
                raise DocumentError("CSV header names must not be empty")
            if len(set(header)) != len(header):
                raise DocumentError("CSV has duplicate header names")
            if any("\x00" in name for name in header):
                raise DocumentError("CSV contains a NUL character")
            rows: list[list[str]] = []
            for row_number, row in enumerate(reader, start=1):
                if row_number > CSV_MAX_ROWS:
                    raise DocumentError(f"CSV exceeds {CSV_MAX_ROWS} rows")
                if len(row) != len(header):
                    raise DocumentError(
                        f"CSV row {row_number} has {len(row)} cells; expected {len(header)}"
                    )
                if any("\x00" in cell for cell in row):
                    raise DocumentError("CSV contains a NUL character")
                rows.append(row)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DocumentError(f"malformed CSV: {exc}") from exc
    return CSVSample(header, rows, len(rows), len(header))


def csv_prompt_context(sample: CSVSample) -> str:
    """Build format-specific context for generic summary enrichment."""
    rendered = [",".join(sample.header)]
    rendered.extend(",".join(row) for row in sample.rows[:20])
    return (
        "Create one English knowledge card describing this table, its fields, and useful "
        f"patterns visible in the sample. Total data rows: {sample.row_count}. "
        f"Columns: {sample.column_count}.\n\nCSV sample:\n" + "\n".join(rendered)
    )


async def build_csv_card(
    sample: CSVSample,
    summarize: Callable[[str, str], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Summarize a CSV sample with the generic enrichment operation."""
    context = csv_prompt_context(sample)
    return await summarize(context, "The input is a sampled tabular document.")


def _qualify_document_id(namespace: str, document_id: str) -> str:
    """Namespace-qualify a document_id for row-id construction.

    'default' keeps the legacy unqualified id so existing rows are
    unaffected; every other namespace gets a distinct id space so the same
    document_id ingested into two namespaces never collides on the
    memory_chunks primary key.
    """
    if namespace == "default":
        return document_id
    return f"{namespace}:{document_id}"


def _base_row(
    *,
    row_id: str,
    document_id: str,
    kind: str,
    raw: str,
    distilled: str | None,
    embedding_text: str,
    timestamp: float,
    metadata: dict[str, Any],
    namespace: str = "default",
) -> dict[str, Any]:
    return {
        "id": row_id,
        "source_type": "document",
        "source_ref": document_id,
        "chunk_kind": kind,
        "session_id": document_id,
        "content_raw": raw,
        "distilled": distilled,
        "embedding_text": embedding_text,
        "ts_last_active": timestamp,
        "idf_score": None,
        "namespace": namespace,
        "metadata": metadata,
    }


def map_document_rows(
    chunks: Sequence[Chunk],
    *,
    tags: Sequence[str],
    filename: str,
    document_id: str,
    content_hash: str,
    format_name: str,
    converter: str,
    origin: str | None,
    timestamp: float,
    namespace: str = "default",
) -> list[dict[str, Any]]:
    """Map document chunks to memory_chunks row dictionaries."""
    rows: list[dict[str, Any]] = []
    qualified_document_id = _qualify_document_id(namespace, document_id)
    for chunk in chunks:
        if chunk.ordinal is None:
            raise ValueError("chunk ordinals must be assigned before row mapping")
        heading = " > ".join(chunk.heading_path)
        metadata = {
            "filename": filename,
            "document_id": document_id,
            "heading_path": list(chunk.heading_path),
            "ordinal": chunk.ordinal,
            "content_hash": content_hash,
            "format": format_name,
            "converter": converter,
            "origin": origin,
            "tags": list(tags),
            "search_ref": f"{document_id}#chunk-{chunk.ordinal}",
        }
        embedding_text = f"{heading}\n\n{chunk.text}" if heading else chunk.text
        rows.append(
            _base_row(
                row_id=f"doc:{qualified_document_id}:{chunk.ordinal}",
                document_id=document_id,
                kind="doc",
                raw=chunk.text,
                distilled=None,
                embedding_text=embedding_text,
                timestamp=timestamp,
                metadata=metadata,
                namespace=namespace,
            )
        )
    return rows


def map_csv_card_row(
    card: dict[str, Any],
    sample: CSVSample,
    *,
    filename: str,
    document_id: str,
    content_hash: str,
    origin: str | None,
    timestamp: float,
    card_index: int = 0,
    namespace: str = "default",
) -> dict[str, Any]:
    """Map an enriched CSV card to a memory_chunks row dictionary."""
    summary = card["summary"]
    qualified_document_id = _qualify_document_id(namespace, document_id)
    return _base_row(
        row_id=f"doc:{qualified_document_id}:card:{card_index}",
        document_id=document_id,
        kind="doc",
        raw=summary,
        distilled=summary,
        embedding_text=summary,
        timestamp=timestamp,
        namespace=namespace,
        metadata={
            "filename": filename,
            "document_id": document_id,
            "card_index": card_index,
            "format": "csv",
            "row_count": sample.row_count,
            "column_count": sample.column_count,
            "columns": sample.header,
            "table_rows_loaded": True,
            "content_hash": content_hash,
            "origin": origin,
            "tags": card["tags"],
            "search_ref": f"{document_id}#card-{card_index}",
        },
    )


def map_csv_table_rows(sample: CSVSample) -> list[dict[str, Any]]:
    """Map every CSV data row to its ordered jsonb storage representation."""
    return [
        {
            "row_index": row_index,
            "data": {
                name: None if value == "" else value
                for name, value in zip(sample.header, row, strict=True)
            },
        }
        for row_index, row in enumerate(sample.rows)
    ]
