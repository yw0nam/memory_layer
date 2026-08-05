"""Killable MarkItDown conversion worker."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from memory_base.adapters.document import CONVERSION_LIMIT_EXIT_CODE, CONVERSION_MAX_BYTES


class OutputLimitExceeded(Exception):
    """Raised when converted UTF-8 output reaches the byte limit."""


class BoundedWriter:
    """Write UTF-8 text without allowing the output file past a byte limit."""

    def __init__(self, path: Path, limit: int = CONVERSION_MAX_BYTES) -> None:
        self._stream = path.open("wb")
        self._limit = limit
        self._written = 0

    def write(self, text: str) -> None:
        data = text.encode("utf-8")
        remaining = self._limit - self._written
        if len(data) > remaining:
            if remaining:
                self._stream.write(data[:remaining])
                self._written += remaining
                self._stream.flush()
            raise OutputLimitExceeded
        self._stream.write(data)
        self._written += len(data)

    def close(self) -> None:
        self._stream.close()


def convert(input_path: Path, output_path: Path) -> int:
    """Convert one document and return a process exit code."""
    from markitdown import MarkItDown

    writer = BoundedWriter(output_path)
    try:
        result = MarkItDown().convert(input_path)
        text = result.text_content
        for offset in range(0, len(text), 64 * 1024):
            writer.write(text[offset : offset + 64 * 1024])
    except OutputLimitExceeded:
        return CONVERSION_LIMIT_EXIT_CODE
    finally:
        writer.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the conversion worker."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        return 2
    try:
        return convert(Path(arguments[0]), Path(arguments[1]))
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
