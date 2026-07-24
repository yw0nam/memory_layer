"""Logging configuration for the application."""

import logging
import os
import sys
from pathlib import Path

from loguru import logger

_configured = False


class InterceptHandler(logging.Handler):
    """Route stdlib `logging` records (uvicorn, starlette, ...) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    level: str = "INFO",
    rotation: str = "00:00",
    retention: str = "30 days",
) -> None:
    """Configure unified logging with Request ID support and daily rotation.

    Idempotent: repeat calls are no-ops so importing modules under test
    doesn't stack sinks or double-install the stdlib bridge.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate logs (default: "00:00" for daily at midnight)
        retention: How long to keep logs (default: "30 days")
    """
    global _configured
    if _configured:
        return

    # Remove default handler
    logger.remove()

    # Get log directory from env or use default
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Human-readable format with Request ID support
    # Format: [HH:mm:ss.SSS] | LEVEL | module:line | [RequestID] - message
    console_format = (
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<magenta>[{extra[request_id]!s}]</magenta> - "
        "<level>{message}</level>"
    )

    file_format = (
        "{time:HH:mm:ss.SSS} | {level: <8} | {name}:{line} | [{extra[request_id]!s}] - {message}"
    )

    # Console output (colored, for development)
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
    )

    # File output (daily rotation)
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format=file_format,
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    # Configure default request_id for logs without context
    logger.configure(extra={"request_id": "-"})

    # Bridge stdlib logging (uvicorn, starlette, asyncpg, ...) into loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    _configured = True
