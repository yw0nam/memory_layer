"""Logging configuration for the application."""

import logging
import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

from loguru import logger

_configured = False
_MAX_BRIDGE_FRAME_WALK = 12
_ROTATION_MAX_BYTES = 100 * 1024 * 1024


class InterceptHandler(logging.Handler):
    """Route stdlib `logging` records (uvicorn, starlette, ...) into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk past the stdlib logging call chain (Handler.handle, Logger.callHandlers,
        # Logger.handle, Logger._log, Logger.<level>) to the frame that actually logged.
        frame, depth = sys._getframe(1), 1
        while (
            frame is not None
            and depth < _MAX_BRIDGE_FRAME_WALK
            and frame.f_code.co_filename == logging.__file__
        ):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class _SizeOrTimeRotation:
    """Rotate a log file once it exceeds a byte cap or crosses a daily time boundary."""

    def __init__(self, *, max_bytes: int, at: time) -> None:
        now = datetime.now()
        self._max_bytes = max_bytes
        self._next_rotation = now.replace(
            hour=at.hour, minute=at.minute, second=at.second, microsecond=0
        )
        if self._next_rotation <= now:
            self._next_rotation += timedelta(days=1)

    def __call__(self, message, file) -> bool:
        file.seek(0, 2)
        if file.tell() + len(message) > self._max_bytes:
            return True
        record_time = message.record["time"].replace(tzinfo=None)
        if record_time >= self._next_rotation:
            while self._next_rotation <= record_time:
                self._next_rotation += timedelta(days=1)
            return True
        return False


def setup_logging(
    level: str = "INFO",
    rotation: str = "00:00",
    retention: str = "30 days",
) -> None:
    """Configure unified logging with Request ID support and daily-or-100MB rotation.

    Idempotent: repeat calls are no-ops so importing modules under test
    doesn't stack sinks or double-install the stdlib bridge.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: Daily time to rotate logs at (default: "00:00" for midnight);
            the file also rotates early once it exceeds 100 MB
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

    # File output (daily rotation or 100 MB, whichever comes first)
    rotation_trigger = _SizeOrTimeRotation(
        max_bytes=_ROTATION_MAX_BYTES,
        at=datetime.strptime(rotation, "%H:%M").time(),
    )
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format=file_format,
        level=level,
        rotation=rotation_trigger,
        retention=retention,
        encoding="utf-8",
    )

    # Configure default request_id for logs without context
    logger.configure(extra={"request_id": "-"})

    # Bridge stdlib logging (uvicorn, starlette, asyncpg, ...) into loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    _configured = True
