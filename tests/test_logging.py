"""Unit tests for the unified loguru setup: file sink, stdlib bridge, idempotence."""

from __future__ import annotations

import logging
from pathlib import Path

from loguru import logger

from memory_base.core import logger as logger_module
from memory_base.core.logger import setup_logging


def _log_file(log_dir: Path) -> Path:
    files = list(log_dir.glob("app_*.log"))
    assert len(files) == 1
    return files[0]


def _reset(monkeypatch, tmp_path: Path) -> Path:
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    monkeypatch.setattr(logger_module, "_configured", False)
    logger.remove()
    return log_dir


def test_setup_logging_writes_default_request_id_to_file(monkeypatch, tmp_path):
    log_dir = _reset(monkeypatch, tmp_path)
    setup_logging()

    logger.info("hello from test")
    logger.complete()

    content = _log_file(log_dir).read_text(encoding="utf-8")
    assert "[-]" in content
    assert "hello from test" in content


def test_stdlib_logging_is_intercepted(monkeypatch, tmp_path):
    log_dir = _reset(monkeypatch, tmp_path)
    setup_logging()

    logging.getLogger("x").warning("stdlib warning message")
    logger.complete()

    content = _log_file(log_dir).read_text(encoding="utf-8")
    assert "stdlib warning message" in content


def test_setup_logging_is_idempotent(monkeypatch, tmp_path):
    log_dir = _reset(monkeypatch, tmp_path)
    setup_logging()
    setup_logging()
    setup_logging()

    logger.info("single line")
    logger.complete()

    content = _log_file(log_dir).read_text(encoding="utf-8")
    assert content.count("single line") == 1
