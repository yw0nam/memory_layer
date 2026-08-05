"""Unit tests for the unified loguru setup: file sink, stdlib bridge, idempotence."""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from pathlib import Path

from loguru import logger

from memory_base.core import logger as logger_module
from memory_base.core.logger import setup_logging


class _FakeMessage(str):
    """Minimal stand-in for loguru's `Message` (a `str` carrying `.record`)."""

    def __new__(cls, text: str, record: dict):
        instance = str.__new__(cls, text)
        instance.record = record
        return instance


class _FakeFile:
    def __init__(self, size: int) -> None:
        self._size = size

    def seek(self, _offset: int, _whence: int) -> None:
        return None

    def tell(self) -> int:
        return self._size


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


def test_stdlib_bridge_attributes_the_real_caller_module(monkeypatch, tmp_path):
    log_dir = _reset(monkeypatch, tmp_path)
    setup_logging()

    logging.getLogger("x").warning("attributed warning")
    logger.complete()

    content = _log_file(log_dir).read_text(encoding="utf-8")
    line = next(entry for entry in content.splitlines() if "attributed warning" in entry)
    assert f"{__name__}:" in line


def test_rotation_trigger_fires_on_size_cap_and_on_daily_boundary():
    trigger = logger_module._SizeOrTimeRotation(max_bytes=10, at=time(0, 0))
    now_message = _FakeMessage("x", {"time": datetime.now()})

    assert trigger(now_message, _FakeFile(0)) is False
    assert trigger(now_message, _FakeFile(20)) is True

    future_message = _FakeMessage("x", {"time": datetime.now() + timedelta(days=2)})
    assert trigger(future_message, _FakeFile(0)) is True
