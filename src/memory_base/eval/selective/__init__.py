"""Selective-memory benchmark data: Memora weekly timelines and MemOps update units.

Steps, each writing under one output root (default ~/.local/share/memory-base/bench/v1):
  uv run python -m memory_base.eval.selective.memora --source PATH/Memora
  uv run python -m memory_base.eval.selective.memops --source PATH/MemOps
"""

from __future__ import annotations

import subprocess
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".local" / "share" / "memory-base" / "bench" / "v1"
DEV_PERSONAS = ("software_engineer", "creative_designer", "sales_manager")


def manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def source_commit(source: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
