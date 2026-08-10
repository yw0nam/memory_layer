"""Shared git command helper for tests."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_output(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()
