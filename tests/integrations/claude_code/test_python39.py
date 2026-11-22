"""The hooks run under whatever python3 Claude Code invokes; the macOS system one is 3.9."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[3] / "integrations" / "claude_code" / "hooks"
SYSTEM_PYTHON = "/usr/bin/python3"

CHECK = """
import importlib, pathlib, sys
sys.path.insert(0, ".")
for path in sorted(pathlib.Path(".").rglob("*.py")):
    compile(path.read_text(), str(path), "exec")
    importlib.import_module(".".join(path.with_suffix("").parts))
"""


def _is_old_python() -> bool:
    try:
        out = subprocess.run(
            [SYSTEM_PYTHON, "-c", "import sys; print(sys.version_info < (3, 10))"],
            capture_output=True,
            text=True,
        ).stdout
    except OSError:
        return False
    return out.strip() == "True"


@pytest.mark.skipif(not _is_old_python(), reason="no system python3 older than 3.10")
def test_every_hook_module_compiles_and_imports_under_the_system_python():
    result = subprocess.run(
        [SYSTEM_PYTHON, "-B", "-c", CHECK], cwd=HOOKS, capture_output=True, text=True, timeout=30
    )

    assert result.returncode == 0, result.stderr
