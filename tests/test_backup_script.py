"""scripts/backup.sh: gzipped pg_dump with pruning, driven through a stubbed docker CLI."""

from __future__ import annotations

import gzip
import os
import pathlib
import subprocess

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "backup.sh"
FAKE_DUMP_LINE = "CREATE TABLE fake;"


def _run(tmp_path: pathlib.Path, backup_dir: pathlib.Path, *args: str, docker_exit: int = 0):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        f"#!/usr/bin/env bash\necho '{FAKE_DUMP_LINE}'\nexit {docker_exit}\n"
    )
    fake_docker.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    return subprocess.run(
        [str(SCRIPT), str(backup_dir), *args],
        env=env,
        capture_output=True,
        text=True,
    )


def test_backup_writes_a_gzipped_dump(tmp_path):
    backup_dir = tmp_path / "backups"
    result = _run(tmp_path, backup_dir)
    assert result.returncode == 0, result.stderr
    out = pathlib.Path(result.stdout.strip())
    assert out.parent == backup_dir
    assert out.name.startswith("memory_base_") and out.name.endswith(".sql.gz")
    with gzip.open(out, "rt") as f:
        assert FAKE_DUMP_LINE in f.read()


def test_backup_prunes_old_dumps_beyond_keep(tmp_path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    for i, name in enumerate(
        ["memory_base_20200101_000000.sql.gz", "memory_base_20200102_000000.sql.gz"]
    ):
        old = backup_dir / name
        old.write_bytes(b"old")
        os.utime(old, (1000 + i, 1000 + i))
    result = _run(tmp_path, backup_dir, "2")
    assert result.returncode == 0, result.stderr
    remaining = sorted(p.name for p in backup_dir.glob("memory_base_*.sql.gz"))
    assert len(remaining) == 2
    assert "memory_base_20200101_000000.sql.gz" not in remaining
    assert "memory_base_20200102_000000.sql.gz" in remaining


def test_backup_failure_leaves_no_partial_dump(tmp_path):
    backup_dir = tmp_path / "backups"
    result = _run(tmp_path, backup_dir, docker_exit=1)
    assert result.returncode != 0
    assert list(backup_dir.glob("*.sql.gz")) == []


def test_backup_requires_a_backup_dir_argument():
    result = subprocess.run([str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "usage" in result.stderr
