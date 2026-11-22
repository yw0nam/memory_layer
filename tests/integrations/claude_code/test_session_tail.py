"""The Stop hook that records the last assistant text — the real process."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from recall import tail

PLUGIN = Path(__file__).resolve().parents[3] / "integrations" / "claude_code"


def _run(tmp_path, payload, configured=True):
    entries = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]["Stop"]
    [hook] = [h for entry in entries for h in entry["hooks"] if "session_tail.py" in h["command"]]
    assert hook["timeout"] <= 3
    env = {
        **os.environ,
        "CLAUDE_PLUGIN_ROOT": str(PLUGIN),
        "CLAUDE_PLUGIN_OPTION_URL": "http://127.0.0.1:9",
        "CLAUDE_PLUGIN_OPTION_API_KEY": "key" if configured else "",
        "CLAUDE_PLUGIN_OPTION_AUTHOR": "claude-code",
        "CLAUDE_PLUGIN_DATA": str(tmp_path / "data"),
    }
    return subprocess.run(
        hook["command"],
        shell=True,
        input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


def test_the_stop_hook_writes_the_tail_and_prints_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "data"))

    result = _run(tmp_path, {"session_id": "sess-1", "last_assistant_message": "Done. " * 400})

    assert (result.returncode, result.stdout) == (0, "")
    assert tail.previous("sess-1", "") == ("Done. " * 400)[-1500:]


def test_a_bad_payload_or_an_unconfigured_plugin_exits_zero_and_writes_nothing(tmp_path):
    for payload, configured in (
        ("not json", True),
        ({"session_id": "s", "last_assistant_message": "text"}, False),
    ):
        result = _run(tmp_path, payload, configured)
        assert (result.returncode, result.stdout) == (0, "")

    assert not list((tmp_path / "data").rglob("*.tail"))
