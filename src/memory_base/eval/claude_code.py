"""A headless Claude Code session as the eval harness's model client."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass

CALL_TIMEOUT_SECONDS = 120.0


@dataclass
class ClaudeCodeModel:
    """One headless Claude Code session per prompt, with nothing but the model and a system prompt."""

    model: str
    effort: str
    system_prompt: str
    timeout: float = CALL_TIMEOUT_SECONDS
    provider: str = "claude-code"

    @property
    def thinking(self) -> str:
        return f"effort {self.effort}"

    async def complete(self, prompt: str, *, max_tokens: int) -> tuple[str, int, int]:
        argv = (
            "claude", "-p", "--model", self.model, "--effort", self.effort,
            "--tools", "", "--setting-sources", "", "--strict-mcp-config",
            "--no-session-persistence", "--system-prompt", self.system_prompt,
            "--output-format", "json",
        )  # fmt: skip
        env = {**os.environ, "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1"}
        # An empty working directory keeps any CLAUDE.md out of the session.
        with tempfile.TemporaryDirectory() as cwd:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            out, err = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=self.timeout
            )
        if proc.returncode:
            raise RuntimeError(f"claude exited {proc.returncode}: {err.decode()[:300]}")
        result = json.loads(out)
        if result.get("is_error") or result.get("num_turns") != 1:
            raise RuntimeError(f"claude session did not end in one clean turn: {result}")
        if set(result.get("modelUsage", {})) != {self.model}:
            raise RuntimeError(f"claude answered with {list(result.get('modelUsage', {}))}")
        usage = result["usage"]
        in_tok = (
            usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
        )
        return result["result"].strip(), in_tok, usage.get("output_tokens", 0)


def strip_fence(text: str) -> str:
    """The reply without a surrounding ```json fence, which Claude adds without a JSON mode."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()
