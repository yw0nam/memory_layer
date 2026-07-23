"""Registered source adapters."""

from memory_base.adapters.base import SourceAdapter
from memory_base.adapters.claude_code import ClaudeCodeAdapter

ADAPTERS: dict[str, SourceAdapter] = {"claude_code": ClaudeCodeAdapter()}
