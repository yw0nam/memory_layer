"""Registered source adapters.

Empty today: interactive agent consoles contribute through the MCP realtime
channel (save_memory), not batch ingestion. Future corpus adapters (e.g.
Slack) register here.
"""

from memory_base.adapters.base import SourceAdapter

ADAPTERS: dict[str, SourceAdapter] = {}
