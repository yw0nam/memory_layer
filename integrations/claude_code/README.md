# memory-base plugin for Claude Code

A Claude Code plugin that connects a session to memory-base: the repo's handoff headers at
session start, and the memory tools over MCP.

## Install

```
/plugin marketplace add yw0nam/memory_layer
/plugin install memory-base@memory_layer
```

Enabling the plugin asks for its options. Change them later in `/config`.

| option | meaning | default |
|---|---|---|
| REST URL | the REST API the hooks call | `http://127.0.0.1:8010` |
| MCP URL | the MCP server (streamable HTTP) | `http://localhost:8765/mcp` |
| API key | this agent's key, sent as `X-API-Key`; held in the secure store | none |
| Author | the author slug this agent writes under and reads by default; sent as the `X-Author` header | `claude-code` |

The `connect-agent` skill walks through picking the author and minting a key for an agent.

## What the plugin provides

| part | file | when | does |
|---|---|---|---|
| SessionStart hook | `hooks/session_start.py` | start, resume, clear, compaction | prints one `<memory-context>` fence with the repo's open handoffs and one line counting its completed handoffs, or nothing when there are none; records the connection in `status.json` |
| MCP server | `.mcp.json` | session | the memory tools, with the key and author from the plugin options as the `X-API-Key` and `X-Author` headers |
| `/memory-base:init` | `skills/init` | on demand | reports `status.json` and flags old hand-installed entries |
| `/memory-base:connect-agent` | `skills/connect-agent` | on demand | connects, changes, or disconnects an agent |

Hooks are stdlib Python, fail open (exit 0),
and run synchronously.

## Data directory

`${CLAUDE_PLUGIN_DATA}` (`~/.claude/plugins/data/<id>/`) holds `status.json` (configured,
reachable, URL, author, check time, error). Without `CLAUDE_PLUGIN_DATA` the hooks use a `memory_base_plugin`
directory under the system temp directory.

## Check

Run `/memory-base:init` to read the recorded connection and to find hand-installed
`memory_base_*` hook entries in `~/.claude/settings.json` or a `memory-base` entry in
`~/.claude.json`; remove them, because the plugin provides both.
