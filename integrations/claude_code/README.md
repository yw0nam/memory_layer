# memory-base plugin for Claude Code

A Claude Code plugin that connects a session to memory-base: the agent's and the user's
cards and the repo's handoffs at session start, relevant notes on every prompt, and the verbatim turn
record kept beside them (the turn lane).

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
| Author | the author slug this agent writes under and reads by default; sent as the `X-Author` header; its card is delivered at session start | `claude-code` |
| User entity | slug of the user's own entity (kind `person`), as `list_entities` shows it; its card is delivered at session start; empty: no user card | none |
| Namespace | a registered namespace for this agent's turns and for the facts on its session-start cards; empty means `default`; the server refuses every turn in an unregistered namespace | none |

The namespace does not choose where `save_memory` writes and does not narrow the
per-prompt search; reads are scoped by the author. The `connect-agent` skill walks through
picking the author, the user entity, and the namespace, registering a new namespace, and
minting a key for an agent.

## What the plugin provides

| part | file | when | does |
|---|---|---|---|
| SessionStart hook | `hooks/session_start.py` | start, resume, clear, compaction | prints one `<memory-context>` fence: the author's card, the user's card when `User entity` is set, a `Session: claude_code <id>` line, the repo's pending handoffs; records the connection in `status.json` |
| UserPromptSubmit hook | `hooks/prompt_submit.py` | every prompt | appends the prompt to the turn lane (not `/` or `!` prompts); searches notes and prints fresh, relevant hits |
| Stop hook | `hooks/stop.py` | end of each turn | appends the final answer to the turn lane |
| MCP server | `.mcp.json` | session | the memory tools, with the key and author from the plugin options as the `X-API-Key` and `X-Author` headers |
| `/memory-base:init` | `skills/init` | on demand | reports `status.json` and flags old hand-installed entries |
| `/memory-base:connect-agent` | `skills/connect-agent` | on demand | connects, changes, or disconnects an agent |

The search spans every namespace and sends no `author` filter, so it returns this agent's
own notes and the shared notes; a note line names its author only when another agent
wrote it. Hooks are stdlib Python, fail open (exit 0),
and run synchronously.

## Turn capture

```
 prompt / final answer ──► POST /turns {id (uuid), origin: claude_code, session_id,
                                        role, content, repo?, namespace?}
```

`repo` is the `repo:<host>/<path>` scope of the working directory's `origin` remote and is
left out when there is none. Content is sent whole; the server refuses a turn over 64,000
characters, a credential, or a bad key.

The user sees these lines: the unreachable and recovery lines once per outage, a refusal
line once per status per session. None enters the model's context.

| condition | line |
|---|---|
| server unreachable (first time) | `memory-base unreachable; turns are not being saved` |
| first success after a failure | `memory-base is reachable again; turns are being saved` |
| server refusal (once per status) | `memory-base refused this turn (<status>: <message>)`; a 401 adds `; check the API key in /config` |

After an unreachable failure the session makes no request for 60 seconds. Nothing is
spooled: turns that fail are not saved.

## Data directory

`${CLAUDE_PLUGIN_DATA}` (`~/.claude/plugins/data/<id>/`) holds `status.json` (configured,
card reachable, URL, author, namespace, check time, error) and `sessions/<session_id>.json`
(the failure marker). Without `CLAUDE_PLUGIN_DATA` the hooks use a `memory_base_plugin`
directory under the system temp directory. Each prompt appends one row to
`MEMORY_PREFETCH_LOG` (default `~/.claude/memory_prefetch_log.jsonl`).

## Check

Run `/memory-base:init` to read the recorded connection and to find hand-installed
`memory_base_*` hook entries in `~/.claude/settings.json` or a `memory-base` entry in
`~/.claude.json`; remove them, because the plugin provides both.
