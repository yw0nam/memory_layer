---
name: init
description: "Check the memory-base plugin: the connection the hooks last recorded, the author and URLs in use, and leftover hand-installed entries"
---

# Check the memory-base plugin

Report the state of the plugin. Never ask for, print, or write the API key; the key is in
the secure store and this skill cannot read it.

## 1. Read the recorded connection

Read `${CLAUDE_PLUGIN_DATA}/status.json` with the Read tool. The SessionStart hook writes
it at every session start, resume, clear, and compaction.

If the file is missing, say: "no session has run with the plugin yet; start a new session
or run /clear", and continue with step 3.

## 2. Report it

Show one short table:

| field | value |
|---|---|
| configured | `configured` (false: the URL or the key is empty) |
| reachable | `reachable`, with `error` when false |
| checked at | `checked_at` |
| author | `author` |
| REST URL | `url` |
| namespace | `namespace`, or "the key's home namespace" when null |

## 3. Find leftover hand-installed entries

Read `~/.claude/settings.json` and `~/.claude.json`. The plugin provides the hooks and the
MCP server, so a hand-installed copy runs twice. Tell the user exactly which entries to
delete, and why:

- a `hooks` entry (UserPromptSubmit or SessionStart) whose command runs a `memory_base_*.py`
  script from `~/.claude/hooks/`
- a `mcpServers["memory-base"]` entry that is not the plugin's

Do not edit either file. If none is present, say so.

## 4. How to change values

The plugin's rows in `/config` change the URLs, the author, and the namespace; the key is
held in the secure store, never in a file. Changes apply to the next session.
