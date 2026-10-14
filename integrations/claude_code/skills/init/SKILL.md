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
| card reachable | `card_reachable` (the author's card fetch), with `error` when false |
| checked at | `checked_at` |
| author | `author` |
| REST URL | `url` |
| namespace | `namespace`, or `default` when null |

## 3. Find leftover hand-installed entries

The plugin provides the hooks and the MCP server, so a hand-installed copy runs twice.
Never open `~/.claude/settings.json` or `~/.claude.json`, with Read or any other tool: a
hand-installed MCP entry carries the key in its headers. Run these Bash commands, which
print counts and names only:

```
grep -c "memory_base_" ~/.claude/settings.json
python3 -c "import json,os;d=json.load(open(os.path.expanduser('~/.claude.json')));print(list(d.get('mcpServers',{})))"
```

- A count above 0 means `~/.claude/settings.json` holds a hand-installed hook entry
  (UserPromptSubmit or SessionStart running a `memory_base_*.py` script from
  `~/.claude/hooks/`). Tell the user to delete those entries.
- A list containing `memory-base` means `~/.claude.json` holds a hand-installed MCP
  entry. Tell the user to delete it (`claude mcp remove memory-base`).

Say why: the plugin provides both, and duplicates run twice. If both outputs are clean,
say so. Do not edit either file.

## 4. How to change values

The plugin's rows in `/config` change the URLs, the author, the user entity, and the namespace; the key is
held in the secure store, never in a file. Changes apply to the next session.
