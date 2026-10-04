---
name: memory-base-connect-agent
description: "Use when the user wants to connect a new agent to memory-base, change which author slugs a key may act as, or disconnect an agent. Gives each agent its own key and author slug, so each agent owns its own profile. The user runs every step that creates or holds a key."
version: 1.0.0
metadata:
  hermes:
    tags: [memory-base, setup, keys, profile]
---

# Connect an agent to memory-base

Each agent has one author slug and one key. The slug names the agent's notes and its
profile (`self` and `user` parts). The key's authors list holds that slug, so the agent
can write only its own notes and its own profile.

```
 user: keys new <slug> --author <slug>  ──►  key (authors = {<slug>})  ──►  agent's env file
                                                                               │
 agent's client config: url + key + owner = <slug>  ◄──────────────────────────┘
                                                                               │
 session start ──► GET /profiles?owner=<slug> ──► "Memory: standing profile for <slug>"
 save_memory / update_my_profile / propose_user_profile ──► author or owner = <slug>
```

## Rules

- The user runs every command that mints, prints, or reads a key. You give the user the
  command; the user runs it with `! <command>`.
- Never read, print, or paste a key. Never open a file that holds a key.
- You may copy integration files and edit client settings that hold no key, when the
  user asks.

## 1. Pick the slug

The slug matches `^[a-z0-9][a-z0-9-]{0,39}$` and is unique per agent, for example
`claude-code` or `codex`. `user` and `consolidator` are reserved: `user` is the user's
approval key, `consolidator` is the consolidation agent.

## 2. Mint the key (user, on the server host, in the repo)

The command writes the key straight into the agent's env file and never shows it:

```
(umask 177; uv run python -m memory_base.serve.keys new <slug> --author <slug> \
  | tail -n 1 | sed 's/^/MEMORY_BASE_API_KEY=/' >> <env file>)
```

`new` prints the key once, on its last line. `<env file>` is the client's file from the
table in step 3. When the client runs on another machine, copy the file there (for
example with `scp`) without opening it.

Add `--admin` only when the agent must read private namespaces that its label does not
own, such as the user's personal namespace. An admin key can also rewrite any label's
authors over REST, so with `--admin` the one-slug limit prevents mistakes, not a
determined agent.

## 3. Configure the client

| client | env file (step 2) | owner | MCP server |
|---|---|---|---|
| Claude Code | `~/.config/memory-base/env` | `MEMORY_BASE_AUTHOR=<slug>` in the process environment, for example the `env` block of `~/.claude/settings.json` (default `claude-code`) | `claude mcp add --transport http memory-base <mcp url> --header "X-API-Key: $(sed -n 's/^MEMORY_BASE_API_KEY=//p' ~/.config/memory-base/env)"` |
| Hermes | `$HERMES_HOME/.env` | `memory.memory_base.owner: <slug>` in `$HERMES_HOME/config.yaml` | an `mcp_servers` entry with the header `X-API-Key` set to the same key |
| other MCP client | the client's own secret store | the client's session-start hook, if any | the MCP url with the header `X-API-Key` |

Install the client's integration once: the Claude Code hooks as documented in
`integrations/claude_code/prefetch_hook.py`, the Hermes provider as documented in
`integrations/hermes/README.md`. Install the `memory-profile-approval` skill for the
agent as documented in `integrations/profile_approval/README.md`.

## 4. Check the connection

1. The user restarts the client.
2. The session-start context prints `Memory: standing profile for <slug>.`, then
   `## user (v0)` and `## self (v0)`, each `(empty)` for a new agent.
3. The user runs `uv run python -m memory_base.serve.keys authors <slug>`. It prints
   `<slug>: <slug>`.

The agent then writes its `self` part with `update_my_profile` and proposes its `user`
part with `propose_user_profile`.

## Change or remove an agent

- Show a label's authors: `keys authors <label>`.
- Replace a label's authors: `keys authors <label> <slug> [<slug>...]`. The command
  replaces the whole list on every active key of the label.
- Disconnect an agent: `keys list` shows each key's hash prefix. `keys revoke <prefix>`
  revokes the key. The agent's notes and profile stay stored.
