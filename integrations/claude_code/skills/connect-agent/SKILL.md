---
name: connect-agent
description: "Use when the user wants to connect a new agent to memory-base or disconnect an agent. Gives each agent its own key and author slug. The user runs every step that creates or holds a key."
---

# Connect an agent to memory-base

Each agent has one author slug and one key. The slug is sent as the `X-Author` header on
every request: it names the agent's notes and is the agent's default read scope — its
reads return its own notes and the shared notes. The key is a credential for
the network boundary and grants nothing else. Each agent holds its own key, so revoking it
disconnects that agent alone.

```
 user: keys new <label>  ──►  key  ──►  agent's secret store
                                              │
 agent's client config: url + key + author = <slug>  ◄─┘
 every request ──► X-API-Key + X-Author: <slug>
 save_memory ──► stored under <slug>
```

## Rules

- The user runs every command that mints, prints, or reads a key. You give the user the
  command; the user runs it with `! <command>`.
- Never read, print, or paste a key. Never open a file that holds a key.
- You may edit client settings that hold no key, when the user asks.

## 1. Pick the slug

The slug matches `^[a-z0-9][a-z0-9-]{0,39}$` and is unique per agent, for example
`claude-code` or `codex`.

## 2. Mint the key (user, on the server host, in the repo)

The label names the client that holds the key; the slug is often the same word. The user
runs the command in their own terminal:

```
uv run python -m memory_base.serve.access.keys new <label>
```

`new` prints the key once, on its last line. For Claude Code the user pastes it into the
plugin's configuration prompt (enabling the plugin asks for it; later, `/config`, then the
memory-base rows). The key goes into no file, and the user never pastes it into the chat.
For Hermes or another client, the user stores it in that client's own secret store.

## 3. Configure the client

| client | key (step 2) | author | MCP server |
|---|---|---|---|
| Claude Code | the plugin's API key option, held in the secure store | the plugin's Author option (default `claude-code`), changed in `/config` | provided by the plugin |
| Hermes | the client's own secret store | the `X-Author` header of its MCP entry | an `mcp_servers` entry with the headers `X-API-Key` and `X-Author` set to the same key and slug |
| other MCP client | the client's own secret store | the `X-Author` header | the MCP url with the `X-API-Key` and `X-Author` headers |

Install the Claude Code plugin as documented in `integrations/claude_code/README.md`.

## 4. Check the connection

1. The user restarts the client.
2. In a repo with pending handoffs, the session-start context lists them under
   `Memory: pending handoffs for this repo.`; without handoffs it prints nothing.
3. The user runs `uv run python -m memory_base.serve.access.keys list`. The new key shows
   as `active`.

## Remove an agent

`keys list` shows each key's hash prefix. `keys revoke <prefix>` revokes the key. The
agent's notes stay stored.
