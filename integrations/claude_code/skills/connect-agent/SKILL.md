---
name: connect-agent
description: "Use when the user wants to connect a new agent to memory-base or disconnect an agent. Gives each agent its own key and author slug, so each agent has its own entity card. The user runs every step that creates or holds a key."
---

# Connect an agent to memory-base

Each agent has one author slug and one key. The slug is sent as the `X-Author` header on
every request: it names the agent's notes and its entity, whose description holds the
user's directions to it, and it is the agent's default read scope — its reads return its
own notes and the shared notes. The key is a credential for
the network boundary and grants nothing else. Each agent holds its own key, so revoking it
disconnects that agent alone.

```
 user: keys new <label>  ──►  key  ──►  agent's secret store
                                              │
 agent's client config: url + key + author = <slug>  ◄─┘
   (+ user_entity = <the user's slug>)                 │
 every request ──► X-API-Key + X-Author: <slug>
 session start ──► GET /entities/<slug> ──► "Memory: your directions, <slug> (...)"
               ──► GET /entities/<user_entity> ──► "Memory: the user, <slug> (...)"
 save_memory ──► stored under <slug>;  register_entity ──► changes the description, with consent
```

## Rules

- The user runs every command that mints, prints, or reads a key. You give the user the
  command; the user runs it with `! <command>`.
- Never read, print, or paste a key. Never open a file that holds a key.
- You may edit client settings that hold no key, when the user asks.

## 1. Pick the slug

The slug matches `^[a-z0-9][a-z0-9-]{0,39}$` and is unique per agent, for example
`claude-code` or `codex`.

## 2. Pick the user entity and the namespace

| value | what it is | how to pick it | empty |
|---|---|---|---|
| user entity | the slug of the entity that is the user, kind `person`; its card is delivered at session start as `Memory: the user, <slug> (...)` | call `list_entities` and take the slug of the user's `person` entity; when none exists, register one with `register_entity`, with the user's consent | no user card |
| namespace | a registered label for filtering, not an access boundary | the agent's slug keeps this agent's turns apart from other agents' turns | `default` |

The namespace changes two things only:

- the turns the client posts land in it, and `list_sessions`, `read_session`, and
  `search_transcript` filter by it;
- the facts on both session-start cards are read from it alone; empty, the cards read
  every namespace.

It does not choose where `save_memory` writes: a save without `namespace` lands in
`default`. Reads are scoped by the author, not by the namespace.

`default` always exists. Any other namespace is registered in step 3, before the client
uses it: the server refuses every turn in an unregistered namespace with
`400 unregistered namespace: <name>`.

## 3. Mint the key (user, on the server host, in the repo)

The label names the client that holds the key; the slug is often the same word. The user
runs the command in their own terminal:

```
uv run python -m memory_base.serve.access.keys new <label>
```

`new` prints the key once, on its last line. For Claude Code the user pastes it into the
plugin's configuration prompt (enabling the plugin asks for it; later, `/config`, then the
memory-base rows). The key goes into no file, and the user never pastes it into the chat.
For Hermes or another client, the user stores it in that client's own secret store.

A new namespace is registered with that key. The user runs this in their own terminal,
pastes the key at the silent prompt, and replaces `<REST URL>`, `<slug>`, and `<namespace>`:

```
read -rs K && curl -s -w '\n%{http_code}\n' -X POST <REST URL>/namespaces \
  -H 'Content-Type: application/json' -H "X-API-Key: $K" -H "X-Author: <slug>" \
  -d '{"name":"<namespace>"}'; unset K
```

`201` registers the namespace; `409` means it is already registered. The name matches
`^[a-z0-9_-]{1,64}$`.

## 4. Configure the client

| client | key (step 3) | author, user entity, and namespace | MCP server |
|---|---|---|---|
| Claude Code | the plugin's API key option, held in the secure store | the plugin's Author option (default `claude-code`), User entity option, and Namespace option, changed in `/config` | provided by the plugin |
| Hermes | `$HERMES_HOME/.env` | `memory.memory_base.author: <slug>`, `memory.memory_base.user_entity: <slug>`, and `memory.memory_base.namespace: <namespace>` in `$HERMES_HOME/config.yaml` | an `mcp_servers` entry with the headers `X-API-Key` and `X-Author` set to the same key and slug |
| other MCP client | the client's own secret store | the client's session-start hook, if any | the MCP url with the `X-API-Key` and `X-Author` headers |

Install the client's integration once: the Claude Code plugin as documented in
`integrations/claude_code/README.md`, the Hermes provider as documented in
`integrations/hermes/README.md`.

## 5. Check the connection

1. The user restarts the client.
2. The session-start context prints `Memory: your directions, <slug> (...)` with the
   description of the agent's entity, and, when a user entity is set, `Memory: the user,
   <user_entity> (...)`. A new agent has no entity yet: the context prints `Memory: your
   directions could not be fetched.` until one is registered.
3. With a namespace set, the first turn shows no `memory-base refused` warning;
   a `400: unregistered namespace` line means step 3's registration is missing.
4. The user runs `uv run python -m memory_base.serve.access.keys list`. The new key shows
   as `active`.

An agent's directions live in the description of its own entity. The agent changes them
with `register_entity`, with the user's consent. The description may carry `Remember` and
`Don't remember` lines: rules that narrow or widen what the agent keeps with
`save_memory`, never facts about the user. A companion agent's description may say
`Remember: moments we share`; a user's wish such as `Don't remember: my spending` goes
there too.

## Remove an agent

`keys list` shows each key's hash prefix. `keys revoke <prefix>` revokes the key. The
agent's notes and entity stay stored.
