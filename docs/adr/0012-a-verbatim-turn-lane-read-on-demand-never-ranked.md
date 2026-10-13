# ADR-0012: A verbatim turn lane, read on demand, never ranked

## Status

Accepted; supersedes ADR-0005 and defers the facts part of ADR-0011

## Context

A note is what an agent made of a conversation. The agent's reading can be wrong, and the
owner finds it wrong often enough to want the words themselves: to check a note against
what was said, and to let a later agent that curates notes with a different judgment see
the source rather than the first agent's reading. A second reader is a session a week old
whose decision the current session half remembers; its context is in another agent's
local store. Claude Code keeps one JSONL file per session on the machine it ran on, and
the Hermes personal agent keeps its sessions in its own SQLite store with a search of its
own; neither reads the other, and neither store leaves its machine. A third reader is the
personal agent itself, whose reasoning is part of the record it grows its memory from.

An earlier capture lane stored a session's text turns at session end and had a server job
distill them into notes (ADR-0005 records its removal). It died because nothing read the
stored turns: the only reads went through a link from a note, so a note never written
could not be recovered, and the lane was a privacy surface with no reader. The readers
above do not go through notes. They go by time, by session, and by literal keyword.

## Decision

memory-base keeps a verbatim record of the user's and the assistant's messages, one row
per message, in a lane of its own. The lane is read on demand and is never ranked.

### What is stored

- `turns` holds `id` (a uuid the client chose), `namespace`, `origin` (`claude_code` or
  `hermes`), `session_id` (the client's own session id), `seq` (assigned by the server
  per origin and session), `role` (`user` or `assistant`), `content`, `reasoning`
  (nullable), `repo` (nullable, the `repo:<host>/<path>` scope the session hook derives
  from the working directory's origin), `created_at`. `(origin, session_id, seq)` is
  unique. Content is indexed by trigram; nothing is embedded.
- A message is stored as the client delivered it. The user's text keeps anything the
  user pasted. Tool calls, tool results, attachments, and context the client injected are
  not messages and are not sent. The server scans content and reasoning for credentials
  and refuses a message that holds one; it stores nothing partial.
- Each of content and reasoning is at most 64,000 characters; a longer message is refused
  and not stored.
- The record is as complete as the client makes it. Claude Code delivers the user's
  prompt and the final answer of a turn, without the model's thinking. Hermes delivers
  what it hands its memory providers: the user's text with skill scaffolding removed and
  secrets masked, the answer, and the reasoning of the turn's assistant entries.
- A message is stored once. A client that retries sends the same `id`; the server answers
  a replay with the row it already holds and refuses a different payload under the same id.
- Retention is unlimited. A session is deleted whole through the admin path.

### How it is captured

- The capture is a client hook, not a server job. Claude Code's plugin posts the prompt
  when it is submitted and the final answer when the turn stops; prompts that are slash or
  shell commands are not posted. The Hermes memory provider posts both halves of a turn
  when Hermes syncs it, and only for a primary session: never for a cron or subagent run.
- A capture never blocks the agent. The hook has a short deadline, exits without error
  whatever happens, and after a failure skips the attempt for a minute. The user sees one
  line when the server becomes unreachable and one when it is back, and nothing enters the
  model's context. Nothing is spooled locally; a message that could not be sent is lost.
- Turns land in the namespace the plugin or provider is configured with, by default the
  key's home namespace. Personal conversation belongs in a private namespace.

### How it is read

- `list_sessions` lists sessions by time window, origin, repository scope, and namespace,
  each with its first user message as a title.
- `read_session` reads one session in order; with `contains`, it returns the matching
  messages and one neighbour on each side, matches marked, so a decision is read with the
  proposal before it and the qualification after it.
- `search_transcript` finds messages holding a literal substring, newest first, each with
  a window around the match and its position, so `read_session` can expand it. Substring
  matching is what a reader who remembers a word needs, and it holds for Korean words
  with particles attached.
- A note may name the session it was distilled from, as `source_session`
  `{origin, session_id}` in its metadata, reported by the client and not verified. A
  curator reads that session around the note.
- `search_memory` never returns a turn, and nothing prefetches turns into a session. The
  lane is not memory; it is the record memory is checked against.

### Delivery

- The Claude Code integration is a plugin. Enabling it asks for the server's REST and MCP
  URLs, the key, the author, and an optional namespace, through the plugin's user
  configuration; the key is held in the client's secure store. The plugin carries the
  hooks, the MCP server entry, and an `init` skill that reports the connection state the
  hooks last recorded and names what to change.
- The Hermes provider takes the same values through Hermes's provider configuration.

## Consequences

- Notes stay distilled, and distillation stays the writer's job. The lane adds no writer
  and no gate; it adds a way to check the writer.
- The lane answers "what was said" and "what did we decide last week", across agents and
  machines. It does not answer "what is true now": that is a note's job, and a value that
  changed is still a note to supersede.
- Facts as rows keyed by entity and attribute (ADR-0011) are deferred, not rejected. The
  entity registry stays. When facts are built, the quote a fact carries can be a reference
  into this lane rather than text the writer copied.
- The record is partial by construction. A client that does not expose the model's
  thinking cannot deliver it, and a client that masks secrets before handing a turn to
  providers delivers the masked form. The tool descriptions say what each origin delivers.
- Capture is best effort. An unreachable server loses the messages sent while it was
  down, by choice: the agent's work must not depend on the server.
- A private namespace keeps a conversation from other keys. It does not keep it from an
  admin key or from a key sharing the owner's label.
