# memory-base plugin for Claude Code

A Claude Code plugin that connects a session to memory-base: the repo's handoffs and decision
titles at session start, decisions and notes recalled for each prompt, the session record of each
session's transcript, and the memory tools over MCP.

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
| SessionStart hook | `hooks/session_start.py` | start, resume, clear, compaction | prints one `<memory-context>` fence: the repo's open handoffs, a line counting its completed handoffs, the repository scope line, and the newest active decision titles; records the connection in `status.json` |
| UserPromptSubmit hook | `hooks/prompt_recall.py` | every prompt | prints one `<memory-recall>` fence of the decisions and notes this session has not been shown yet |
| Stop hook | `hooks/session_tail.py` | end of each assistant turn | saves the tail of the assistant's last message for the next prompt's query; prints nothing |
| Session record hook | `hooks/session_record.py` | Stop, SessionEnd, start (the start sweeps unfinished transcripts) | starts a detached uploader that posts the session's new transcript lines to `/session_events`; prints nothing |
| MCP server | `.mcp.json` | session | the memory tools, with the key and author from the plugin options as the `X-API-Key` and `X-Author` headers |
| `/memory-base:init` | `skills/init` | on demand | reports `status.json` and flags old hand-installed entries |
| `/memory-base:connect-agent` | `skills/connect-agent` | on demand | connects, changes, or disconnects an agent |
| `/memory-base:save-notes` | `skills/save-notes` | on request, before `/compact` | saves the session's lessons as notes, superseding stale ones; lists the owner's decisions as drafts |
| `/memory-base:send-handoff` | `skills/send-handoff` | on request, before `/compact` with work unfinished | sends the repository's work state and next steps, and cancels the caller's stale handoffs |
| `/memory-base:record-decision` | `skills/record-decision` | when the owner says to record, supersede, or retire | checks the active decisions, records each one, and shows its D-number |

Hooks are stdlib Python, fail open (exit 0), and run synchronously.

### SessionStart

The hook prints these blocks in order; a block with nothing to show is left out.

- **Handoffs.** Open handoffs of the repository from `GET /messages?purpose=handoff`: at most
  10 lines from the first 100 the server lists, each with its date, subject, status and id. Nothing
  is claimed. One line counts the completed handoffs among those 100.
- **Scope.** `Repository scope for decisions: repo:<host>/<path>`, from the `origin` remote.
- **Decisions.** The repository's active decisions from `GET /decisions` (200 fetched): titles
  only, the newest 50 as `D-<number> (<decided_on>) <title>`, then `and N more: list_decisions`
  where N counts the fetched rows past 50.

Without a portable `origin` the hook prints nothing. A portable origin names a dotted, non-local host and a path:
a local path, `localhost`, or a private or loopback address is not. A failed fetch drops
only its own block; the scope line still prints. Titles and subjects are defused, so they cannot
close the fence.

### UserPromptSubmit

The query is the last 200 characters of the previous assistant text and the prompt. The previous
text is the session's tail file from the Stop hook (which keeps at most 1500 characters); without it,
the last assistant text of the transcript, read from its last 256 KiB. The prompt has its complete `<memory-context>` and
`<memory-recall>` fences removed and is cut to 2000 characters. The two are joined by a blank line.
A prompt that starts with `/`, is empty, or is under 20 characters with no previous text makes no call.

The hook sends one `POST /recall` with `notes` and `decisions`, `top_k` 5 for both and `min_score` 0.5 for notes and 0.7 for decisions,
and the repository's `scope` when the origin is portable. The server applies those floors; the
hook does not re-score.

The hook gives the call until 2.5 seconds after its start. A late or failed call shows nothing and
logs `errors=unavailable`. The process exits at 3.0 seconds whatever it is doing, and the
`hooks.json` timeout is 4 seconds.

Decisions the session has already been shown are dropped first, then the rest are capped: at most
3 decisions, then at most 3 notes. The fence is one `<memory-recall>` block: a header line, then
the decisions in rank order, then the notes. The decisions are titles only, under one line saying to
read one with `get_decision`: each is `D-<number> <title> (<decided_on>)`, with no decision text. A
note shows its id, date, tags and text. Entries are whole and the fence is at most 4000 characters; the first entry that does
not fit ends the list. The shown keys are merged into the session's shown set.

### Stop

Writes the last 1500 characters of `last_assistant_message` to the session's tail file, replacing it
atomically. It prints nothing.

### Session record

Stop and SessionEnd write the session's checkpoint if it is missing, then start a detached uploader
for its transcript. SessionStart starts the sweep. The hook prints nothing and exits at once.

The uploader reads the main transcript only, never a subagent's, from the checkpoint's offset and
only complete lines. Its text and tool events go to `POST /session_events` in chunks of at most 500
events and about 1,900,000 bytes. Tool input and results are cut at 2,000 characters and other text at
64,000. An event's `cwd` is its transcript entry's own, else the checkpoint's. A chunk the server
answers with 400 is resent line by line, and a line that still gets 400 is skipped and logged; any
other failure stops that session's run and leaves the checkpoint at the last acknowledged line, so
the next run resumes there. A second uploader for a session exits while the first holds its lock.
The sweep goes on to the next session when one fails.

The record is the session verbatim, secrets included, and it is readable through the MCP tools
`list_sessions`, `read_session`, and `search_transcript` by the same author from any machine
([ADR-0017](../../docs/adr/0017-the-session-record-lane.md)). It never reaches `/search` or `/recall`.

### When memory-base is not configured

When the REST URL, API key or author option is empty, no call is made and nothing is saved. SessionStart
prints only the scope line, and only when the origin is portable. The UserPromptSubmit and Stop hooks
print nothing, and the session record hook uploads nothing. SessionStart records `configured: false` in `status.json`.

## Data directory

`${CLAUDE_PLUGIN_DATA}` (`~/.claude/plugins/data/<id>/`) holds:

- `status.json`: configured, URL, author, reachable, check time, error. SessionStart writes it.
- `sessions/<session>.json`: the keys a session was shown (`decision:<scope>#<number>` and note ids),
  writers hold `sessions/<session>.lock` while they merge.
  SessionStart removes both on `clear` and `compact`.
- `sessions/<session>.tail`: the last 1500 characters of the assistant's last message, replaced each
  turn by the Stop hook and removed by SessionStart on `clear` and `compact`. It never leaves the machine.
- `records/<session>.json`: the session record checkpoint: the transcript path, session id, cwd, the
  file's device and inode, the acknowledged offset, and the sha256 of the last acknowledged line.
  `records/<session>.lock` is held by an uploader while it runs. Neither is removed by a hook; the
  sweep removes both once the transcript file is gone and no uploader holds the lock.
- `recall.log`: one line per prompt that reaches the lookup: time, elapsed milliseconds, the keys shown,
  and the errors. It holds no prompt text, is appended to, and is never rotated.
- `records.log`: one line per post or failure: time, session, counts, event ids, and errors. It
  never holds event content, is appended to, and is never rotated.

Without `CLAUDE_PLUGIN_DATA` the hooks use a `memory_base_plugin` directory under the system temp directory.

## Scripts

All three live in the repository's `scripts/` directory and are run by hand, not by the plugin.

- `scripts/import_decision_issues.py --repo OWNER/NAME --url URL [--plugin-data DIR] [--dry-run]`:
  one-time, resumable import of a repository's closed `decision` issues into its decision lane. A
  decision's text is the last comment with a `Decision` heading and every comment after it. The
  author in `MEMORY_BASE_AUTHOR` must equal the `author` in the plugin's `status.json`. It refuses a
  scope that already lists 200 decisions. A dry run prints each issue's title and text length.
- `scripts/backfill_sessions.py --url URL [--projects DIR] [--since YYYY-MM-DD] [--state DIR] [--dry-run]`:
  one-time backfill of the main-session transcripts under `~/.claude/projects` into the session record, in
  modification order; `MEMORY_BASE_API_KEY` and `MEMORY_BASE_AUTHOR` must be set. Each transcript's checkpoints
  live in `--state` (default `~/.cache/memory-base-backfill`), so a rerun resumes and a new state directory resends all.
- `scripts/probe_recall.py`: the recall probe. `seed-notes --from URL --to URL` copies notes to a
  sandbox server (the target key is `MEMORY_BASE_TO_API_KEY`). `run --repo OWNER/NAME` sweeps the
  tail length (0, 200, 400, 800) and the decision floor on half of a local case file through
  `/recall`, freezes the best pair, and then every case goes through the hook with that pair.

## Check

Run `/memory-base:init` to read the recorded connection and to find hand-installed
`memory_base_*` hook entries in `~/.claude/settings.json` or a `memory-base` entry in
`~/.claude.json`; remove them, because the plugin provides both.
