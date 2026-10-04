# Profile approval

An agent's `user` profile part changes only when the user approves the agent's proposal
with a key that carries the `user` author. Two pieces serve that step:

- `mb_profile.py` — the user's command-line tool (Python standard library only).
- `../skills/memory-profile-approval/SKILL.md` — the skill that tells an agent to hand a
  pending proposal to the user and never to decide it itself.

```
 agent ── propose_user_profile ──► pending proposal
 session start ──► "proposal <id> awaits the user's approval"
 user  ── mb_profile.py show <id>    ──► proposal, reason, diff against the current user part
 user  ── mb_profile.py approve <id> ──► next user version   (or reject <id>)
```

## Commands

| command | request | prints |
|---|---|---|
| `pending [--owner OWNER]` | `GET /profiles/user/proposals?status=pending` (every owner unless `--owner`) | one entry per pending proposal: id, owner, base version, created time, reason |
| `show ID` | `GET /profiles/user/proposals/ID` | the proposal, its status and decision, the reason, and a unified diff against the owner's current user content; a warning when a pending proposal's base is not the current user version (approval would be refused as stale) |
| `approve ID [--note TEXT]` | `POST /profiles/user/proposals/ID/approve` | the new user version |
| `reject ID [--note TEXT]` | `POST /profiles/user/proposals/ID/reject` | the rejection |

A decision without `--note` sends `{}`. Every request has a 10-second timeout. A server
refusal, a transport failure, or a malformed response prints one `error:` line on stderr
and exits 1; the key is never printed.

## Configuration

| setting | source | default |
|---|---|---|
| `MEMORY_BASE_USER_ENV` | environment | `~/.config/memory-base/user.env` (`~` expanded) |
| `MEMORY_BASE_URL` | environment, else the user env file | `http://127.0.0.1:8010` |
| `MEMORY_BASE_USER_KEY` | environment, else the user env file | none: the CLI exits 1 before any request |

The user env file holds `NAME=value` lines. It is read as data and never executed: blank
lines and `#` comments are skipped, names other than the two settings are ignored, values
are trimmed, and nothing is expanded. A non-empty environment value wins over the file.
The CLI never reads `MEMORY_BASE_API_KEY` or the agents' env file.

## Install

On every machine where the user approves:

```sh
mkdir -p ~/.config/memory-base
cp integrations/profile_approval/mb_profile.py ~/.config/memory-base/mb_profile.py
(umask 177; printf 'MEMORY_BASE_URL=http://127.0.0.1:8010\n' > ~/.config/memory-base/user.env
 uv run python -m memory_base.serve.access.keys new <user label> --author user \
   | tail -n 1 | sed 's/^/MEMORY_BASE_USER_KEY=/' >> ~/.config/memory-base/user.env)
```

The `keys` command runs once, on the server host, in the repo. It mints the user's key
with the `user` author and writes the key into the file without showing it. On another
machine, copy that file over (for example with `scp`) without opening it.

The session-start notice names `~/.config/memory-base/mb_profile.py`, so install the CLI
at that path.

Install the skill for each agent:

```sh
mkdir -p ~/.claude/skills
cp -r integrations/skills/memory-profile-approval ~/.claude/skills/memory-profile-approval
cp -r integrations/skills/memory-profile-approval "$HERMES_HOME/skills/memory-profile-approval"
```

The second line targets the Hermes profile whose `config.yaml` sets `memory.memory_base.owner`.
