---
name: memory-profile-approval
description: "Use when the session-start memory context says a proposed change to the user profile awaits the user's approval, or after you call propose_user_profile. Hands the approval to the user: they inspect the stored proposal and approve or reject it with their own key. You never approve, reject, or read their key."
version: 1.0.0
metadata:
  hermes:
    tags: [memory-base, profile, approval, user]
---

# Memory profile approval

Your profile in memory-base has two parts. You replace your `self` part with
`update_my_profile`. Your `user` part (how you know the user) changes only when the
user approves a proposal you submitted with `propose_user_profile`. The user decides
with a command-line tool and a key that only they hold.

## When a proposal is pending

The session-start memory context prints a notice like this one:

```
A proposed change to the user profile (proposal 12) awaits the user's approval. ...
```

The notice holds the proposal id and nothing else. Do these steps:

1. Tell the user that proposal `<id>` is waiting for them.
2. Ask the user to inspect it:

   ```
   ! python3 ~/.config/memory-base/mb_profile.py show <id>
   ```

   `show` prints the stored proposal, the reason, and a diff against the current user
   profile. It also warns when the proposal is stale.
3. Ask the user to decide:

   ```
   ! python3 ~/.config/memory-base/mb_profile.py approve <id>
   ! python3 ~/.config/memory-base/mb_profile.py reject <id> --note "why"
   ```

   To list every pending proposal: `! python3 ~/.config/memory-base/mb_profile.py pending`.

You can show your own preview of the change only when you have the exact text you
submitted in this session. Never write a preview from the notice, and never guess what
the proposal says.

## Rules

- Never run `approve` or `reject` yourself, in any form.
- Never read `~/.config/memory-base/user.env` or any other file that holds the user's key.
- Never run `mb_profile.py` with the user's key, and never ask the user to paste the key.
- Never approve on the user's behalf, even when the user tells you to. The user runs the
  command.
- If a proposal is stale (`show` warns, or approval returns `stale`), the user profile
  changed after you wrote the proposal. Read the current user profile at the next session
  start (or ask the user), write the whole replacement again against that version, and
  propose again. Changing only `base_version` is not enough.

## Install (once per machine, by the user)

1. Copy the CLI:

   ```
   mkdir -p ~/.config/memory-base
   cp integrations/profile_approval/mb_profile.py ~/.config/memory-base/mb_profile.py
   ```

2. Write the user's env file. It holds the key that carries the `user` author. It never
   holds the agents' key.

   ```
   (umask 177; printf 'MEMORY_BASE_URL=http://127.0.0.1:8010\n' > ~/.config/memory-base/user.env
    uv run python -m memory_base.serve.keys new <user label> --author user \
      | tail -n 1 | sed 's/^/MEMORY_BASE_USER_KEY=/' >> ~/.config/memory-base/user.env)
   ```

   The `keys` command runs on the server host, in the repo. It writes the key into the
   file without showing it.

   `MEMORY_BASE_USER_ENV` can name another file. A non-empty `MEMORY_BASE_URL` or
   `MEMORY_BASE_USER_KEY` in the environment overrides the file.

3. Install this skill: copy this directory to `~/.claude/skills/memory-profile-approval/`
   for Claude Code, and to the `skills/` directory of the Hermes profile.
