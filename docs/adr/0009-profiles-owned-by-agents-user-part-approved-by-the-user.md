# ADR-0009: Profiles are owned by agents; the user part changes only by the user's approval

## Status

Accepted

## Context

A profile is standing text that a client delivers at session start, because rules and facts
that apply to every task match the topic of almost no message. A profile generated per
namespace from notes, and delivered to every agent, mixes three things that belong to
different readers: an agent's persona and working rules, facts about the user, and work
logs. Each agent needs a different view of the user: a coding agent needs the user's
review and approval habits, a companion agent needs their daily life. The consolidation
agent judges whether notes restate each other; it cannot know what each agent needs to
know about the user. A text about the user also binds every later session of that agent,
so the user must see and accept each change to it.

## Decision

Profiles belong to agents, and the user approves every change to an agent's view of them.

- An owner is an agent's author slug. `user` and `consolidator` are never owners.
- Each owner has two versioned parts. `self` holds the agent's persona, working rules, and
  conventions; the owner replaces it whenever it chooses. `user` holds the user as this
  agent needs to know them; each owner's `user` part is its own document.
- An agent changes its `user` part only by proposing a full replacement written against
  the current user version. A new proposal supersedes the owner's pending one. Only a key
  carrying the `user` author approves or rejects; an approval whose base is not the
  current user version is refused as stale and stays pending.
- Authority comes only from the key's authors: the owner's author to write or propose, the
  owner's or `user` to read, `user` to decide. Admin status, label, home namespace, and
  namespace permissions grant no profile access. Only the user's key carries `user`.
- Every version and every proposal is kept with its decision. Profiles have no namespace,
  are never embedded, and are never read by search or consolidation.
- Writes to one owner serialize on an advisory lock per owner; status, base, and the
  latest version are read only under it.
- The consolidation agent does not write profiles. Each agent knows what it needs about
  the user and proposes it; the user accepts or refuses it.
- The user decides with a standard-library CLI and their own key; a skill tells agents to
  hand each pending proposal to the user and never to run the decision themselves. The
  Claude Code SessionStart hook and the Hermes provider deliver their configured owner's
  two parts with their versions and a notice while a proposal is pending.
- The server calls no chat model for profiles.

## Consequences

- Each agent receives a profile written for it: its own rules, and a view of the user that
  the user accepted.
- The user reviews every change to how an agent knows them, at the cost of one command per
  proposal. A proposal left pending changes nothing.
- A proposal written against an older user version cannot overwrite a newer one; the agent
  must reread the current version and write the replacement again.
- The separation prevents mistakes, not a determined agent on the same host. The agents'
  key stays an admin key, because it needs the private `personal` namespace and the
  owner-or-admin cleanup routes, and an admin key can rewrite any label's authors through
  `PUT /keys/{label}/authors`, including adding `user` to its own label.
- Read isolation between agents is client-side. A key whose authors include several
  owners can read each of them; the hook and the Hermes provider request only their
  configured owner.
