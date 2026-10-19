# ADR-0014: One kind of key; the server grants nothing beyond authentication

## Status

Accepted; supersedes the access statements of ADR-0002 ("the namespace is the access
boundary"), ADR-0011 (an import made with the user's own key), ADR-0012 (a private
namespace keeps a conversation from other keys), and ADR-0013 (facts written only by an
admin key carrying the `consolidator` author; facts and notes scoped to the namespaces the
key may read). The sentence "Nothing in the store is hidden from another agent" is
superseded by [ADR-0015](0015-author-declared-by-the-client-header-and-default-read-scope.md)

## Context

memory-base serves one person. Several agents reach it, from several hosts: a coding
agent, a personal agent, and a scheduled consolidation agent. The server grew an access
model piece by piece without a decision of its own: a key carried a home namespace, an
admin flag, and an allowlist of authors; a namespace was public or private to an owner;
documents and repositories had a creator; facts and consolidation needed the
`consolidator` author; an entity import needed the `user` author; some routes were
admin-only.

The deployed data showed what that model protected. Every active key was admin, so
namespace permissions and owner rules granted nothing. What remained was friction: an
agent had to borrow another agent's key to dry-run a fact write, connecting an agent
needed an author provisioned on its key, and every new feature had to answer "who may
call this" before it could be designed. The one boundary that matters is the network:
the REST and MCP ports are reachable from other hosts.

## Decision

- A key is a credential for the network boundary and nothing else. A request without a
  valid, unrevoked key is refused with 401; any valid key may call every route. Keys are
  equal; each client holds its own, so one host or agent can be cut off by revoking its
  key. A key keeps a `label`, the name of the client that holds it.
- `author` is attribution the client declares in its configuration. The server checks
  that it is a slug and records it; it grants nothing. It stays required wherever a write
  is attributed.
- A namespace is a label that partitions memories for filtering. A write without a
  namespace lands in `default`; a read without a namespace filter covers every namespace.
  Namespaces are registered, a non-empty namespace cannot be deleted, and nothing about a
  namespace is private.
- The `/admin/*` routes are maintenance routes; any valid key may call them. That only
  the consolidation agent writes facts, and that its scheduled run stays in `dry-run`
  until the user says otherwise, are rules of the consolidation procedure, not of the
  server.
- Changing an entity still needs the user's consent inside the agent loop (ADR-0011,
  ADR-0013). Consent is the user's approval of a change, not a credential; the
  `consent_method` a request reports is recorded as the client states it.
- The server keeps two key identifiers that are not authority: the key that queued a job
  bounds that key's backlog, and the key that sent a message partitions its retries.

## Consequences

- Connecting an agent is giving it a key and an author slug. There is no allowlist to
  provision and no admin flag to choose.
- Any agent can archive, delete, undo, and write facts. The safeguards that remain are
  the operations' own: previews, `confirm` and `dry_run`, consent for entity changes,
  stale and undo checks, and the procedures the agents follow. An unattended agent is
  limited by its instructions, as an admin key already was.
- Nothing in the store is hidden from another agent. What a client delivers at session
  start and what it searches is the client's choice of namespace filter.
- A new feature needs no access design: it authenticates and records the author.
- The access code, the key attributes, the namespace owner and visibility, and the
  creator records are removed from the server, the schema, and the tests.
