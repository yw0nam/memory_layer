# ADR-0011: Entities and facts beside notes, with the user's consent inside the agent loop

## Status

Accepted for the entity registry and the consent step; the Facts, Reading, Existing notes, and Surface sections are replaced by ADR-0013

Access statements superseded by ADR-0014.

## Context

A note is free text. The thing a note is about, and the values it holds, exist only
inside that text. The selective pilot's stored notes show where this fails. A composite
note, such as a proposal with six stakeholders or a financial setup with several rates,
keeps an old value after the user changes it. The update turn does not resemble the
bundle, so a cosine search misses it. An update that does resemble the bundle is refused
by the near-duplicate check at 0.85. Confirmed stale notes were 3 to 6% of a persona's
active notes after a week, and the dominant junk after a quarter. Storing the source
turns verbatim did not raise read recall (0.578 against 0.578). The lever is not more
text but a key: the thing and the property an update names.

Nothing links notes to the people, projects, and things they describe, and nothing links
a profile to the notes behind it. A spike over 415 notes measured three things. Writers
decompose a note into 2 to 5 (entity, attribute, value) rows. 15 of 17 confirmed stale
values are one such row, which a writer handling the update would look up. Two writers
choose the same entity slug 89% of the time when they see the existing names, and 25% of
the time when they do not; for slug and attribute together the figures are 83% and 15%.
The two stale values no row could address were lists stored as one value ("Embedded
Software Team and Head of Engineering").

The user wants to approve every new entity, in the conversation where the agent is
working, without a separate key or command. The user part of a profile changes the same
way (ADR-0009 required a user-held key and a CLI, which the user found too heavy).

## Decision

A semantic layer sits beside the notes: a registry of entities the user has consented to,
and facts keyed by entity and attribute, each with a validity window and an optional
evidence quote. Notes stay as they are.

### Entities

- `entities` holds `slug` (the key, `^[a-z0-9][a-z0-9-]{0,39}$`), `kind` (`person`,
  `agent`, `project`, `org`, `place`, `thing`), `name`, `aliases`, `description`,
  `created_by`, `consent_method`, `consented_at`, `archived_at`. A proposal or a meeting
  is an entity of kind `thing` when it has properties of its own, such as stakeholders,
  dates, or decisions. A fact belongs to the entity the user would name when asking about
  it later.
- The user is an entity of kind `person`; a project entity carries its repository origin
  among its aliases.
- The server normalizes names and aliases (NFKC, case fold, whitespace) and refuses a slug
  or alias that already resolves to another entity. It infers no equivalence between names
  and merges nothing.
- Creating, editing, and archiving an entity needs the user's consent, given inside the
  conversation. The new entities of one save are put to the user together, in one request.

### Facts

- `facts` holds `id` (a uuid), `namespace`, `entity`, `attribute` (a free slug),
  `value_text` (at most 300 characters), `value_entity` (a slug when the value is itself
  an entity; this is how a relation between two entities is stored), `valid_from`,
  `valid_to`, `note_id`, `evidence`, `author`, `created_at`, `archived_at`. A fact is
  current while `valid_to` is null. Facts are never embedded.
- A write is one of three operations. `add` inserts a current row. `replace(fact_id)`
  closes that row at the new row's `valid_from` (else now) and inserts the new value.
  `remove(fact_id)` closes the row. A removed row stays readable as history and can be
  restored; permanent deletion goes through the admin path, as for notes.
- `add` on an entity and attribute that already has a current row is refused with that
  row's id; the writer answers with `replace` or, when the attribute really holds several
  values, `allow_multiple`. A multi-valued attribute holds one row per element: a value
  shaped as a conjunction of short names ("X, Y and Z") is refused.
- A fact stands alone; `note_id` is set when the fact was distilled with a note. A note is
  written when there is a reason or a context worth keeping, not for every value.
- Attribute names are free. The server shows the writer the entities it may use and the
  attribute names already in use, and refuses nothing about the name.
- `evidence` is the user's own words the fact was taken from, at most 500 characters.
  The writer supplies it and the server does not verify it. It is scanned for credentials,
  never embedded, and returned only on request. A note carries no evidence. This narrows
  ADR-0005: no conversation turn is stored, but a bounded quote attached to a fact is.

### Reading

- `list_facts(entity, attribute?, include_past?)` reads facts deterministically; relations
  are read from either side through `value_entity`, one hop, by a join.
- `search_memory` matches the query against entity names and aliases and returns the
  current facts of the matched entities beside the note hits, up to three entities and
  twenty facts each.
- At session start the client receives up to thirty current facts: those of the user
  entity and those of the entity whose alias is the repository's origin. They are
  delivered as knowledge beside the profile, which stays the agent's policy text.

### Consent inside the loop

One consent step guards entity changes and the user part of a profile. The path is chosen
by the MCP client:

- A client that advertises elicitation gets a dialog inside the tool call; accept applies
  the change in the same call, decline or cancel applies nothing.
- A client without elicitation gets `needs_consent` with a summary, and nothing is
  written. The agent asks the user in the conversation and calls again with `confirmed`
  and the same payload. The server validates the payload again; a profile's base version
  must still be current. A client that advertises elicitation cannot use `confirmed`.
- The server cannot tell an unattended session from an interactive one. The server
  instructions tell an agent in an unattended session (a cron turn, no user present) never
  to pass `confirmed`; it leaves a message on the messages lane, and the next interactive
  session asks the user.
- The row records `consent_method` (`elicitation`, `conversation`, or `import`) and
  `consented_at`. The method is what the client reported, not proof that a person
  answered, and the tool descriptions say so.
- There is no pending-request table. `propose_user_profile` becomes
  `update_user_profile`; `profile_proposals`, its routes, the approval CLI, and the
  profile-approval skill are deleted. Profile versions stay.

### Existing notes

A writer decomposes the active notes once into a file of entities and facts. The user
edits the file and imports it with their own key through an admin route. The import is
recorded with `consent_method = import`. After that, facts accumulate from saves.

### Surface

Four MCP tools: `list_entities`, `register_entity` (create or edit, with consent),
`save_fact` (`add`, `replace`, `remove`), `list_facts`. Every tool reaches the store
through the REST API. The server calls no chat model for entities, facts, or consent.

## Consequences

- An update names its key. A stale value is one row to replace or remove, and the other
  facts of the same entity are untouched; the near-duplicate threshold no longer decides
  whether an update is stored.
- A writer sees the existing names on every save; that is what makes two writers agree.
  A writer that ignores them creates a second slug. The user merges it by hand, because
  the server infers no equivalence.
- Every new entity costs the user one answer; the first import of the existing notes is
  one review of a file instead of a hundred answers.
- Conversational consent is the agent's word. Only elicitation puts the question in front
  of the user by the client's own means; the recorded method tells the two apart.
- A note that is only narrative (a lesson, a procedure) yields no fact and stays a note;
  the fact layer indexes values, it does not replace prose.
- Memory now has three layers: episodic notes, semantic facts about entities, and the
  profile as policy; consolidation can later turn facts into profile text, which it could
  not do from prose.
- A fact about a meeting's subject and a fact about the meeting are both allowed. The rule
  "the entity the user would name later" is guidance a writer applies. The spike shows
  this choice is the residual source of naming disagreement.
