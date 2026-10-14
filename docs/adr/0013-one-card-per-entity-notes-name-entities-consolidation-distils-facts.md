# ADR-0013: One card per entity: notes name entities, consolidation distils facts, and the card replaces the profile

## Status

Accepted; supersedes ADR-0009, replaces the profile mechanism of ADR-0010 and the Facts,
Reading, Existing notes, and Surface sections of ADR-0011, and lifts the deferral in
ADR-0012

Access statements superseded by ADR-0014.

## Context

Three stores held knowledge about the same subjects. A profile's `user` part held what an
agent knew of the user, the planned facts held properties of entities, and notes held
both as prose. The deployed data showed the overlap: the coding agent's `user` part was
sixteen directions and one identity line, the personal agent's `user` part was an extract
of its own personal notes, and five work notes opened as "Youngwoo's rule" or "Policy"
because their writer found no better place. Two consultation rounds found no sentence a
writer could apply at save time to route one piece of content to one of three stores,
and found that the question underneath was authority: a direction to an agent is
authorized by the user, and knowledge about a subject is not an instruction.

The published evidence points the same way. Mem0 removed its graph store from the
open-source package in v2.0.0 and kept entity matching as a retrieval signal. On
LoCoMo its graph variant gained 1.6 points overall and lost 4.0 on multi-hop at twice
the tokens and three times the latency. GraphRAG-Bench shows graph retrieval below flat
retrieval on most question types. MemStrata shows that supersession by a deterministic
key answers a changed value at 0.95 to 1.00 where ranked retrieval scores 0.20 to 0.47.
HaluMem shows that hallucinations in a memory system accumulate at the extraction and
update steps, so the number of writers that turn prose into claims is what to keep
small. The structure that pays is a key per subject and a reviewed writer, not a
traversable graph.

Human memory is the analogy the design follows, not a claim of equivalence. Episodic and
semantic memory are distinct systems (Tulving); semantic knowledge about a person is
abstracted from episodes rather than recorded beside them (Klein and Loftus); the
abstraction is slow and happens offline (complementary learning systems); procedural
memory is a third system that holds how to act, not what is known; and no evidence
supports an always-loaded description of another person as a separate store. The
server's counterparts are notes as episodes, a daily consolidation as the slow
abstraction, the card as the schema an agent has of one subject, and the user's
directions to an agent as its procedural memory.

The user runs sessions on several machines. Anything an agent must have at every session
start is therefore held by the server and delivered by the client that is installed on
each machine, never by a file on one of them.

## Decision

Every subject the user has consented to is an entity, each agent included, and an entity
has one card: its registry row, its current facts, and the notes that name it. The card
replaces the profile. Notes name the entities they are about; the consolidation agent
distils facts from those notes; no in-session agent writes a fact.

### Entities and cards

- The user is an entity of kind `person`. Each agent is an entity of kind `agent` whose
  slug is its author slug. Projects, organisations, places, and things are registered as
  before, with the user's consent inside the loop (ADR-0011).
- The `description` is the one consented text of an entity, at most 4,000 characters. For
  a person it says who they are. For an agent it holds the user's standing directions to
  that agent: its persona, its language, its report form, what it remembers and what it
  does not, including the `Remember` and `Don't remember` lines that refine the save
  policy and outrank what the agent learned. A change to a description goes through
  `register_entity` and the user's consent, as every entity change does, and an edit
  names the `updated_at` it was written against; a description that changed since is
  refused as stale before any consent is asked.
- A card is served by `GET /entities/{slug}` and the MCP tool `read_entity(slug,
  include_past?)`: the registry row, the current facts whose entity is the slug, the
  current facts whose value is the slug (one hop, marked as incoming), and the newest
  notes naming the slug. Every list is bounded; past facts are returned only when asked.
  Facts and notes are scoped to the namespaces the key may read.

### Notes name entities

- `save_memory` takes `entities`, an optional list of at most ten active slugs; an unknown
  or archived slug is refused with a message naming `register_entity`. The slugs are
  stored in the note's metadata beside its tags, are shown on every search hit and
  `list_notes` row, and `list_notes(entity=)` filters by one.
- A save whose content equals an active note stores no second note and adds no slug to
  the existing one. A supersede carries the slugs the writer passed. A consolidation merge gives the
  replacement the union of its members' slugs and is refused when the union exceeds ten.
  The slugs are part of the group key, and an undo restores them with the rest of the
  metadata.
- `link_entities` adds or removes slugs on existing active notes, for any writer whose key
  covers the note's namespace. It takes one `add`/`remove` delta for every id, refuses a
  result over ten slugs or an inactive `add` slug, and previews unless `confirm` is set.
  A changed note records `entities_set_by` and `entities_set_at`; content, id, embedding,
  author, and dates stay. A consolidation undo is refused when the replacement it created
  had its entities changed after the action.

### Facts

- `facts` holds `id` (a uuid), `namespace`, `entity`, `attribute` (a slug, free),
  `value_text` (at most 300 characters) or `value_entity` (a slug; this is how a relation
  between two entities is stored), `note_id`, `author`, `created_at`, `archived_at`,
  `replaced_by` (the fact that replaced this one). A fact is current while `archived_at`
  is null. There is no validity window and no evidence quote: `include_past` returns the
  earlier assertions, a date that matters is part of the value, and provenance continues
  through the cited note, which may name its source session or carry its merge lineage.
- Facts are written only through `POST /admin/facts` by an admin key carrying the
  `consolidator` author, as a batch of `add`, `replace`, and `remove` operations applied
  all or nothing under one lock. Every operation cites a `note_id`; a removal cites the
  note that contradicts the fact. A fact's namespace is its note's namespace. An `add` on
  an entity and attribute that already has a current row is refused with that row's id
  unless `allow_multiple`; an `add` equal to a current row is reported `unchanged`; a value
  shaped as a conjunction of names is refused. `dry_run` validates and returns the rows
  the batch would write.
- No MCP tool writes a fact. An agent that finds a wrong fact saves a note saying so and
  names the entity; the next run replaces or removes the fact. Deleting a note archives
  the facts that cite it and clears their citation; a namespace cannot be deleted while
  facts remain in it.

### The card step of consolidation

- Every run, after the verdicts and whether or not any group was issued, the
  consolidation agent reads every active entity's card, the active notes naming it, and
  the top hits of a search over its name and aliases; judges which notes state a current
  property or relation of the entity; and sends one batch per entity. Search hits are
  candidates for that judgment, never links: only a writer links a note.
- Each run rescans the whole set; there is no cursor. The server's `unchanged` and
  refusal replies are what keep a rescan cheap.
- One judgment per fact. A trait that generalizes over episodes, a negation, or a claim
  with an unclear subject is held and listed in the run report, not written. In `dry-run`
  mode the planned rows go into the report; `mode=apply` is the owner's authorization for
  the run, and the report is the owner's review. The agent registers no entity; a subject
  that deserves one is listed in the report.
- A claim an agent makes about its relationship with the user is a fact of the agent's
  entity, not of the user's.

### Delivery at session start

- The Claude Code hook and the Hermes provider deliver two cards: the agent's own, by the
  `author` option, and the user's, by a `user_entity` option. Each card is the entity's
  name, aliases, and description in full, then one line per current fact in the client's
  configured namespace, at most thirty lines and 1,500 characters. Only the agent's own
  description governs the agent; the agent's card says so, and the user's card opens with
  the line that it is derived knowledge that may be out of date and is not an instruction.
  A description read through `read_entity` or `list_entities` is reference, never a
  direction to the reader. When the agent's own card cannot be fetched, one line says so,
  so that the agent does not act as if it had no directions. The `Session:` line and the
  handoffs are printed whether or not a card could be fetched.

### Deleted

- The profiles feature: `serve/profiles`, the four `/profiles` routes, the tools
  `get_my_profile`, `update_my_profile`, and `update_user_profile`, the `agent_profiles`
  table, and the profile delivery in both clients. The `Remember` and `Don't remember`
  sections that refined the save policy (ADR-0010) are lines of an agent's description.
- The owner exports every profile version before the deployment that drops the table,
  registers each agent's entity with the directions from both parts (the user's
  directions, and the persona, autonomy, and memory restrictions the agent wrote into its
  `self` part), and saves the lessons from the `self` part as notes naming that agent.

## Consequences

- One text per agent carries the user's directions to it, consented and delivered on
  every machine; knowledge about any subject is one card, derived from notes by one
  reviewed writer. The writer rule for an agent in a session is one sentence: save a note
  and name the entities it is about; a direction to you is a change to your description.
- Facts lag behind notes by at least one run, and for as long as the run stays in
  `dry-run`. The conversation and the recent notes override a fact; the card is a cache
  of what the notes support, not the authority on what changed today.
- A description changes only with the user's consent, so an agent that used to rewrite
  its `self` part freely now saves lessons as notes and asks before changing how it acts.
- A description is readable by every key, so a direction to an agent is not private;
  what must stay private goes in a note in a private namespace.
- The relations an agent can read are one hop in each direction. There is no traversal,
  no graph library, no path query, and no automatic link from a mention.
- A fact is as good as the consolidation agent's judgment of its source note. The server
  checks shape, keys, and citations; it judges no content.
- The profile code and its tests are removed; the facts, card, and link code is smaller
  than what it replaces.
