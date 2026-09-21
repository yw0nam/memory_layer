# Hermes memory provider

`memory_base/` is a Hermes external memory provider. Nothing is injected by recency;
every injected note matched the turn's query at `min_score`.

- **Every turn** — runs a semantic search over memory (all kinds) with the profile's
  configured `top_k`/`min_score` and returns the hits as prefetched context. Returns
  nothing on any error, timeout, or when there is nothing to add.
- **The query is what the turn says.** A `<client_context>` block, which the client
  labels as not typed by the user, is dropped whole before searching — its fields are
  the client's own and change without notice, and their wording matches notes about
  that machinery instead of the turn's subject. A turn holding nothing else skips the
  search.
- **A desire tick is not searched.** A turn carrying `MONITOR CHANGE DETECTED` or
  `DESIRE_STATE_DIR` names the prompt file that already tells it how to act, so a
  search over it returns a copy of that file at best; the turn is dropped whole.

The provider never registers tools — the MCP server already exposes `search`/`search_memory`/
`save_memory` for on-demand recall.

## Layout

- `memory_base/client.py` — pure REST client (stdlib + httpx only, no Hermes imports).
  Talks to the memory-base API's `/search` route. Unit-tested from this repo
  under `tests/integrations/`.
- `memory_base/__init__.py` — the Hermes-facing `MemoryProvider` subclass and `register(ctx)`
  entry point. Imports Hermes types at load time, so it only runs inside a Hermes process.
- `memory_base/plugin.yaml` — plugin metadata (name, version, description, required env var).

## Configuration

Read from `memory.memory_base` in the Hermes profile's `config.yaml`:

| Key           | Default                  | Meaning                                    |
|---------------|---------------------------|---------------------------------------------|
| `url`         | *(required)*              | memory-base REST API base URL              |
| `timeout`     | `5`                        | request timeout, in seconds                 |
| `top_k`       | `5`                        | max prefetch search results                 |
| `min_score`   | `0.6`                      | relevance floor for prefetch search          |
| `api_key`     | *(none)*                   | API key value; takes precedence over `api_key_env` |
| `api_key_env` | `MEMORY_BASE_API_KEY`     | env var holding the memory-base API key     |

The API key is `api_key` when set, otherwise the environment variable named by
`api_key_env`.

## Deployment

Symlink the plugin directory into the Hermes profile's user-plugin directory, then select it
as the active provider:

```sh
ln -s "$(pwd)/integrations/hermes/memory_base" "$HERMES_HOME/plugins/memory_base"
```

```yaml
# $HERMES_HOME/config.yaml
memory:
  provider: memory_base
  memory_base:
    url: "https://memory-base.example.com"
```
