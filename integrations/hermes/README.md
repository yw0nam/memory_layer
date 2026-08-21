# Hermes memory provider

`memory_base/` is a Hermes external memory provider. It pre-injects
memory-base content into every conversation instead of relying on on-demand MCP recall:

- **Session start** — fetches the 5 most recent `kind=episode` notes and renders them as a
  static "recent events" digest, included once in the system prompt for the session.
- **Every turn** — runs a semantic search over memory (all kinds) with the profile's
  configured `top_k`/`min_score`, drops any hit already shown in the digest, and returns
  the rest as prefetched context. Returns nothing on any error, timeout, or when there is
  nothing new to add.

The provider never registers tools — the MCP server already exposes `search`/`search_memory`/
`save_memory` for on-demand recall.

## Layout

- `memory_base/client.py` — pure REST client (stdlib + httpx only, no Hermes imports).
  Talks to the memory-base API's `/notes` and `/search` routes. Unit-tested from this repo
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
| `api_key_env` | `MEMORY_BASE_API_KEY`     | env var holding the memory-base API key     |

The API key itself is read from the environment variable named by `api_key_env`, never from
`config.yaml`.

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
