# Implementation spec: Docker-served MCP server + SSE transport (Issue #10)

## Requirements

Serve the MCP server as a Docker container, with clients connecting over SSE. stdio stays the local-dev default.

## Target files

- Modify: `src/mcp_server.py`, `docker-compose.yml`
- New: `Dockerfile`, `.dockerignore`, `tests/test_mcp_transport.py`
- Do not modify other src files.

## Change 1 — transport selection in `src/mcp_server.py`

- Add a pure function:
  ```python
  def resolve_transport(env: Mapping[str, str]) -> tuple[str, str, int]:
      """(transport, host, port). MCP_TRANSPORT: stdio(default)|sse|streamable-http.
      MCP_HOST default "0.0.0.0", MCP_PORT default 8765. ValueError on an invalid transport value."""
  ```
- `__main__`: branch on the result of `resolve_transport(os.environ)` —
  for stdio, keep `mcp.run(transport="stdio")` as-is;
  otherwise set host/port on the `FastMCP` instance and `mcp.run(transport=<value>)`.
  Note: FastMCP takes host/port via the constructor/settings (`FastMCP("memory-base", host=..., port=...)`).
  Make env parsing happen only in `__main__`, not at module import (for testability).
- Update the registration examples in the file's top docstring to two forms:
  - stdio (local): keep existing
  - SSE (Docker): `claude mcp add --transport sse memory-base http://localhost:8765/sse`

## Change 2 — `Dockerfile`

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src/ src/
ENV MCP_TRANSPORT=sse
EXPOSE 8765
CMD ["uv", "run", "--no-sync", "python", "src/mcp_server.py"]
```
(the exact uv image tag/options may be adjusted after confirming behavior at implementation time. Principle: lock-based reproducible build, dev dependencies excluded)

`.dockerignore`: `.venv`, `.git`, `.cocoindex`, `__pycache__`, `tests`, `docs`, `.env`

## Change 3 — add an `mcp` service to `docker-compose.yml`

```yaml
  mcp:
    build: .
    container_name: memory_base_mcp
    depends_on: [db]
    ports:
      - "8765:8765"
    environment:
      MCP_TRANSPORT: sse
      DB_URL: postgres://memory:memory@db:5432/memory_base   # db:5432 inside the container
      EMB_URL: ${EMB_URL}
      RERANK_URL: ${RERANK_URL}
      LLM_URL: ${LLM_URL}
```
Do not modify the existing `db` service (keep volumes & ports).

## TDD — tests first

`tests/test_mcp_transport.py` (no DB/network dependency; write first, confirm red, then implement):
- default: empty env → ("stdio", "0.0.0.0", 8765)
- MCP_TRANSPORT=sse → ("sse", ...), streamable-http allowed, case-insensitive
- MCP_PORT=9000 → port 9000 (int), MCP_HOST reflected
- invalid transport ("tcp") → ValueError
- non-numeric MCP_PORT → ValueError

## Acceptance criteria

1. `uv run pytest tests/test_mcp_transport.py -q` red→green, all existing tests preserved
2. (orchestrator) `docker compose up -d --build mcp` → the SSE endpoint (`curl http://localhost:8765/sse`) responds with an event stream
3. (orchestrator) an MCP client connects over SSE and successfully calls tools/list + search_code

## Notes

- In the MCP protocol SSE is legacy and streamable-http is the current standard, but **the user explicitly requested SSE** —
  default to SSE while allowing streamable-http via env (a one-line difference).
- The external vLLM (192.168.x.x) is reachable from the container over the bridge network.
