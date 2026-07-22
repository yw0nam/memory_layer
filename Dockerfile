FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src/ src/
ENV MCP_TRANSPORT=sse
EXPOSE 8765
CMD ["uv", "run", "--no-sync", "python", "src/mcp_server.py"]
