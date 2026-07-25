FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ src/
RUN uv sync --frozen --no-dev
ENV MCP_TRANSPORT=sse
EXPOSE 8010 8765
CMD ["uv", "run", "--no-sync", "python", "-m", "memory_base.serve.mcp_server"]
