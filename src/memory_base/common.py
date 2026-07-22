"""Shared config + vLLM (OpenAI-compatible) clients for LLM / embedding / rerank."""

from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv
from numpy.typing import NDArray
from openai import AsyncOpenAI

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "GPT-OSS-120B")
EMB_MODEL = os.getenv("EMB_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
EMB_DIM = 2048  # Qwen3-VL-Embedding-2B, no matryoshka -> halfvec(2048) in pgvector

DB_URL = os.getenv("DB_URL", "postgres://memory:memory@localhost:5439/memory_base")
PG_SCHEMA = "memory"

# Qwen3 embedding convention: instruction-prefixed query, raw document.
_QUERY_PREFIX = (
    "Instruct: Given a search query, retrieve relevant code or conversation "
    "passages that answer the query\nQuery: "
)


def llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=os.environ["LLM_URL"], api_key="EMPTY")


class VllmEmbedder:
    """Embedder backed by vLLM /v1/embeddings. float16 output -> halfvec column.

    Implements CocoIndex's VectorSchemaProvider protocol so it can annotate
    embedding columns in table schemas.
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(base_url=os.environ["EMB_URL"], api_key="EMPTY")

    async def __coco_vector_schema__(self):  # noqa: ANN201 - cocoindex protocol
        from cocoindex.resources.schema import VectorSchema

        return VectorSchema(dtype=np.dtype(np.float16), size=EMB_DIM)

    async def embed(self, text: str, *, query: bool = False) -> NDArray:
        if query:
            text = _QUERY_PREFIX + text
        r = await self._client.embeddings.create(model=EMB_MODEL, input=text)
        return np.asarray(r.data[0].embedding, dtype=np.float16)
