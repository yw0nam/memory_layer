"""Chat-provider selection from .env API keys and the single chat entry point."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
from loguru import logger
from openai import AsyncOpenAI

_DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/coding/paas/v4"

# First non-empty key wins; empty base_url means the SDK default.
_KEYED_PROVIDERS = (
    ("ZAI_API_KEY", "zai", "ZAI_MODEL", "glm-5.3-flash", "ZAI_BASE_URL", _DEFAULT_ZAI_BASE_URL),
    ("OPENAI_API_KEY", "openai", "OPENAI_MODEL", "gpt-5.6-luna", None, None),
    ("CLAUDE_API_KEY", "claude", "CLAUDE_MODEL", "claude-haiku-4-5", None, None),
)


@dataclass(frozen=True)
class LlmProvider:
    """The chat backend selected from .env keys; vLLM is the no-key case."""

    name: Literal["zai", "openai", "claude", "vllm"]
    model: str
    base_url: str | None
    api_key: str | None


def resolve_llm_provider(env: Mapping[str, str]) -> LlmProvider:
    """Pick the provider from the first non-empty API key, else the vLLM endpoint."""
    for key_var, name, model_var, default_model, base_url_var, default_base_url in _KEYED_PROVIDERS:
        if env.get(key_var):
            base_url = None
            if base_url_var:
                base_url = env.get(base_url_var) or default_base_url
            return LlmProvider(
                name=name,
                model=env.get(model_var) or default_model,
                base_url=base_url,
                api_key=env[key_var],
            )
    missing = [name for name in ("VLLM_URL", "VLLM_MODEL") if not env.get(name)]
    if missing:
        raise RuntimeError(
            f"no chat API key is set and {', '.join(missing)} is missing; define them in .env"
        )
    return LlmProvider(name="vllm", model=env["VLLM_MODEL"], base_url=env["VLLM_URL"], api_key=None)


_selected_logged = False


def _openai_client(provider: LlmProvider) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=provider.base_url, api_key=provider.api_key or "EMPTY")


def _anthropic_client(provider: LlmProvider) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=provider.api_key)


def _schema_prompt(schema: dict[str, Any]) -> str:
    return f"Return only JSON matching this schema: {json.dumps(schema)}"


def _with_schema_prompt(
    messages: list[dict[str, str]], schema: dict[str, Any]
) -> list[dict[str, str]]:
    """Append the schema request to the system message, adding one if absent."""
    prepared = [dict(message) for message in messages]
    prompt = _schema_prompt(schema)
    if prepared and prepared[0].get("role") == "system":
        prepared[0]["content"] += "\n\n" + prompt
    else:
        prepared.insert(0, {"role": "system", "content": prompt})
    return prepared


async def _openai_json(
    provider: LlmProvider, messages: list[dict[str, str]], schema: dict[str, Any]
) -> dict[str, Any]:
    # json_object mode does not enforce a shape, so the schema rides in the prompt too.
    client = _openai_client(provider)
    response = await client.chat.completions.create(
        model=provider.model,
        messages=_with_schema_prompt(messages, schema),
        response_format={"type": "json_object"},
        # GLM reasons by default; JSON classification gains nothing from it and waits 3-10x longer.
        extra_body={"thinking": {"type": "disabled"}} if provider.name == "zai" else None,
    )
    return json.loads(response.choices[0].message.content)


async def _anthropic_json(
    provider: LlmProvider, messages: list[dict[str, str]], schema: dict[str, Any]
) -> dict[str, Any]:
    prepared = _with_schema_prompt(messages, schema)
    system = "\n\n".join(m["content"] for m in prepared if m["role"] == "system")
    client = _anthropic_client(provider)
    response = await client.messages.create(
        model=provider.model,
        max_tokens=4096,
        system=system,
        messages=[m for m in prepared if m["role"] != "system"],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    return json.loads(response.content[0].text)


async def chat_json(
    messages: list[dict[str, str]], schema: dict[str, Any], *, timeout: float
) -> dict[str, Any]:
    """Run one chat completion and return the parsed JSON object."""
    global _selected_logged
    provider = resolve_llm_provider(os.environ)
    if not _selected_logged:
        logger.info("chat model: {} / {}", provider.name, provider.model)
        _selected_logged = True
    call = _anthropic_json if provider.name == "claude" else _openai_json
    return await asyncio.wait_for(call(provider, messages, schema), timeout=timeout)
