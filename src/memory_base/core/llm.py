"""Chat-provider selection from .env API keys and the single chat entry point."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import anthropic
from loguru import logger
from openai import AsyncOpenAI


class _KeyedSpec(NamedTuple):
    key_var: str
    name: Literal["opencode", "zai", "openai", "claude"]
    model_var: str
    default_model: str
    base_url_var: str | None
    default_base_url: str | None


_ZAI = _KeyedSpec(
    "ZAI_API_KEY",
    "zai",
    "ZAI_MODEL",
    "glm-5.3-flash",
    "ZAI_BASE_URL",
    "https://api.z.ai/api/coding/paas/v4",
)

# First non-empty key wins; no base URL means the SDK default.
_KEYED_PROVIDERS = (
    _KeyedSpec(
        "OPENCODE_API_KEY",
        "opencode",
        "OPENCODE_MODEL",
        "deepseek-v4.1-flash",
        None,
        "https://opencode.ai/inference/go/openai/v1",
    ),
    _ZAI,
    _KeyedSpec("OPENAI_API_KEY", "openai", "OPENAI_MODEL", "gpt-5.6-luna", None, None),
    _KeyedSpec("CLAUDE_API_KEY", "claude", "CLAUDE_MODEL", "claude-haiku-4-5", None, None),
)


@dataclass(frozen=True)
class LlmProvider:
    """The chat backend selected from .env keys; vLLM is the no-key case."""

    name: Literal["opencode", "zai", "openai", "claude", "vllm"]
    model: str
    base_url: str | None
    api_key: str | None


def _keyed_provider(env: Mapping[str, str], spec: _KeyedSpec) -> LlmProvider:
    return LlmProvider(
        name=spec.name,
        model=env.get(spec.model_var) or spec.default_model,
        base_url=(spec.base_url_var and env.get(spec.base_url_var)) or spec.default_base_url,
        api_key=env[spec.key_var],
    )


def zai_provider(env: Mapping[str, str]) -> LlmProvider:
    """Build the z.ai provider from its own keys, whatever other keys are set."""
    if not env.get(_ZAI.key_var):
        raise RuntimeError("ZAI_API_KEY is not set; define it in .env")
    return _keyed_provider(env, _ZAI)


def resolve_llm_provider(env: Mapping[str, str]) -> LlmProvider:
    """Pick the provider from the first non-empty API key, else the vLLM endpoint."""
    for spec in _KEYED_PROVIDERS:
        if env.get(spec.key_var):
            return _keyed_provider(env, spec)
    missing = [name for name in ("VLLM_URL", "VLLM_MODEL") if not env.get(name)]
    if missing:
        raise RuntimeError(
            f"no chat API key is set and {', '.join(missing)} is missing; define them in .env"
        )
    return LlmProvider(name="vllm", model=env["VLLM_MODEL"], base_url=env["VLLM_URL"], api_key=None)


_selected_logged = False


def _openai_client(provider: LlmProvider) -> AsyncOpenAI:
    headers = None
    if provider.name == "opencode":
        headers = {
            "User-Agent": f"memory-base/{importlib.metadata.version('memory-base')}",
            "x-opencode-session": uuid.uuid4().hex,
        }
    return AsyncOpenAI(
        base_url=provider.base_url, api_key=provider.api_key or "EMPTY", default_headers=headers
    )


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
