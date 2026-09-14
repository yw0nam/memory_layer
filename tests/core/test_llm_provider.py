"""Unit tests for chat-provider selection and the shared chat_json entry point."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from memory_base.core import llm

_ENV_VARS = (
    "ZAI_API_KEY",
    "OPENAI_API_KEY",
    "CLAUDE_API_KEY",
    "ZAI_BASE_URL",
    "ZAI_MODEL",
    "OPENAI_MODEL",
    "CLAUDE_MODEL",
    "VLLM_URL",
    "VLLM_MODEL",
)


def _env(**overrides: str) -> dict[str, str]:
    return {name: "" for name in _ENV_VARS} | overrides


def _vllm() -> llm.LlmProvider:
    return llm.LlmProvider(name="vllm", model="m", base_url="http://vllm.test", api_key=None)


@pytest.mark.parametrize(
    "env, expected",
    [
        (_env(ZAI_API_KEY="z", OPENAI_API_KEY="o", CLAUDE_API_KEY="c"), "zai"),
        (_env(OPENAI_API_KEY="o", CLAUDE_API_KEY="c"), "openai"),
        (_env(CLAUDE_API_KEY="c"), "claude"),
        (_env(ZAI_API_KEY="", OPENAI_API_KEY="o"), "openai"),
    ],
)
def test_first_non_empty_key_selects_the_provider(env, expected):
    assert llm.resolve_llm_provider(env).name == expected


@pytest.mark.parametrize(
    "key_var, name, model, base_url",
    [
        ("ZAI_API_KEY", "zai", "glm-5.3-flash", "https://api.z.ai/api/coding/paas/v4"),
        ("OPENAI_API_KEY", "openai", "gpt-5.6-luna", None),
        ("CLAUDE_API_KEY", "claude", "claude-haiku-4-5", None),
    ],
)
def test_each_single_key_gets_its_default_model_and_base_url(key_var, name, model, base_url):
    provider = llm.resolve_llm_provider(_env(**{key_var: "secret"}))
    assert provider == llm.LlmProvider(name=name, model=model, base_url=base_url, api_key="secret")


def test_explicit_model_and_base_url_override_the_defaults():
    provider = llm.resolve_llm_provider(
        _env(ZAI_API_KEY="secret", ZAI_MODEL="glm-custom", ZAI_BASE_URL="http://proxy.test")
    )
    assert provider.model == "glm-custom"
    assert provider.base_url == "http://proxy.test"


def test_no_key_selects_vllm_from_its_own_vars():
    provider = llm.resolve_llm_provider(_env(VLLM_URL="http://vllm.test", VLLM_MODEL="qwen"))
    assert provider == llm.LlmProvider(
        name="vllm", model="qwen", base_url="http://vllm.test", api_key=None
    )


@pytest.mark.parametrize("missing", ["VLLM_URL", "VLLM_MODEL"])
def test_no_key_and_missing_vllm_var_raises_naming_it(missing):
    env = _env(**{name: "x" for name in ("VLLM_URL", "VLLM_MODEL")} | {missing: ""})
    with pytest.raises(RuntimeError, match=missing):
        llm.resolve_llm_provider(env)


class _FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def _patch_openai(monkeypatch, completions) -> None:
    monkeypatch.setattr(
        llm,
        "_openai_client",
        lambda provider: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )


def test_openai_compatible_branch_sends_json_object_mode_and_schema_prompt(monkeypatch):
    completions = _FakeCompletions('{"ok": true}')
    _patch_openai(monkeypatch, completions)
    monkeypatch.setattr(llm, "resolve_llm_provider", lambda env: _vllm())
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    result = asyncio.run(
        llm.chat_json(
            [{"role": "system", "content": "You summarize."}, {"role": "user", "content": "text"}],
            schema,
            timeout=5,
        )
    )

    assert result == {"ok": True}
    assert completions.kwargs["response_format"] == {"type": "json_object"}
    system = completions.kwargs["messages"][0]
    assert system["role"] == "system"
    assert system["content"].startswith("You summarize.")
    assert json.dumps(schema) in system["content"]
    assert completions.kwargs["messages"][1] == {"role": "user", "content": "text"}


def test_openai_compatible_branch_inserts_system_prompt_when_messages_have_none(monkeypatch):
    completions = _FakeCompletions('{"ok": 1}')
    _patch_openai(monkeypatch, completions)
    monkeypatch.setattr(llm, "resolve_llm_provider", lambda env: _vllm())

    asyncio.run(llm.chat_json([{"role": "user", "content": "hi"}], {"type": "object"}, timeout=5))

    assert completions.kwargs["messages"][0]["role"] == "system"


def test_claude_branch_sends_json_schema_output_config_and_parses_the_text_block(monkeypatch):
    captured: dict[str, Any] = {}

    class _FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text='{"ok": true}')])

    monkeypatch.setattr(
        llm, "_anthropic_client", lambda provider: SimpleNamespace(messages=_FakeMessages())
    )
    monkeypatch.setattr(
        llm,
        "resolve_llm_provider",
        lambda env: llm.LlmProvider(name="claude", model="m", base_url=None, api_key="secret"),
    )
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    result = asyncio.run(
        llm.chat_json(
            [{"role": "system", "content": "You summarize."}, {"role": "user", "content": "text"}],
            schema,
            timeout=5,
        )
    )

    assert result == {"ok": True}
    assert captured["output_config"] == {"format": {"type": "json_schema", "schema": schema}}
    assert captured["messages"] == [{"role": "user", "content": "text"}]
    assert "Return only JSON matching this schema" in captured["system"]


def test_chat_json_enforces_the_caller_timeout(monkeypatch):
    class _SlowCompletions:
        async def create(self, **kwargs):
            del kwargs
            await asyncio.sleep(5)

    monkeypatch.setattr(
        llm,
        "_openai_client",
        lambda provider: SimpleNamespace(chat=SimpleNamespace(completions=_SlowCompletions())),
    )
    monkeypatch.setattr(llm, "resolve_llm_provider", lambda env: _vllm())

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(llm.chat_json([{"role": "user", "content": "hi"}], {}, timeout=0.01))


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (
            llm.LlmProvider(name="zai", model="glm-5.3-flash", api_key="k", base_url="https://z"),
            {"thinking": {"type": "disabled"}},
        ),
        (_vllm(), None),
    ],
)
def test_openai_compatible_branch_disables_glm_thinking_only_on_zai(
    monkeypatch, provider, expected
):
    completions = _FakeCompletions('{"ok": true}')
    _patch_openai(monkeypatch, completions)
    monkeypatch.setattr(llm, "resolve_llm_provider", lambda env: provider)

    asyncio.run(llm.chat_json([{"role": "user", "content": "hi"}], {"type": "object"}, timeout=5))

    assert completions.kwargs["extra_body"] == expected
