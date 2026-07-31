"""Generic JSON-mode enrichment for stored content."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from collections.abc import Callable
from typing import Any

from memory_base.core.config import SERVICE_TIMEOUT_SECONDS, llm_client, llm_model

_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9 -]{1,40}$")
_PRONOUN_RE = re.compile(
    r"\b(it|its|he|him|his|she|her|hers|they|them|their|the company|the person)\b",
    re.IGNORECASE,
)


class EnrichmentError(RuntimeError):
    """Raised when both enrichment attempts fail validation."""


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def normalize_enrichment_tags(value: Any) -> list[str]:
    """Drop malformed tags and return normalized unique phrases."""
    if not isinstance(value, list):
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        tag = item.strip().lower()
        if _TAG_RE.fullmatch(tag) and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags[:7]


def parse_atom_response(payload: Any, *, atoms_generate: bool = True) -> dict[str, Any] | None:
    """Validate and normalize atom enrichment JSON."""
    if not isinstance(payload, dict):
        return None
    questions = payload.get("atom_questions")
    raw_tags = payload.get("tags")
    if not isinstance(questions, list) or not isinstance(raw_tags, list):
        return None
    tags = normalize_enrichment_tags(raw_tags)
    if not tags:
        return None
    atoms: list[str] = []
    seen: set[str] = set()
    if atoms_generate:
        for item in questions:
            if not isinstance(item, str):
                continue
            question = item.strip()
            key = question.casefold()
            if (
                question
                and len(question) <= 200
                and not _PRONOUN_RE.search(question)
                and key not in seen
            ):
                seen.add(key)
                atoms.append(question)
            if len(atoms) == 10:
                break
    return {"atom_questions": atoms, "tags": tags}


def parse_summary_response(payload: Any) -> dict[str, Any] | None:
    """Validate and normalize summary enrichment JSON."""
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    raw_tags = payload.get("tags")
    if not isinstance(summary, str) or not summary.strip() or not isinstance(raw_tags, list):
        return None
    tags = normalize_enrichment_tags(raw_tags)
    if not tags:
        return None
    return {"summary": summary.strip(), "tags": tags}


async def _notify_retry(callback: Callable[[], Any] | None) -> None:
    if callback is None:
        return
    result = callback()
    if inspect.isawaitable(result):
        await result


async def _json_call(
    messages: list[dict[str, str]],
    parser: Callable[[Any], dict[str, Any] | None],
    *,
    semaphore: asyncio.Semaphore | None,
    on_retry: Callable[[], Any] | None,
) -> dict[str, Any]:
    last_error = "invalid response"
    for attempt in range(2):
        if attempt:
            await _notify_retry(on_retry)
        try:
            async with semaphore or _NullAsyncContext():
                response = await asyncio.wait_for(
                    llm_client().chat.completions.create(
                        model=llm_model(),
                        messages=messages,
                        response_format={"type": "json_object"},
                    ),
                    timeout=SERVICE_TIMEOUT_SECONDS,
                )
            content = response.choices[0].message.content
            payload = json.loads(content)
            parsed = parser(payload)
            if parsed is not None:
                return parsed
            last_error = "response failed validation"
        except Exception as exc:
            last_error = str(exc) or type(exc).__name__
    raise EnrichmentError(f"enrichment failed after retry: {last_error}")


class _NullAsyncContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        return None


async def atomize_and_tag(
    text: str,
    context: str,
    *,
    semaphore: asyncio.Semaphore | None = None,
    on_retry: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Generate entity-explicit atom questions and English tags."""
    atoms_generate = _env_enabled("ATOMS_GENERATE")
    atom_instruction = (
        "Extract diverse questions answerable by the content. Every question must contain "
        "necessary entity names and must not use pronouns such as it, he, she, they, the "
        "company, or the person. Return at most 10 questions."
        if atoms_generate
        else "Return an empty atom_questions list."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You understand source content and return only valid JSON. All generated "
                "questions and tags are English."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{atom_instruction}\nGenerate 3 to 7 concise topical tag phrases.\n"
                'Return {"atom_questions": [...], "tags": [...]}.\n\n'
                f"Context:\n{context}\n\nContent:\n{text}"
            ),
        },
    ]
    return await _json_call(
        messages,
        lambda payload: parse_atom_response(payload, atoms_generate=atoms_generate),
        semaphore=semaphore,
        on_retry=on_retry,
    )


async def summarize_and_tag(
    text: str,
    context: str,
    *,
    semaphore: asyncio.Semaphore | None = None,
    on_retry: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Generate one English summary and English tags."""
    messages = [
        {
            "role": "system",
            "content": (
                "You summarize source content faithfully and return only valid JSON. "
                "All generated content is English."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write a non-empty English knowledge summary and 3 to 7 concise topical tag "
                'phrases. Return {"summary": "...", "tags": [...]}.\n\n'
                f"Context:\n{context}\n\nContent:\n{text}"
            ),
        },
    ]
    return await _json_call(
        messages,
        parse_summary_response,
        semaphore=semaphore,
        on_retry=on_retry,
    )
