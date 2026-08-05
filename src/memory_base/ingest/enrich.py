"""Generic JSON-mode enrichment for stored content."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
from collections.abc import Callable
from typing import Any

from loguru import logger

from memory_base.core.config import SERVICE_TIMEOUT_SECONDS, llm_client, llm_model

_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9 -]{1,40}$")


class EnrichmentError(RuntimeError):
    """Raised when both enrichment attempts fail validation."""


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
            async with semaphore or contextlib.nullcontext():
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
            logger.opt(exception=True).debug(
                "enrichment attempt {} failed: {}", attempt + 1, last_error
            )
    raise EnrichmentError(f"enrichment failed after retry: {last_error}")


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
