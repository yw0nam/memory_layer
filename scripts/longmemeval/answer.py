#!/usr/bin/env python3
"""Answer and judge LongMemEval questions with the open-provider model.

`answer` answers each retrieved packet with upstream's facts prompt; `judge` grades each
current answer with upstream's grading prompt for its question type. Both call
glm-5.3-flash on the z.ai provider from .env (temperature 0, thinking disabled, upstream's
token limits) and append one fsynced line per question to answers[-run].jsonl or
judgments[-run].jsonl in the data dir; a rerun skips every question whose current prompt
already has a row.

Usage:
  uv run python scripts/longmemeval/answer.py answer --dataset PATH [--gate off | --variant dated]
  uv run python scripts/longmemeval/answer.py judge --dataset PATH [--gate off | --variant dated]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from memory_base.core import llm
from memory_base.eval import longmemeval as lme

DEFAULT_MODEL = "glm-5.3-flash"
DEFAULT_CONCURRENCY = 5
# Answers keep upstream's 500; the judge gets room for the reasoning glm emits despite thinking off.
MAX_TOKENS = {"answer": 500, "judge": 200}
ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5.0
CALL_TIMEOUT_SECONDS = 120.0


@dataclass
class ChatModel:
    """The answer/judge model on the z.ai OpenAI-compatible endpoint chosen from the env."""

    client: AsyncOpenAI
    model: str
    provider: str = "zai"

    @classmethod
    def from_env(cls, env: Mapping[str, str], *, model: str) -> ChatModel:
        provider = llm.resolve_llm_provider(env)
        if provider.name != "zai":
            raise RuntimeError(f"answering needs the zai provider, not {provider.name}")
        return cls(AsyncOpenAI(base_url=provider.base_url, api_key=provider.api_key), model)

    async def complete(self, prompt: str, *, max_tokens: int) -> tuple[str, int, int]:
        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=0,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            ),
            timeout=CALL_TIMEOUT_SECONDS,
        )
        usage = response.usage
        text = (response.choices[0].message.content or "").strip()
        return text, usage.prompt_tokens, usage.completion_tokens


def pending_prompts(
    stage: str, data_dir: Path, run: str, questions: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Current prompts of the stage that have no row for that exact prompt yet."""
    expected = lme.stage_prompts(stage, data_dir, run, questions)
    done = lme.current_rows(lme.read_jsonl(lme.stage_output_path(data_dir, stage, run)), expected)
    return {qid: prompt for qid, prompt in expected.items() if qid not in done}


async def _complete_with_retry(client: Any, prompt: str, max_tokens: int):
    for attempt in range(ATTEMPTS):
        try:
            reply = await client.complete(prompt, max_tokens=max_tokens)
            # glm can spend the whole token budget on reasoning despite thinking being disabled
            if not reply[0]:
                raise ValueError("empty reply")
            return reply
        except Exception:
            if attempt == ATTEMPTS - 1:
                raise
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * 2**attempt)
    raise AssertionError("unreachable")


async def run_stage(
    stage: str,
    prompts: dict[str, str],
    out_path: Path,
    *,
    client: Any,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, Any]:
    """Send every prompt once; each reply is appended as soon as it arrives."""
    semaphore = asyncio.Semaphore(concurrency)
    summary: dict[str, Any] = {"completed": 0, "failed": 0, "failures": []}

    async def one(qid: str, prompt: str) -> None:
        async with semaphore:
            started = time.monotonic()
            try:
                text, in_tok, out_tok = await _complete_with_retry(
                    client, prompt, MAX_TOKENS[stage]
                )
            except Exception as exc:
                summary["failed"] += 1
                summary["failures"].append(f"{qid}: {exc!r}")
                return
            row = {
                "question_id": qid,
                "text": text,
                "model": getattr(client, "model", None),
                "prompt_sha256": lme.prompt_sha(prompt),
                "in_tok": in_tok,
                "out_tok": out_tok,
                "seconds": round(time.monotonic() - started, 3),
            }
            lme.append_jsonl(out_path, [row])
            summary["completed"] += 1

    await asyncio.gather(*(one(qid, prompt) for qid, prompt in prompts.items()))
    return summary


def _stage_manifest(
    stage: str,
    run: str,
    data_dir: Path,
    client: ChatModel,
    concurrency: int,
    code: dict[str, Any],
) -> dict[str, Any]:
    out_path = lme.stage_output_path(data_dir, stage, run)
    rows = lme.read_jsonl(out_path)
    upstream = lme.upstream_manifest()
    return {
        "code": code,
        "provider": client.provider,
        "model": client.model,
        "temperature": 0,
        "thinking": "disabled",
        "max_tokens": MAX_TOKENS[stage],
        "concurrency": concurrency,
        "templates_sha256": (
            upstream["answer_template_sha256"]
            if stage == "answer"
            else upstream["judge_templates_sha256"]
        ),
        "rows": len(rows),
        "questions": len({row["question_id"] for row in rows}),
        "in_tok": sum(row["in_tok"] for row in rows),
        "out_tok": sum(row["out_tok"] for row in rows),
        "seconds": sum(row["seconds"] for row in rows),
        "output_sha256": lme.sha256_file(out_path) if out_path.exists() else None,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=lme.STAGES)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=lme.DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=lme.DEFAULT_MANIFEST)
    parser.add_argument("--variant", choices=lme.VARIANTS, default="baseline")
    parser.add_argument("--gate", choices=lme.GATES, default="on")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args(argv)

    code = lme.code_revision()
    load_dotenv()
    run = lme.run_name(args.variant, args.gate)
    questions = {q["question_id"]: q for q in lme.load_dataset(args.dataset)}
    client = ChatModel.from_env(os.environ, model=args.model)
    prompts = pending_prompts(args.stage, args.data_dir, run, questions)
    print(f"{args.stage} ({run}): {len(prompts)} pending", flush=True)
    out_path = lme.stage_output_path(args.data_dir, args.stage, run)
    summary = asyncio.run(
        run_stage(args.stage, prompts, out_path, client=client, concurrency=args.concurrency)
    )
    for failure in summary["failures"]:
        print(f"failed: {failure}")
    lme.update_manifest(
        args.manifest,
        f"{args.stage}{lme.run_suffix(run)}",
        _stage_manifest(args.stage, run, args.data_dir, client, args.concurrency, code),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}))


if __name__ == "__main__":
    main()
