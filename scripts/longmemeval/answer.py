#!/usr/bin/env python3
"""Answer and judge LongMemEval questions with the open-provider model.

`answer` answers each retrieved packet with upstream's facts prompt; `judge` grades each
current answer with upstream's grading prompt for its question type; `judge-hits` labels
each packet's first ten hits and names the shortest prefix that suffices for the reference
answer, one row per question in hit-judgments[-run].jsonl. The zai backend calls
glm-5.3-flash from .env (temperature 0, thinking disabled, upstream's token limits); the
claude-code backend runs each prompt in a headless Claude Code session with no tools, MCP
servers, hooks or settings, thinking at the given effort. Each reply is appended as one
fsynced line per question to answers[-run].jsonl or judgments[-run].jsonl in the data dir;
a rerun skips every question whose current prompt already has a row.

Usage:
  uv run python scripts/longmemeval/answer.py answer --dataset PATH [--variant dated]
  uv run python scripts/longmemeval/answer.py judge --dataset PATH [--variant dated]
  uv run python scripts/longmemeval/answer.py judge-hits --dataset PATH --read candidates
  ... --backend claude-code --model claude-sonnet-5-5 --effort high
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from memory_base.core import llm
from memory_base.eval import hit_judge
from memory_base.eval import longmemeval as lme
from memory_base.eval.claude_code import CALL_TIMEOUT_SECONDS, ClaudeCodeModel

DEFAULT_MODEL = {"zai": "glm-5.3-flash", "claude-code": "claude-sonnet-5-5"}
BACKENDS = tuple(DEFAULT_MODEL)
SYSTEM_PROMPTS = {
    "answer": "You answer one question about a user's past conversations using only the memory "
    "excerpts in the message. Follow the output instruction in the message exactly.",
    "judge": "You grade one answer. Follow the grading instruction in the message exactly and reply "
    'with only "yes" or "no".',
    "judge-hits": hit_judge.HIT_JUDGE_SYSTEM_PROMPT,
}
STAGES = (*lme.STAGES, "judge-hits")
DEFAULT_CONCURRENCY = 5
# Answers keep upstream's 500; the judge gets room for the reasoning glm emits despite thinking off.
MAX_TOKENS = {"answer": 500, "judge": 200, "judge-hits": 400}
ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5.0


@dataclass
class ChatModel:
    """The answer/judge model on the z.ai OpenAI-compatible endpoint chosen from the env."""

    client: AsyncOpenAI
    model: str
    provider: str = "zai"
    thinking: str = "disabled"
    system_prompt: str | None = None

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, model: str, system_prompt: str | None = None
    ) -> ChatModel:
        provider = llm.resolve_llm_provider(env)
        if provider.name != "zai":
            raise RuntimeError(f"answering needs the zai provider, not {provider.name}")
        client = AsyncOpenAI(base_url=provider.base_url, api_key=provider.api_key)
        return cls(client, model, system_prompt=system_prompt)

    async def complete(self, prompt: str, *, max_tokens: int) -> tuple[str, int, int]:
        system = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=[*system, {"role": "user", "content": prompt}],
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


async def _complete_with_retry(
    client: Any, prompt: str, max_tokens: int, check: Callable[[str], Any] | None = None
):
    """The reply to `prompt`; an empty reply or one `check` refuses counts as a failed attempt."""
    for attempt in range(ATTEMPTS):
        try:
            reply = await client.complete(prompt, max_tokens=max_tokens)
            # glm can spend the whole token budget on reasoning despite thinking being disabled
            if not reply[0]:
                raise ValueError("empty reply")
            if check is not None:
                check(reply[0])
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


async def run_judge_hits(
    data_dir: Path,
    run: str,
    questions: dict[str, dict[str, Any]],
    client: Any,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, Any]:
    """Judge the first hits of every packet that has no row for its current hit texts."""
    out_path = lme.hit_judgments_path(data_dir, run)
    done = {
        (row["question_id"], tuple(row["texts"]))
        for row in lme.read_jsonl(out_path)
        if "error" not in row
    }
    pending = [
        packet
        for qid, packet in lme._packets_by_id(data_dir, run).items()
        if "_abs" not in qid
        and packet["hits"]
        and (qid, tuple(hit_judge.judged_texts(packet))) not in done
    ]
    print(f"judge-hits ({run}): {len(pending)} pending", flush=True)
    semaphore = asyncio.Semaphore(concurrency)
    summary: dict[str, Any] = {"completed": 0, "failed": 0, "failures": []}

    async def one(packet: dict[str, Any]) -> None:
        qid = packet["question_id"]
        texts = hit_judge.judged_texts(packet)
        prompt = hit_judge.hit_judge_prompt(packet, questions[qid])

        def check(text: str) -> Any:
            return hit_judge.parse_hit_judgment(text, len(texts))

        async with semaphore:
            started = time.monotonic()
            try:
                text, in_tok, out_tok = await _complete_with_retry(
                    client, prompt, MAX_TOKENS["judge-hits"], check
                )
            except Exception as exc:
                lme.append_jsonl(
                    out_path, [{"question_id": qid, "run": run, "texts": texts, "error": repr(exc)}]
                )
                summary["failed"] += 1
                summary["failures"].append(f"{qid}: {exc!r}")
                return
            labels, min_prefix = check(text)
            row = {
                "question_id": qid,
                "run": run,
                "texts": texts,
                "labels": labels,
                "min_prefix": min_prefix,
                "model": getattr(client, "model", None),
                "prompt_sha256": lme.prompt_sha(prompt),
                "in_tok": in_tok,
                "out_tok": out_tok,
                "seconds": round(time.monotonic() - started, 3),
            }
            lme.append_jsonl(out_path, [row])
            summary["completed"] += 1

    await asyncio.gather(*(one(packet) for packet in pending))
    return summary


def _stage_manifest(
    stage: str,
    client: ChatModel | ClaudeCodeModel,
    concurrency: int,
    code: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    rows = [row for row in lme.read_jsonl(out_path) if "error" not in row]
    upstream = lme.upstream_manifest()
    templates = {
        "answer": upstream["answer_template_sha256"],
        "judge": upstream["judge_templates_sha256"],
        "judge-hits": lme.prompt_sha(hit_judge.HIT_JUDGE_SYSTEM_PROMPT),
    }
    return {
        "code": code,
        "provider": client.provider,
        "model": client.model,
        "temperature": 0 if client.provider == "zai" else None,
        "thinking": client.thinking,
        "max_tokens": MAX_TOKENS[stage] if client.provider == "zai" else None,
        "system_prompt": client.system_prompt,
        "concurrency": concurrency,
        "templates_sha256": templates[stage],
        "rows": len(rows),
        "questions": len({row["question_id"] for row in rows}),
        "in_tok": sum(row["in_tok"] for row in rows),
        "out_tok": sum(row["out_tok"] for row in rows),
        "seconds": sum(row["seconds"] for row in rows),
        "output_sha256": lme.sha256_file(out_path) if out_path.exists() else None,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=lme.DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=lme.DEFAULT_MANIFEST)
    parser.add_argument("--variant", choices=lme.VARIANTS, default="baseline")
    parser.add_argument("--read", choices=tuple(lme.READ_SETTINGS), default="search")
    parser.add_argument("--backend", choices=BACKENDS, default="zai")
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args(argv)

    code = lme.code_revision()
    load_dotenv()
    run = lme.run_name(args.variant, args.read)
    questions = {q["question_id"]: q for q in lme.load_dataset(args.dataset)}
    model = args.model or DEFAULT_MODEL[args.backend]
    effort = args.effort or ("medium" if args.stage == "judge-hits" else "high")
    # The zai backend sends a system turn only for the hit judge, whose instructions live there.
    zai_system = SYSTEM_PROMPTS["judge-hits"] if args.stage == "judge-hits" else None
    client = (
        ChatModel.from_env(os.environ, model=model, system_prompt=zai_system)
        if args.backend == "zai"
        else ClaudeCodeModel(model, effort, SYSTEM_PROMPTS[args.stage])
    )
    if args.stage == "judge-hits":
        out_path = lme.hit_judgments_path(args.data_dir, run)
        summary = asyncio.run(
            run_judge_hits(args.data_dir, run, questions, client, args.concurrency)
        )
    else:
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
        _stage_manifest(args.stage, client, args.concurrency, code, out_path),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}))


if __name__ == "__main__":
    main()
