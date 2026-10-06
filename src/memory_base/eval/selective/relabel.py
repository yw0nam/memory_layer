"""Keep or skip labels for the dev personas' gold facts, from two judges and the owner.

Memora's memory flag means "the benchmark asks about it later", not "worth keeping". `run`
asks Sonnet (headless `claude -p`) and glm-5.3-flash (z.ai, from .env) to label every dev
fact keep or skip with the save_memory description as the whole policy; its sha256 is the
contract version. Verdicts append to <root>/relabel/judgments.jsonl, and a rerun judges
only facts without a verdict under the current contract. Disagreements go to
disagreements.md and a seeded 10% sample of agreements to audit.md, both Markdown tables
whose last column is the owner's keep or skip; a rewrite keeps the owner's entries.
`labels` reads the owner column back into labels.jsonl: an owner entry wins, an agreement
stands, and a disagreement without an owner entry stays pending.

  uv run python -m memory_base.eval.selective.relabel run [--zai-key-var SUB_ZAI_API_KEY]
  uv run python -m memory_base.eval.selective.relabel labels
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import random
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from memory_base.core import llm
from memory_base.eval.claude_code import CALL_TIMEOUT_SECONDS, ClaudeCodeModel, strip_fence
from memory_base.eval.longmemeval import (
    append_jsonl,
    read_jsonl,
    read_manifest,
    update_manifest,
    write_jsonl_atomic,
)
from memory_base.eval.selective import DEFAULT_ROOT, DEV_PERSONAS, manifest_path
from memory_base.serve.notes.tools import save_memory

Judge = Callable[[dict[str, Any]], Awaitable[Any]]

LABELS = ("keep", "skip")
SCHEMA = {
    "type": "object",
    "properties": {"label": {"enum": list(LABELS)}, "reason": {"type": "string"}},
    "required": ["label", "reason"],
}
SONNET_MODEL = "claude-sonnet-5-5"
SONNET_EFFORT = "medium"
AUDIT_SHARE = 0.10
DEFAULT_CONCURRENCY = 5
ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5.0


def system_prompt(policy: str) -> str:
    return (
        "You decide whether an AI agent with long-term memory should save one fact its user "
        "told it. The agent saves notes only through its save_memory tool, and the tool "
        "description below is the whole policy; judge by it alone.\n\n"
        f"<save_memory description>\n{policy}\n</save_memory description>\n\n"
        'Reply with JSON only: {"label": "keep" | "skip", "reason": "<one line>"}'
    )


def judge_prompt(fact: dict[str, Any]) -> str:
    return (
        f"Conversation date: {fact['date']}\n"
        f"What the user said:\n{fact['evidence']}\n\n"
        f"The fact in it, as the dataset records it ({fact['operation']}, {fact['category']}): "
        f"{json.dumps(fact['value'], ensure_ascii=False)}\n"
    )


def parse_judgment(reply: Any) -> dict[str, str]:
    if not isinstance(reply, dict) or reply.get("label") not in LABELS:
        raise ValueError(f"not a keep/skip judgment: {reply!r}")
    return {"label": reply["label"], "reason": str(reply.get("reason", ""))}


def sonnet_judge(policy: str) -> Judge:
    model = ClaudeCodeModel(SONNET_MODEL, SONNET_EFFORT, system_prompt(policy))

    async def judge(fact: dict[str, Any]) -> Any:
        text, _, _ = await model.complete(judge_prompt(fact), max_tokens=200)
        return json.loads(strip_fence(text))

    return judge


def glm_judge(policy: str) -> Judge:
    async def judge(fact: dict[str, Any]) -> Any:
        messages = [
            {"role": "system", "content": system_prompt(policy)},
            {"role": "user", "content": judge_prompt(fact)},
        ]
        return await llm.chat_json(messages, SCHEMA, timeout=CALL_TIMEOUT_SECONDS)

    return judge


def _dev_facts(root: Path) -> list[dict[str, Any]]:
    return [f for p in DEV_PERSONAS for f in read_jsonl(root / "memora" / p / "facts.jsonl")]


def _verdicts(path: Path, contract: str) -> dict[str, dict[str, dict[str, Any]]]:
    verdicts: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(path):
        if row["contract"] == contract:
            verdicts.setdefault(row["fact_id"], {})[row["judge"]] = row
    return verdicts


async def _judge_all(
    facts: list[dict[str, Any]],
    judges: dict[str, Judge],
    path: Path,
    contract: str,
    concurrency: int,
) -> None:
    done = _verdicts(path, contract)
    limits = {name: asyncio.Semaphore(concurrency) for name in judges}

    async def one(name: str, fact: dict[str, Any]) -> None:
        async with limits[name]:
            for attempt in range(ATTEMPTS):
                try:
                    verdict = parse_judgment(await judges[name](fact))
                    break
                except Exception as exc:
                    if attempt == ATTEMPTS - 1:
                        logger.warning("{} left {} unjudged: {}", name, fact["fact_id"], exc)
                        return
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS * 2**attempt)
        append_jsonl(
            path, [{"fact_id": fact["fact_id"], "judge": name, "contract": contract, **verdict}]
        )

    await asyncio.gather(
        *(
            one(name, fact)
            for fact in facts
            for name in judges
            if name not in done.get(fact["fact_id"], {})
        )
    )


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def read_owner(path: Path) -> dict[str, str]:
    """fact_id -> the owner column, for rows where the owner wrote something."""
    if not path.exists():
        return {}
    owners = {}
    for line in path.read_text().splitlines():
        if not line.startswith("| "):
            continue
        # An escaped pipe inside a cell only shifts the middle cells; the ends stay put.
        cells = [cell.strip() for cell in line.strip()[1:-1].split("|")]
        if cells[0] != "fact_id" and cells[-1]:
            owners[cells[0]] = cells[-1]
    return owners


def _write_table(
    path: Path,
    title: str,
    facts: list[dict[str, Any]],
    verdicts: dict[str, dict[str, dict[str, Any]]],
    judges: list[str],
) -> None:
    owners = read_owner(path)
    header = ["fact_id", "date", "category", "operation", "fact", "evidence", *judges, "owner"]
    lines = [
        f"# {title}",
        "",
        "Write keep or skip in the owner column.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for fact in facts:
        by = verdicts[fact["fact_id"]]
        cells = [
            fact["fact_id"], fact["date"], fact["category"], fact["operation"],
            json.dumps(fact["value"], ensure_ascii=False), fact["evidence"],
            *(f"{by[name]['label']}: {by[name]['reason']}" for name in judges),
            owners.get(fact["fact_id"], ""),
        ]  # fmt: skip
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    path.write_text("\n".join(lines) + "\n")


async def relabel(
    root: Path,
    judges: dict[str, Judge],
    *,
    contract: str,
    seed: int = 0,
    concurrency: int = DEFAULT_CONCURRENCY,
    models: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Judge every dev fact, write the owner's two tables, and return the counts."""
    started = time.monotonic()
    facts = _dev_facts(root)
    out = root / "relabel"
    out.mkdir(parents=True, exist_ok=True)
    await _judge_all(facts, judges, out / "judgments.jsonl", contract, concurrency)
    verdicts = _verdicts(out / "judgments.jsonl", contract)
    names = list(judges)
    judged = [f for f in facts if set(names) <= set(verdicts.get(f["fact_id"], {}))]
    agreed, disagreed = [], []
    for fact in judged:
        labels = {verdicts[fact["fact_id"]][name]["label"] for name in names}
        (agreed if len(labels) == 1 else disagreed).append(fact)
    agreed_ids = sorted(f["fact_id"] for f in agreed)
    sampled = set(random.Random(seed).sample(agreed_ids, round(AUDIT_SHARE * len(agreed_ids))))
    audit = [f for f in agreed if f["fact_id"] in sampled]
    _write_table(out / "disagreements.md", "Judge disagreements", disagreed, verdicts, names)
    _write_table(out / "audit.md", "Agreement audit sample", audit, verdicts, names)
    counts = {
        "facts": len(facts),
        "judged": len(judged),
        "labels": {
            name: dict(Counter(verdicts[f["fact_id"]][name]["label"] for f in judged))
            for name in names
        },
        "agreed": len(agreed),
        "disagreed": len(disagreed),
        "agreement_rate": round(len(agreed) / len(judged), 4) if judged else None,
        "audit": len(audit),
        "seconds": round(time.monotonic() - started, 1),
    }
    update_manifest(
        manifest_path(root),
        "relabel",
        {
            "contract_version": contract,
            "judges": names,
            "models": models or {},
            "seed": seed,
            "counts": counts,
        },
    )
    return counts


def _decision(owner: str, fact_id: str) -> str:
    decision = owner.lower()
    if decision not in LABELS:
        raise ValueError(f"owner entry for {fact_id} is {owner!r}, not keep or skip")
    return decision


def finalize(root: Path) -> dict[str, int]:
    """Write labels.jsonl from the judges' agreements and the owner's entries."""
    out = root / "relabel"
    run = read_manifest(manifest_path(root))["relabel"]
    contract, names = run["contract_version"], run["judges"]
    verdicts = _verdicts(out / "judgments.jsonl", contract)
    owners = {**read_owner(out / "audit.md"), **read_owner(out / "disagreements.md")}
    rows, pending, overturned = [], 0, 0
    for fact in _dev_facts(root):
        fact_id = fact["fact_id"]
        by = verdicts.get(fact_id, {})
        if set(names) - set(by):
            continue
        labels = {by[name]["label"] for name in names}
        if fact_id in owners:
            label, source = _decision(owners[fact_id], fact_id), "owner"
            overturned += len(labels) == 1 and label not in labels
        elif len(labels) == 1:
            label, source = labels.pop(), "judges"
        else:
            pending += 1
            continue
        rows.append({"fact_id": fact_id, "label": label, "source": source, "contract": contract})
    write_jsonl_atomic(out / "labels.jsonl", rows)
    return {
        "labels": len(rows),
        "owner": sum(row["source"] == "owner" for row in rows),
        "pending": pending,
        "audit_overturned": overturned,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="judge the dev facts and write the owner's tables")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    run.add_argument(
        "--zai-key-var", default="ZAI_API_KEY", help="the .env variable holding the z.ai key"
    )
    labels = commands.add_parser("labels", help="read the owner column back into labels.jsonl")
    for sub in (run, labels):
        sub.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    if args.command == "labels":
        print(json.dumps(finalize(args.root)))
        return

    load_dotenv()
    os.environ["ZAI_API_KEY"] = os.environ[args.zai_key_var]
    provider = llm.resolve_llm_provider(os.environ)
    if provider.name != "zai":
        raise SystemExit(f"the glm judge needs the zai provider, not {provider.name}")
    policy = inspect.getdoc(save_memory)
    counts = asyncio.run(
        relabel(
            args.root,
            {"sonnet": sonnet_judge(policy), "glm": glm_judge(policy)},
            contract=hashlib.sha256(policy.encode()).hexdigest(),
            seed=args.seed,
            concurrency=args.concurrency,
            models={"sonnet": f"{SONNET_MODEL} effort {SONNET_EFFORT}", "glm": provider.model},
        )
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
