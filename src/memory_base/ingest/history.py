"""Source-agnostic selection core for conversation history.

Distills a source session into ``memory_chunks`` rows: session validity,
triage (heuristic + LLM), LLM distillation, bursting with the weighted burst
gate, and IDF corpus statistics. There is no I/O entrypoint here — future
corpus adapters (e.g. Slack) feed sessions through these functions.
Interactive agent consoles contribute through the MCP realtime channel
(save_memory) instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from memory_base.adapters.base import Burst, Message, Session, SourceAdapter
from memory_base.common import LLM_MODEL, SERVICE_TIMEOUT_SECONDS, llm_client

LOGGER = logging.getLogger("history_index")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[\uac00-\ud7a3]{2,}")
TRUNCATION_MARKER = "\n...[truncated]...\n"


@dataclass
class Distillation:
    one_line_question: str
    summary: str
    resolution: str
    references: list[str]
    decisions: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(
            (
                self.one_line_question,
                self.summary,
                self.resolution,
                ", ".join(self.references),
            )
        )


def build_transcript(messages: Sequence[Message], max_chars: int = 100_000) -> str:
    transcript = "\n".join(f"{m.role.upper()}: {m.text}" for m in messages)
    if len(transcript) <= max_chars:
        return transcript
    return (
        transcript[: int(max_chars * 0.6)] + TRUNCATION_MARKER + transcript[-int(max_chars * 0.4) :]
    )


def group_sessions(messages: Iterable[Message], fallback_timestamp: float = 0.0) -> list[Session]:
    grouped: dict[str, list[Message]] = defaultdict(list)
    for message in messages:
        grouped[message.session_id].append(message)
    sessions: list[Session] = []
    for session_id, session_messages in grouped.items():
        timestamps = [m.timestamp for m in session_messages if m.timestamp is not None]
        sessions.append(
            Session(
                session_id=session_id,
                messages=session_messages,
                transcript=build_transcript(session_messages),
                ts_last_active=max(timestamps, default=fallback_timestamp),
                tool_names=[name for m in session_messages for name in m.tool_names],
                tool_error_count=sum(m.tool_error for m in session_messages),
            )
        )
    return sessions


def group_bursts(messages: Sequence[Message], min_chars: int = 200) -> list[Burst]:
    """Group consecutive same-role text messages and apply the length filter."""
    grouped: list[list[Message]] = []
    for message in messages:
        if grouped and grouped[-1][0].role == message.role:
            grouped[-1].append(message)
        else:
            grouped.append([message])

    bursts: list[Burst] = []
    for index, members in enumerate(grouped):
        text = "\n".join(m.text for m in members)
        if len(text) < min_chars:
            continue
        has_error = any(m.tool_error for m in members)
        followed_by_quick_user = False
        if index + 1 < len(grouped) and grouped[index + 1][0].role == "user":
            end_ts = members[-1].timestamp
            next_ts = grouped[index + 1][0].timestamp
            followed_by_quick_user = (
                end_ts is not None and next_ts is not None and 0 <= next_ts - end_ts <= 180
            )
        bursts.append(
            Burst(
                role=members[0].role,
                text=text,
                messages=list(members),
                social_weight=1.5 if has_error or followed_by_quick_user else 1.0,
            )
        )
    return bursts


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def mean_idf(text: str, document_count: int, dfs: dict[str, int]) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return sum(
        math.log((document_count + 1) / (dfs.get(token, 0) + 1)) + 1 for token in tokens
    ) / len(tokens)


def build_corpus_df(transcripts: Iterable[str]) -> tuple[dict[str, int], int]:
    """Return document frequencies and the number of supplied transcripts."""
    dfs: Counter[str] = Counter()
    document_count = 0
    for transcript in transcripts:
        document_count += 1
        dfs.update(tokenize(transcript))
    return dict(dfs), document_count


def triage_heuristic(session: Session) -> str:
    """Cheaply classify a session before any LLM or embedding work."""
    text_messages = [message for message in session.messages if message.text]
    assistant_messages = [message for message in text_messages if message.role == "assistant"]
    if not assistant_messages:
        return "skip"

    user_messages = [message for message in text_messages if message.role == "user"]
    user_chars = sum(len(message.text) for message in user_messages)
    if len(user_messages) <= 2 and user_chars < 200:
        return "skip"

    total_chars = sum(len(message.text) for message in text_messages)
    if total_chars >= 5_000 or len(text_messages) >= 20:
        return "keep"
    return "borderline"


def burst_score(mean_idf_value: float, has_social: bool) -> float:
    return mean_idf_value + (1.0 if has_social else 0.0)


def passes_burst_gate(
    burst: Burst,
    mean_idf_value: float,
    document_count: int,
    has_social: bool | None = None,
) -> bool:
    if document_count < 20:
        return True
    if has_social is None:
        has_social = burst.social_weight > 1.0
    return burst_score(mean_idf_value, has_social) >= 4.0


def parse_distillation(parsed: dict[str, Any]) -> Distillation:
    question = str(parsed.get("one_line_question") or "").strip()
    summary = str(parsed.get("summary") or "").strip()
    resolution = str(parsed.get("resolution") or "unresolved").strip()

    refs = parsed.get("references")
    references = [str(ref).strip() for ref in refs] if isinstance(refs, list) else []

    raw_decisions = parsed.get("decisions")
    decisions = (
        [str(decision).strip() for decision in raw_decisions]
        if isinstance(raw_decisions, list)
        else []
    )
    if not question or not summary:
        raise ValueError("LLM distillation omitted required fields")
    return Distillation(
        question,
        summary,
        resolution,
        [ref for ref in references if ref],
        [decision for decision in decisions if decision][:10],
    )


async def _distill(session: Session, semaphore: asyncio.Semaphore) -> Distillation:
    prompt = (
        "Distill the following agent session in English, even when the transcript "
        "is in Korean or Japanese. Keep code, error strings, identifiers, and file paths "
        "verbatim. Return only a JSON object using this schema: "
        '{"one_line_question":"one-line question for later retrieval in English",'
        '"summary":"3-5 sentence summary in English",'
        '"resolution":"final solution or conclusion in English, or unresolved",'
        '"references":["mentioned files, systems, or commands"],'
        '"decisions":["one sentence in English for each key decision, stating what was '
        'decided and why; use an empty array if there were none"]}\n\n' + session.transcript
    )
    async with semaphore:
        response = await asyncio.wait_for(
            llm_client().chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            ),
            timeout=SERVICE_TIMEOUT_SECONDS,
        )
    content = response.choices[0].message.content or ""
    parsed = json.loads(content)
    raw_decisions = parsed.get("decisions")
    if isinstance(raw_decisions, list) and len(raw_decisions) > 10:
        LOGGER.warning(
            "distillation returned %d decisions for %s; keeping first 10",
            len(raw_decisions),
            session.session_id,
        )
    return parse_distillation(parsed)


def build_rows(
    session: Session,
    distillation: Distillation,
    adapter: SourceAdapter,
    dfs: dict[str, int],
    n: int,
) -> list[dict[str, Any]]:
    source_ref = session.session_id
    metadata = {
        "cwd": session.messages[-1].cwd,
        "git_branch": session.messages[-1].git_branch,
        "tool_names": sorted(set(session.tool_names)),
        "tool_error_count": session.tool_error_count,
    }
    rows = [
        {
            "id": f"{session.session_id}:session",
            "source_ref": source_ref,
            "kind": "session",
            "session_id": session.session_id,
            "raw": session.transcript,
            "distilled": distillation.text,
            "timestamp": session.ts_last_active,
            "idf": None,
            "metadata": metadata,
            "source_type": adapter.source_type,
        }
    ]
    for index, decision in enumerate(distillation.decisions):
        decision_distilled = f"[{distillation.one_line_question}] Decision: {decision}"
        rows.append(
            {
                "id": f"{session.session_id}:decision:{index}",
                "source_ref": source_ref,
                "kind": "decision",
                "session_id": session.session_id,
                "raw": decision,
                "distilled": decision_distilled,
                "timestamp": session.ts_last_active,
                "idf": None,
                "metadata": {**metadata, "index": index},
                "source_type": adapter.source_type,
            }
        )
    if not adapter.emit_bursts:
        return rows
    accepted_bursts: list[Burst] = []
    for burst in group_bursts(session.messages):
        burst.mean_idf = mean_idf(burst.text, n, dfs)
        if passes_burst_gate(burst, burst.mean_idf, n, adapter.has_social(burst)):
            accepted_bursts.append(burst)
    for index, burst in enumerate(accepted_bursts):
        burst_distilled = f"[{distillation.one_line_question}] {burst.text}"
        rows.append(
            {
                "id": f"{session.session_id}:burst:{index}",
                "source_ref": source_ref,
                "kind": "burst",
                "session_id": session.session_id,
                "raw": burst.text,
                "distilled": burst_distilled,
                "timestamp": session.ts_last_active,
                "idf": burst.mean_idf,
                "metadata": {
                    **metadata,
                    "role": burst.role,
                    "social_weight": burst.social_weight,
                },
                "source_type": adapter.source_type,
            }
        )
    return rows
