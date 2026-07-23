"""Source adapter contracts and source-neutral history records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class SourceFile:
    path: Path
    mtime: float
    size: int


@dataclass
class Message:
    role: str
    text: str
    timestamp: float | None
    session_id: str
    cwd: str | None = None
    git_branch: str | None = None
    uuid: str | None = None
    parent_uuid: str | None = None
    tool_names: tuple[str, ...] = ()
    tool_error: bool = False


@dataclass
class Burst:
    role: str
    text: str
    messages: list[Message]
    social_weight: float
    mean_idf: float | None = None


@dataclass
class Session:
    session_id: str
    messages: list[Message]
    transcript: str
    ts_last_active: float
    tool_names: list[str] = field(default_factory=list)
    tool_error_count: int = 0


class SourceAdapter(ABC):
    source_type: ClassVar[str]

    @abstractmethod
    def discover(self) -> list[SourceFile]: ...

    @abstractmethod
    def parse(self, text: str, file: SourceFile) -> list[Session]: ...

    @abstractmethod
    def has_social(self, burst: Burst) -> bool: ...
