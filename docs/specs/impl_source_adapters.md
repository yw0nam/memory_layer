# Implementation spec: source adapter architecture

## Contract

`src/memory_base/adapters/base.py` defines the source-agnostic records and the adapter ABC:

```python
@dataclass(frozen=True)
class SourceFile:
    path: pathlib.Path
    mtime: float
    size: int

@dataclass
class Message: ...   # role, text, timestamp, session_id, cwd, git_branch, tool_names, tool_error

@dataclass
class Burst: ...      # role, text, messages, social_weight, mean_idf

@dataclass
class Session: ...     # session_id, messages, transcript, ts_last_active, tool_names, tool_error_count

class SourceAdapter(ABC):
    source_type: ClassVar[str]           # DB source_type value
    emit_bursts: ClassVar[bool] = True    # whether the core should emit burst rows for this source

    @abstractmethod
    def discover(self) -> list[SourceFile]: ...

    @abstractmethod
    def parse(self, text: str, file: SourceFile) -> list[Session]: ...

    @abstractmethod
    def has_social(self, burst: Burst) -> bool: ...
```

A concrete adapter implementing all three abstract methods can register in `src/memory_base/adapters/__init__.py`'s `ADAPTERS: dict[str, SourceAdapter]` registry, keyed by `source_type`. Instantiating a subclass with a missing abstract method raises `TypeError`.

## Current state

`ADAPTERS` is empty. Interactive agent consoles (Claude Code, Hermes) contribute through the MCP `save_memory` tool instead of a batch file adapter — see `docs/specs/impl_mcp_only_agents.md`. The registry and contract stay in place for a future batch corpus source (e.g. Slack) that is not already open in an interactive agent session.

The source-agnostic selection core (`src/memory_base/ingest/history.py`) — `group_sessions`, `group_bursts`, `triage_heuristic`, `_distill`/`parse_distillation`, `build_corpus_df`/`mean_idf`, `passes_burst_gate`, `build_rows` — consumes `Session`/`Burst` objects regardless of which adapter (if any) produced them, and has no I/O entrypoint of its own.
