# Implementation spec: source adapter architecture (Issue #13)

## Principle

**Pure refactor — zero observable behavior change.** The existing test files must pass
unmodified, and `--dry-run` statistics must be identical before and after. Redefining the
claude_code burst signal is a separate decision (#17) and **out of scope** here — current
behavior moves as-is.

## Target structure

```
src/memory_base/adapters/__init__.py    # ADAPTERS: dict[str, SourceAdapter] registry = {"claude_code": ...}
src/memory_base/adapters/base.py        # contract (below)
src/memory_base/adapters/claude_code.py # first adapter: existing code moved
src/memory_base/ingest/history.py       # source-agnostic core: triage, distillation, bursting, gates, incremental state, DB writes
```

### `base.py` contract (ABC inheritance — user decision)

```python
@dataclass(frozen=True)
class SourceFile:
    path: pathlib.Path
    mtime: float
    size: int

class SourceAdapter(ABC):
    source_type: ClassVar[str]     # DB source_type value = adapter module name

    @abstractmethod
    def discover(self) -> list[SourceFile]: ...       # enumerate this source's input files

    @abstractmethod
    def parse(self, text: str, file: SourceFile) -> list[Session]: ...

    @abstractmethod
    def has_social(self, burst: Burst) -> bool: ...   # source-specific social signal
```

- Naming convention: `adapters/<source_type>.py` contains class `<CamelCase>Adapter`
  (e.g. `adapters/claude_code.py` → `ClaudeCodeAdapter`). Registry key = `source_type`.
- Contract violations (unimplemented abstract methods) fail with TypeError at
  instantiation — pinned by tests.
- The `Session`/`Message`/`Burst` dataclasses are source-neutral, so they **move to
  base.py** (re-exported from `memory_base.ingest.history`).
- The claude_code `has_social` implementation = current behavior verbatim:
  `burst.social_weight > 1.0` (social_weight is computed in group_bursts — see the
  move table below).

### Code-move mapping (behavior identical)

| Current (ingest/history.py) | After |
|---|---|
| `parse_jsonl`, `_timestamp` | `adapters/claude_code.py` |
| `~/.claude/projects` glob in `_select_files` | `discover()` in `claude_code.py` (limit/project filters stay in the core — applied to the SourceFile list) |
| `"claude_code"` literal in `_write_file` | `adapter.source_type` |
| `Message`/`Session`/`Burst` | `adapters/base.py` |
| `group_sessions`, `group_bursts`, `build_transcript` | source-neutral — **stay in the core** (logic future sources reuse) |
| triage, distillation, gates, IDF, DB, incremental state | stay in the core |

- The core `run()` iterates the `ADAPTERS` registry (currently claude_code only).
  CLI gains an `--adapter <source_type>` filter (default: all) — the only new behavior
  allowed.
- `passes_burst_gate(burst, mean_idf, n)` gains external social injection:
  `passes_burst_gate(burst, mean_idf, n, has_social=None)` — when `has_social` is None
  it evaluates `burst.social_weight > 1.0` internally, exactly like today. (Keeps
  existing tests passing unmodified.)

### Backward compatibility (no test churn)

Re-export at the top of `ingest/history.py`:
```python
from memory_base.adapters.base import Message, Session, Burst
from memory_base.adapters.claude_code import parse_jsonl
```
The four existing test files importing `from memory_base.ingest.history import
parse_jsonl, Message, ...` must pass unmodified.

## TDD procedure

**Step A (tests)**: new `tests/test_adapter_contract.py` —
- A FakeAdapter (source_type="fake", fixed Session list, has_social always False)
  subclasses the ABC and proves the pure part of the core pipeline is source-agnostic:
  - the core's session processing accepts FakeAdapter sessions through triage/bursting
  - generated DB rows carry source_type "fake" (at the row-dict level, no DB needed)
  - this pins a pure function on the core:
    `build_rows(session, distillation, adapter, dfs, n) -> list[dict]`
    (currently inlined in `_process_file` — extraction required)
- `ADAPTERS` contains "claude_code" and its value is a `SourceAdapter` instance
  (`isinstance` check). A subclass missing an abstract method fails instantiation
  with TypeError.
- red: `from memory_base.adapters.base import ...` and
  `from memory_base.ingest.history import build_rows` raise ImportError.

**Step B (implementation)**: move code into the target structure. Existing + new tests
all green.

## Acceptance criteria

1. Step A red → Step B green. **Existing test files pass unmodified** (full suite + new).
2. Full `--dry-run` statistics equal the main-branch result (orchestrator compares):
   `triage_keep=96 triage_skip_heuristic=9 triage_borderline=6 noise_session=27` etc.
3. `--limit 5 --full` real ingest + incremental re-run work (orchestrator).
4. No changes to embedding, schema, `retrieval/search.py`, or other src files.
5. `ruff format --check` + `ruff check` pass (CI lint job).

## Forbidden

- Behavior changes (gate values, statistic meanings, row contents). New dependencies.
  Changes to `retrieval/search.py`, `common.py`, `serve/*`, `ingest/code.py`.
