"""The exception that rolls a transaction back after a validation-only run."""

from __future__ import annotations

from typing import Any


class DryRun(Exception):
    """Raised inside a transaction to roll it back; `result` is what the run would return."""

    def __init__(self, result: Any) -> None:
        self.result = result
