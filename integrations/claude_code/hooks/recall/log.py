"""The recall log: one line per call, never holding the prompt or the assistant text."""

from __future__ import annotations

import time
from datetime import datetime
from datetime import timezone

from hook_common import data_dir

LOG = "recall.log"


def write(started: float, keys: list[str], errors: list[str]) -> None:
    """Append one line for this call; a failed write is dropped."""
    elapsed = int((time.monotonic() - started) * 1000)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with open(data_dir() / LOG, "a") as handle:
            handle.write(
                f"{stamp} ms={elapsed} emitted={','.join(keys)} errors={','.join(errors)}\n"
            )
    except OSError:
        pass
