"""Claude Code Stop hook: record the tail of the assistant's last message for the next prompt.

Prints nothing and always exits 0. Stdlib only.
"""

from __future__ import annotations

import json
import sys

from hook_common import Config
from recall import tail


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if Config.from_env().configured:
            tail.save(payload.get("session_id") or "", payload.get("last_assistant_message") or "")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
