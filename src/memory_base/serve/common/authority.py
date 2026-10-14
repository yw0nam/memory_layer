"""The consolidator authority: an admin key carrying the `consolidator` author."""

from __future__ import annotations

from typing import Any

from starlette.responses import JSONResponse

from memory_base.serve.common.http import error

CONSOLIDATE_AUTHOR = "consolidator"


def consolidator_denied(key: Any) -> JSONResponse | None:
    """None for an admin key carrying `consolidator`; otherwise the 403 response."""
    if key.is_admin and CONSOLIDATE_AUTHOR in key.authors:
        return None
    return error(f"admin key with {CONSOLIDATE_AUTHOR!r} in its authors required", 403)
