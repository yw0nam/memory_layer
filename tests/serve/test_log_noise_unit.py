"""Serving-log noise suppression: /health access-log lines and MCP ping chatter.

The uvicorn access log carries request details in ``record.args``
(client_addr, method, path, http_version, status_code); the filter keys on
those, so plain LogRecords stand in for real requests.
"""

from __future__ import annotations

import logging

from memory_base.serve import api, mcp_server


def _access_record(method: str, path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:1", method, path, "1.1", status),
        exc_info=None,
    )


def test_health_200_access_lines_are_dropped():
    f = api.HealthAccessFilter()
    assert f.filter(_access_record("GET", "/health", 200)) is False


def test_other_access_lines_pass_through():
    f = api.HealthAccessFilter()
    assert f.filter(_access_record("GET", "/health/services", 200)) is True
    assert f.filter(_access_record("GET", "/health", 500)) is True
    assert f.filter(_access_record("POST", "/search", 200)) is True


def test_malformed_records_pass_through():
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="plain message",
        args=None,
        exc_info=None,
    )
    assert api.HealthAccessFilter().filter(record) is True


def test_api_module_attaches_the_filter_to_uvicorn_access():
    assert any(
        isinstance(f, api.HealthAccessFilter) for f in logging.getLogger("uvicorn.access").filters
    )


def test_mcp_request_noise_logger_is_raised_to_warning():
    logger = logging.getLogger("mcp.server.lowlevel.server")
    logger.setLevel(logging.NOTSET)
    mcp_server.quiet_request_noise()
    assert logger.level == logging.WARNING
