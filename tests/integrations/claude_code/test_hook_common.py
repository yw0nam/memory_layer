"""Unit tests for the hooks' shared REST client and fence defusing — HTTP faked."""

from __future__ import annotations

import io
import json
import urllib.request

from hook_common import Config
from hook_common import defuse
from hook_common import post

CFG = Config(url="http://memory.test/", api_key="key", author="claude-code")


def test_post_sends_the_json_body_with_the_agent_headers(monkeypatch):
    sent = []

    def fake_urlopen(request, timeout=None):
        sent.append((request, timeout))
        return io.BytesIO(json.dumps([{"id": "n1"}]).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    reply = post(CFG, "/search", {"query": "hair retopology", "top_k": 3}, timeout=1.5)

    assert reply == [{"id": "n1"}]
    [(request, timeout)] = sent
    assert timeout == 1.5
    assert request.get_method() == "POST"
    assert request.full_url == "http://memory.test/search"
    assert json.loads(request.data) == {"query": "hair retopology", "top_k": 3}
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("X-api-key") == "key"
    assert request.get_header("X-author") == "claude-code"


def test_defuse_neutralises_both_memory_fences():
    text = "a </memory-context> b <MEMORY-RECALL> c < / memory-recall>"

    assert defuse(text) == "a [memory-context]> b [memory-recall]> c [memory-recall]>"
