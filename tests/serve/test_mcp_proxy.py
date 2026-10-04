"""Unit tests for the MCP server as an httpx proxy over the REST API.

Each tool body is an httpx call to REST_URL, never the DB/search pipeline. Tests inject a
mocked transport by monkeypatching ``rest_client.client``, a zero-arg factory returning an
``httpx.AsyncClient(base_url=REST_URL, ...)`` (no real network, no DB).
"""

from __future__ import annotations

import asyncio
import inspect
import json

import httpx
import pytest

from memory_base.serve.documents import tools as document_tools
from memory_base.serve.common import rest_client
from memory_base.serve.tables import tools as table_tools
from memory_base.serve.notes import tools as note_tools
from memory_base.serve.search import tools as search_tools


def _patch_client(monkeypatch, handler):
    def fake_client():
        return httpx.AsyncClient(
            base_url=rest_client.REST_URL, transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(rest_client, "client", fake_client)


# ---- search proxying --------------------------------------------------------


def test_search_code_posts_to_search_with_source_code(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_code(query="halfvec index", top_k=5))
    assert captured["method"] == "POST"
    assert captured["path"] == "/search"
    assert captured["json"] == {
        "query": "halfvec index",
        "source": "code",
        "top_k": 5,
    }


def test_search_memory_posts_with_source_memory(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_memory(query="burst gate", top_k=3))
    assert captured["json"] == {
        "query": "burst gate",
        "source": "memory",
        "top_k": 3,
    }


def test_search_memory_forwards_kind_and_tags(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(
        search_tools.search_memory(
            query="decision",
            top_k=4,
            kind="work",
            tags=["infra"],
        )
    )
    assert captured["json"] == {
        "query": "decision",
        "source": "memory",
        "top_k": 4,
        "kind": "work",
        "tags": ["infra"],
    }


def test_search_code_forwards_repo_filter(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_code("marker", repo=["repo_a"]))
    assert captured["json"]["repo"] == ["repo_a"]


def test_search_all_does_not_expose_repo_filter():
    assert "repo" not in inspect.signature(search_tools.search_all).parameters


def test_search_memory_forwards_since_and_until(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_memory("last week", since="2026-08-01", until="2026-08-12"))
    assert captured["json"]["since"] == "2026-08-01"
    assert captured["json"]["until"] == "2026-08-12"


@pytest.mark.parametrize("tool", ["search_all", "search_code"])
def test_only_search_memory_exposes_time_bounds(tool):
    parameters = inspect.signature(getattr(search_tools, tool)).parameters
    assert "since" not in parameters
    assert "until" not in parameters


# ---- list_notes proxying ----------------------------------------------------


def test_list_notes_gets_notes_with_repeated_params(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = request.url.params.multi_items()
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(
        note_tools.list_notes(
            tags=["infra", "db"],
            kind="work",
            since="2026-08-01",
            until="2026-08-12",
            include_archived=True,
            namespace="team-a",
            limit=5,
        )
    )
    assert captured["method"] == "GET"
    assert captured["path"] == "/notes"
    assert sorted(captured["params"]) == sorted(
        [
            ("tags", "infra"),
            ("tags", "db"),
            ("kind", "work"),
            ("since", "2026-08-01"),
            ("until", "2026-08-12"),
            ("include_archived", "true"),
            ("namespace", "team-a"),
            ("limit", "5"),
        ]
    )


def test_search_memory_forwards_the_author_filter(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_memory("who decided", author="natsume"))
    assert captured["json"]["author"] == "natsume"


def test_list_notes_forwards_the_author_filter(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = request.url.params.multi_items()
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.list_notes(author="natsume"))
    assert captured["params"] == [("author", "natsume")]


def test_list_notes_omits_unset_filters(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = request.url.params.multi_items()
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.list_notes())
    assert captured["params"] == []


def test_search_all_does_not_expose_memory_only_filters():
    params = inspect.signature(search_tools.search_all).parameters
    assert "kind" not in params
    assert "tags" not in params


def test_search_all_forwards_include_archived(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_all("query", include_archived=True))
    assert captured["json"]["include_archived"] is True


def test_search_memory_forwards_include_archived(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_memory("query", include_archived=True))
    assert captured["json"]["include_archived"] is True


def test_search_memory_forwards_min_score(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_memory(query="burst gate", min_score=0.3))
    assert captured["json"]["min_score"] == 0.3


@pytest.mark.parametrize("tool", [search_tools.search_all, search_tools.search_memory])
def test_search_tools_forward_budget_tokens(monkeypatch, tool):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(tool(query="burst gate", budget_tokens=4000))
    assert captured["json"]["budget_tokens"] == 4000


def test_search_all_posts_with_source_all(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=[])

    _patch_client(monkeypatch, handler)
    asyncio.run(search_tools.search_all(query="anything", top_k=10))
    assert captured["json"] == {
        "query": "anything",
        "source": "all",
        "top_k": 10,
    }


def test_search_returns_rest_response_body_unmodified(monkeypatch):
    hits = [
        {"source": "code", "ref": "a.py:L1-L2", "date": "2026-01-01", "score": 0.9, "text": "x"}
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=hits)

    _patch_client(monkeypatch, handler)
    result = asyncio.run(search_tools.search_code(query="q", top_k=5))
    assert result == hits


# ---- lifecycle tool proxying ------------------------------------------------


def test_list_memory_duplicates_gets_admin_duplicates(monkeypatch):
    captured = {}
    payload = {"pairs": [{"a": {"id": "note:a"}, "b": {"id": "note:b"}, "score": 0.97}]}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = request.url.params.multi_items()
        captured["header"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=payload)

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("MEMORY_API_KEY", "env-key")
    result = asyncio.run(note_tools.list_memory_duplicates(threshold=0.95, kind="work", limit=5))
    assert captured["method"] == "GET"
    assert captured["path"] == "/admin/duplicates"
    assert sorted(captured["params"]) == sorted(
        [("threshold", "0.95"), ("kind", "work"), ("limit", "5")]
    )
    assert captured["header"] == "env-key"
    assert result == payload


def test_list_memory_duplicates_omits_unset_params(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = request.url.params.multi_items()
        return httpx.Response(200, json={"pairs": []})

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.list_memory_duplicates())
    assert captured["params"] == []


def test_archive_notes_posts_ids_and_author(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        captured["header"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={"rows": []})

    _patch_client(monkeypatch, handler)
    monkeypatch.setenv("MEMORY_API_KEY", "env-key")
    asyncio.run(note_tools.archive_notes(["note:a"], "natsume"))
    assert captured["method"] == "POST"
    assert captured["path"] == "/admin/archive"
    assert captured["json"] == {"ids": ["note:a"], "author": "natsume"}
    assert captured["header"] == "env-key"


def test_archive_notes_sends_confirm_only_when_true(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"archived": 1})

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.archive_notes(["note:a"], "natsume", confirm=True))
    assert captured["json"]["confirm"] is True


def test_restore_notes_posts_ids(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"rows": []})

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.restore_notes(["note:a"]))
    assert captured["path"] == "/admin/restore"
    assert captured["json"] == {"ids": ["note:a"]}


def test_restore_notes_sends_confirm_only_when_true(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"restored": 1})

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.restore_notes(["note:a"], confirm=True))
    assert captured["json"] == {"ids": ["note:a"], "confirm": True}


def test_delete_notes_posts_ids(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"rows": []})

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.delete_notes(["note:a"]))
    assert captured["path"] == "/admin/notes/delete"
    assert captured["json"] == {"ids": ["note:a"]}


def test_delete_notes_sends_confirm_only_when_true(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"deleted": 1})

    _patch_client(monkeypatch, handler)
    asyncio.run(note_tools.delete_notes(["note:a"], confirm=True))
    assert captured["json"] == {"ids": ["note:a"], "confirm": True}


def test_query_table_posts_sql_and_namespace_and_returns_body(monkeypatch):
    captured = {}
    payload = {
        "columns": ["group", "mean"],
        "rows": [["a", 2.5]],
        "row_count": 1,
        "truncated": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json=payload)

    _patch_client(monkeypatch, handler)
    result = asyncio.run(
        table_tools.query_table(
            "SELECT data->>'group', AVG((data->>'value')::numeric) FROM memory.doc_rows",
            namespace="team-a",
        )
    )

    assert captured == {
        "path": "/tables/query",
        "json": {
            "sql": ("SELECT data->>'group', AVG((data->>'value')::numeric) FROM memory.doc_rows"),
            "namespace": "team-a",
        },
    }
    assert result == payload


def test_ingest_document_posts_text_as_multipart_and_returns_job_reference(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = request.content
        return httpx.Response(
            202,
            json={
                "job_id": "job-1",
                "status": "queued",
                "status_url": "/ingest/jobs/job-1",
            },
        )

    _patch_client(monkeypatch, handler)
    result = asyncio.run(
        document_tools.ingest_document(
            "# Guide",
            "guide.md",
            document_id="guide",
            origin="mcp:test",
            mode="force",
        )
    )
    assert captured["path"] == "/ingest/document"
    assert captured["content_type"].startswith("multipart/form-data")
    assert b"# Guide" in captured["body"]
    assert b'name="document_id"' in captured["body"]
    assert result == {"job_id": "job-1", "status_url": "/ingest/jobs/job-1"}


def test_ingest_document_forwards_tags_as_repeated_form_fields(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            202,
            json={"job_id": "job-1", "status": "queued", "status_url": "/ingest/jobs/job-1"},
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(document_tools.ingest_document("# Guide", "guide.md", tags=["zx bank", "policy"]))
    assert captured["body"].count(b'name="tags"') == 2
    assert b"zx bank" in captured["body"]
    assert b"policy" in captured["body"]


def test_ingest_document_omitted_tags_send_no_tags_field(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            202,
            json={"job_id": "job-1", "status": "queued", "status_url": "/ingest/jobs/job-1"},
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(document_tools.ingest_document("# Guide", "guide.md"))
    assert b'name="tags"' not in captured["body"]


def test_ingest_document_mcp_rejects_binary_formats():
    with pytest.raises(ValueError, match="text formats only"):
        asyncio.run(document_tools.ingest_document("content", "guide.pdf"))


def test_ingest_document_mcp_accepts_csv(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content
        return httpx.Response(
            202,
            json={"job_id": "job-1", "status": "queued", "status_url": "/ingest/jobs/job-1"},
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(document_tools.ingest_document("name,value\none,1\n", "table.csv"))
    assert b"name,value" in captured["body"]


# ---- remove_document proxying ----------------------------------------------


def test_remove_document_deletes_with_default_namespace(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200, json={"document_id": "guide.md", "namespace": "default", "deleted": 3}
        )

    _patch_client(monkeypatch, handler)
    result = asyncio.run(document_tools.remove_document("guide.md"))
    assert captured["method"] == "DELETE"
    assert captured["path"] == "/ingest/documents/guide.md"
    assert captured["params"] == {}
    assert result == {"document_id": "guide.md", "namespace": "default", "deleted": 3}


def test_remove_document_forwards_namespace(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200, json={"document_id": "guide.md", "namespace": "team-a", "deleted": 1}
        )

    _patch_client(monkeypatch, handler)
    asyncio.run(document_tools.remove_document("guide.md", namespace="team-a"))
    assert captured["params"] == {"namespace": "team-a"}
