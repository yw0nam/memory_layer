"""Every write path refuses a credential before any model call or content write."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import httpx
import pytest
from loguru import logger

from memory_base.adapters import document
from memory_base.ingest import enrich
from memory_base.serve import api, ingest_api, job_store, mcp_server, notes

# Uppercase on purpose: tag normalization lowercases, which would hide it from the detector.
AWS_KEY = "AKIA" + "Q" * 16
GITLAB_TOKEN = "glpat-" + "q" * 20


class Calls:
    """Records every call a refused write must never make."""

    def __init__(self) -> None:
        self.made: list[str] = []

    def forbid(self, monkeypatch, target, name: str) -> None:
        def record(*args, **kwargs):
            self.made.append(name)
            raise AssertionError(f"{name} must not be reached")

        async def arecord(*args, **kwargs):
            record()

        original = getattr(target, name)
        is_async = asyncio.iscoroutinefunction(original)
        monkeypatch.setattr(target, name, arecord if is_async else record)


@pytest.fixture
def calls(monkeypatch):
    recorder = Calls()
    for target, name in [
        (notes, "chat_json"),
        (notes, "embed_text"),
        (enrich, "chat_json"),
        (ingest_api, "summarize_and_tag"),
        (ingest_api, "embed_text"),
        (ingest_api, "replace_document_rows"),
        (ingest_api, "_existing_document_state"),
        (job_store, "admit_document"),
    ]:
        recorder.forbid(monkeypatch, target, name)

    @asynccontextmanager
    async def acquire(*args, **kwargs):
        recorder.made.append("db.acquire")
        raise AssertionError("db must not be reached")
        yield

    async def namespace_exists(name):
        return name == "default"

    monkeypatch.setattr(notes.db, "acquire", acquire)
    monkeypatch.setattr(ingest_api.namespaces, "namespace_exists", namespace_exists)
    return recorder


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), level="TRACE")
    yield lines
    logger.remove(sink)


def _rest():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app),
        base_url="http://testserver",
        headers={"X-API-Key": "test-key"},
    )


def _mcp_through_rest(monkeypatch):
    monkeypatch.setenv("MEMORY_API_KEY", "test-key")
    monkeypatch.setattr(
        mcp_server,
        "_client",
        lambda: httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url=mcp_server.REST_URL
        ),
    )


# ---- notes ------------------------------------------------------------------


def test_save_note_refuses_a_credential_before_the_gate_embedding_and_db(calls, log_lines):
    with pytest.raises(notes.CredentialNoteError) as refused:
        asyncio.run(
            notes.save_note(
                f"deploy with {AWS_KEY} from the vault", tags=["deploy"], kind="work", author="a"
            )
        )
    assert refused.value.secret_type == "AWS Access Key"
    assert isinstance(refused.value, ValueError)
    assert str(refused.value) == (
        "note contains a credential (AWS Access Key); store the fact without the secret"
    )
    assert calls.made == []
    assert not any(AWS_KEY in line for line in log_lines)


def test_save_note_refuses_a_credential_in_a_raw_tag(calls):
    with pytest.raises(notes.CredentialNoteError) as refused:
        asyncio.run(
            notes.save_note(
                "prefer ruff for linting", tags=["deploy", AWS_KEY], kind="work", author="a"
            )
        )
    assert refused.value.secret_type == "AWS Access Key"
    assert calls.made == []


def _save_memory_body(content, tags):
    return {"content": content, "author": "claude-code", "kind": "work", "tags": tags}


@pytest.mark.parametrize(
    ("content", "tags"),
    [(f"the deploy key is {AWS_KEY}", ["deploy"]), ("prefer ruff for linting", [AWS_KEY])],
    ids=["content", "tag"],
)
def test_rest_save_memory_maps_a_credential_to_409_without_echoing_it(calls, content, tags):
    async def run():
        async with _rest() as client:
            return await client.post("/save_memory", json=_save_memory_body(content, tags))

    response = asyncio.run(run())
    assert response.status_code == 409
    assert "note contains a credential (AWS Access Key)" in response.json()["error"]
    assert AWS_KEY not in response.text
    assert calls.made == []


def test_mcp_save_memory_surfaces_the_credential_refusal(monkeypatch, calls):
    _mcp_through_rest(monkeypatch)
    with pytest.raises(ValueError) as refused:
        asyncio.run(
            mcp_server.save_memory(
                content=f"the deploy key is {AWS_KEY}",
                author="claude-code",
                tags=["deploy"],
                kind="work",
            )
        )
    assert "note contains a credential (AWS Access Key)" in str(refused.value)
    assert AWS_KEY not in str(refused.value)
    assert calls.made == []


# ---- upload metadata ------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "filename", "secret_type"),
    [
        ({"tags": ["policy", AWS_KEY]}, "guide.md", "AWS Access Key"),
        ({"origin": f"vault:{AWS_KEY}"}, "guide.md", "AWS Access Key"),
        ({}, f"{AWS_KEY}.md", "AWS Access Key"),
        ({"document_id": GITLAB_TOKEN}, "guide.md", "GitLab Token"),
    ],
    ids=["tag", "origin", "filename", "document_id"],
)
def test_rest_upload_refuses_a_credential_in_its_fields_before_admission(
    calls, data, filename, secret_type
):
    async def run():
        async with _rest() as client:
            return await client.post(
                "/ingest/document", data=data, files={"file": (filename, b"plain body")}
            )

    response = asyncio.run(run())
    assert response.status_code == 400
    assert f"contains a credential ({secret_type})" in response.json()["error"]
    assert AWS_KEY not in response.text and GITLAB_TOKEN not in response.text
    assert calls.made == []
    assert not ingest_api.INGEST_SPOOL.exists() or not any(ingest_api.INGEST_SPOOL.iterdir())


@pytest.mark.parametrize(
    ("kwargs", "secret_type"),
    [
        ({"filename": f"{AWS_KEY}.md"}, "AWS Access Key"),
        ({"filename": "guide.md", "document_id": GITLAB_TOKEN}, "GitLab Token"),
    ],
    ids=["filename", "document_id"],
)
def test_mcp_ingest_document_surfaces_the_field_refusal(monkeypatch, calls, kwargs, secret_type):
    _mcp_through_rest(monkeypatch)
    with pytest.raises(ValueError) as refused:
        asyncio.run(mcp_server.ingest_document(content="plain body", **kwargs))
    assert f"contains a credential ({secret_type})" in str(refused.value)
    assert AWS_KEY not in str(refused.value) and GITLAB_TOKEN not in str(refused.value)
    assert calls.made == []


# ---- document and CSV content -----------------------------------------------


def _queued_job(spool, filename, mode="force"):
    now = time.time()
    return ingest_api.IngestJob(
        job_id="job-1",
        document_id=filename.lower(),
        namespace="default",
        mode=mode,
        filename=filename,
        spool_path=str(spool),
        status="running",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def terminal(monkeypatch):
    recorded: list[tuple[str, str | None]] = []

    async def mark_terminal(job, status, error=None):
        recorded.append((status, error))
        job.status, job.error, job.stage = status, error, "done"

    monkeypatch.setattr(job_store, "mark_terminal", mark_terminal)
    return recorded


def _markdown_with_secret(tmp_path):
    spool = tmp_path / "runbook.md"
    body = "# Runbook\n\n" + "Rotate the deploy credentials every quarter. " * 20
    spool.write_text(f"{body}\n\nThe current key is {AWS_KEY}.\n")
    return spool


def _csv_with_secret_in_row_500(tmp_path):
    spool = tmp_path / "ledger.csv"
    rows = [f"svc-{index},ok" for index in range(600)]
    rows[499] = f"svc-499,{AWS_KEY}"
    spool.write_text("service,status\n" + "\n".join(rows) + "\n")
    return spool


@pytest.mark.parametrize(
    "make_spool", [_markdown_with_secret, _csv_with_secret_in_row_500], ids=["markdown", "csv"]
)
def test_worker_fails_a_credential_document_whole_and_cleans_the_spool(
    tmp_path, calls, terminal, log_lines, make_spool
):
    spool = make_spool(tmp_path)
    job = _queued_job(spool, spool.name)

    asyncio.run(job_store._run_claimed(job))

    expected = "document contains a credential (AWS Access Key); remove it and upload again"
    assert terminal == [("failed", expected)]
    assert job.status == "failed"
    assert job.error == expected
    assert not spool.exists()
    assert calls.made == []
    assert not any(AWS_KEY in line for line in log_lines)


@pytest.mark.parametrize(
    "make_spool", [_markdown_with_secret, _csv_with_secret_in_row_500], ids=["markdown", "csv"]
)
def test_same_hash_reupload_of_a_credential_document_is_still_refused(
    monkeypatch, tmp_path, calls, make_spool
):
    spool = make_spool(tmp_path)
    same_hash = ingest_api._file_hash(spool)

    async def existing(document_id, namespace="default", schema=None):
        return same_hash, True

    monkeypatch.setattr(ingest_api, "_existing_document_state", existing)
    job = _queued_job(spool, spool.name, mode="upsert")
    with pytest.raises(document.CredentialDocumentError) as refused:
        asyncio.run(ingest_api.run_document_job(job))
    assert refused.value.secret_type == "AWS Access Key"
    assert AWS_KEY not in str(refused.value)
    assert job.status != "no_op"
    assert calls.made == []
