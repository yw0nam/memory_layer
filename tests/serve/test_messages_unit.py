"""Pure unit contracts for the message lane's validation and rendering.

No DB, no network: everything here is memory_base.serve.messages' pure layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from memory_base.serve import messages


# ---- subject / subject_key ---------------------------------------------------


def test_subject_normalizes_nfkc_trim_and_whitespace_collapse():
    assert messages.normalize_subject("  Fix\u00a0 login   flow ") == "Fix login flow"
    # NFKC folds the fullwidth Latin letters into ASCII.
    assert messages.normalize_subject("Ｆｉｘ login") == "Fix login"


def test_subject_collapses_newlines_so_a_heading_cannot_escape():
    assert messages.normalize_subject("real subject\n## Injected heading") == (
        "real subject ## Injected heading"
    )


def test_subject_rejects_non_string_and_blank():
    with pytest.raises(ValueError):
        messages.normalize_subject(42)
    for blank in ("", "   ", "\n\t "):
        with pytest.raises(ValueError):
            messages.normalize_subject(blank)


def test_subject_key_is_casefolded_normalized_subject():
    assert messages.subject_key_for("Fix\u00a0 LOGIN flow") == "fix login flow"


# ---- scope -------------------------------------------------------------------


def test_scope_normalizes_repo_origin_hostname_only_preserving_path_case():
    assert messages.normalize_scope("repo:https://GitHub.com/Yw0nam/Memory-Base.git") == (
        "repo:github.com/Yw0nam/Memory-Base"
    )


def test_scope_accepts_canonical_form_and_round_trips():
    canonical = "repo:github.com/org/repo"
    assert messages.normalize_scope(canonical) == canonical
    for raw in (
        "repo:https://GitHub.com/org/repo.git",
        "repo:git@github.com:org/repo.git",
        "repo:https://github.com/org/repo/",
    ):
        normalized = messages.normalize_scope(raw)
        assert messages.normalize_scope(normalized) == normalized


def test_scope_strips_trailing_slash_and_git_suffix():
    assert messages.normalize_scope("repo:https://github.com/org/repo/") == (
        "repo:github.com/org/repo"
    )
    assert messages.normalize_scope("repo:https://github.com/org/repo.git") == (
        "repo:github.com/org/repo"
    )


def test_scope_accepts_ssh_origin_and_normalizes_to_canonical():
    assert messages.normalize_scope("repo:git@github.com:Org/Repo.git") == (
        "repo:github.com/Org/Repo"
    )


def test_scope_rejects_repo_origin_with_credentials():
    with pytest.raises(ValueError):
        messages.normalize_scope("repo:https://user:token@github.com/org/repo")
    with pytest.raises(ValueError):
        messages.normalize_scope("repo:user@github.com:org/repo")


def test_scope_rejects_non_url_repo_origin():
    for bad in ("repo:/home/user/checkout", "repo:not a url"):
        with pytest.raises(ValueError):
            messages.normalize_scope(bad)


def test_scope_accepts_explicit_project_form_and_lowercases():
    assert messages.normalize_scope("project:Acme/Widget-Pro") == "project:acme/widget-pro"


def test_scope_rejects_malformed_project_segments():
    for bad in (
        "project:org",
        "project:org/",
        "project:/repo",
        "project:org/repo/extra",
        "project: org/repo",
    ):
        with pytest.raises(ValueError):
            messages.normalize_scope(bad)


def test_scope_rejects_cwd_and_absolute_path_forms():
    for bad in ("/home/user/project", ".", "./relative", "C:\\work\\repo", "my-scope"):
        with pytest.raises(ValueError):
            messages.normalize_scope(bad)


def test_scope_rejects_non_string_and_blank():
    for bad in (None, "", "   "):
        with pytest.raises(ValueError):
            messages.normalize_scope(bad)


# ---- refs --------------------------------------------------------------------


def test_refs_accept_absolute_https_urls():
    refs = ["https://github.com/org/repo/pull/1", "https://example.com/notes?a=b"]
    assert messages.validate_refs(refs) == refs


def test_refs_none_passes_through():
    assert messages.validate_refs(None) is None


def test_refs_reject_more_than_ten():
    with pytest.raises(ValueError):
        messages.validate_refs([f"https://example.com/{i}" for i in range(11)])


def test_refs_reject_userinfo():
    with pytest.raises(ValueError):
        messages.validate_refs(["https://user:token@example.com/x"])


def test_refs_reject_localhost():
    for bad in ("https://localhost/x", "https://api.localhost/x"):
        with pytest.raises(ValueError):
            messages.validate_refs([bad])


def test_refs_reject_loopback_private_and_link_local_ip_literals():
    for bad in (
        "https://127.0.0.1/x",
        "https://[::1]/x",
        "https://10.1.2.3/x",
        "https://192.168.0.5/x",
        "https://172.16.4.4/x",
        "https://169.254.9.9/x",
        "https://[fe80::1]/x",
    ):
        with pytest.raises(ValueError):
            messages.validate_refs([bad])


def test_refs_reject_non_https_schemes_file_urls_and_local_paths():
    for bad in (
        "file:///etc/passwd",
        "http://example.com/x",
        "ftp://example.com/x",
        "/etc/passwd",
        "~/notes.md",
        "javascript:alert(1)",
    ):
        with pytest.raises(ValueError):
            messages.validate_refs([bad])


def test_refs_reject_non_list_and_non_string_entries():
    with pytest.raises(ValueError):
        messages.validate_refs("https://example.com")
    with pytest.raises(ValueError):
        messages.validate_refs([42])


def test_refs_are_not_fetched(monkeypatch):
    # The validator is pure: patching the process-wide http client factory proves
    # no request leaves the process during validation.
    def _explode(*args, **kwargs):
        raise AssertionError("validate_refs must not issue network requests")

    monkeypatch.setattr("httpx.AsyncClient", _explode, raising=False)
    monkeypatch.setattr("httpx.Client", _explode, raising=False)
    messages.validate_refs(["https://example.com/never-fetched"])
    with pytest.raises(ValueError):
        messages.validate_refs(["https://10.0.0.9/never-fetched"])


# ---- verification --------------------------------------------------------------


def test_verification_accepts_exact_shape():
    verification = {"command": "uv run pytest", "status": "passed", "result": "12 green"}
    assert messages.validate_verification(verification) == verification


def test_verification_rejects_extra_or_missing_keys():
    with pytest.raises(ValueError):
        messages.validate_verification(
            {"command": "c", "status": "passed", "result": "r", "extra": "x"}
        )
    with pytest.raises(ValueError):
        messages.validate_verification({"command": "c", "status": "passed"})


def test_verification_rejects_unknown_status():
    with pytest.raises(ValueError):
        messages.validate_verification({"command": "c", "status": "skipped", "result": "r"})


def test_verification_rejects_blank_fields_and_non_dict():
    with pytest.raises(ValueError):
        messages.validate_verification({"command": " ", "status": "passed", "result": "r"})
    with pytest.raises(ValueError):
        messages.validate_verification("passed")


def test_verification_none_passes_through():
    assert messages.validate_verification(None) is None


# ---- rendering -----------------------------------------------------------------


def test_render_produces_canonical_markdown():
    content = messages.render_content(
        "Fix login flow",
        "in_progress",
        "Token refresh fails on expired sessions.",
        "Rotate the refresh secret.",
        {"command": "uv run pytest tests/auth", "status": "passed", "result": "12 green"},
        ["https://github.com/org/repo/pull/1"],
    )
    assert content == (
        "# Fix login flow\n"
        "\n"
        "## Status\n"
        "\n"
        "> in_progress\n"
        "\n"
        "## Result\n"
        "\n"
        "> Token refresh fails on expired sessions.\n"
        "\n"
        "## Next\n"
        "\n"
        "> Rotate the refresh secret.\n"
        "\n"
        "## Verification\n"
        "\n"
        "> Status: passed\n"
        "> Command: uv run pytest tests/auth\n"
        "> Result: 12 green\n"
        "\n"
        "## References\n"
        "\n"
        "- https://github.com/org/repo/pull/1"
    )


def test_render_verification_uses_labeled_status_command_result_lines():
    content = messages.render_content(
        "S",
        "blocked",
        "r",
        None,
        {
            "command": "uv run pytest tests/auth\n-v",
            "status": "failed",
            "result": "3 failed\nsee log",
        },
        None,
    )
    lines = content.splitlines()
    assert "> Status: failed" in lines
    assert "> Command: uv run pytest tests/auth" in lines
    assert "> -v" in lines
    assert "> Result: 3 failed" in lines
    assert "> see log" in lines


def test_render_references_are_markdown_list_items():
    content = messages.render_content(
        "S",
        "info",
        "r",
        None,
        None,
        ["https://github.com/org/repo/pull/1", "https://example.com/notes"],
    )
    lines = content.splitlines()
    assert "- https://github.com/org/repo/pull/1" in lines
    assert "- https://example.com/notes" in lines
    assert "## References" in lines


def test_render_omits_absent_optional_sections():
    content = messages.render_content("Subject", "info", "The fact.", None, None, None)
    assert content == "# Subject\n\n## Status\n\n> info\n\n## Result\n\n> The fact."


def test_render_blockquotes_every_line_of_user_controlled_scalars():
    content = messages.render_content(
        "Subject",
        "in_progress",
        "first line\n## Fake heading\n> fake quote",
        "next line one\nnext line two",
        None,
        None,
    )
    lines = content.splitlines()
    assert lines[0] == "# Subject"
    # Every line of the injected result is quoted, so no fake heading can escape.
    assert "> ## Fake heading" in lines
    assert "> > fake quote" in lines
    assert "> next line one" in lines
    assert "> next line two" in lines
    # The only unquoted section markers are the canonical ones.
    assert [line for line in lines if line.startswith("#")] == [
        "# Subject",
        "## Status",
        "## Result",
        "## Next",
    ]


def test_render_blockquotes_blank_lines_inside_scalars():
    content = messages.render_content("S", "info", "a\n\nb", None, None, None)
    assert "> \n" not in content
    assert "> a\n>\n> b" in content


def test_render_rejects_content_over_4kib_instead_of_truncating():
    with pytest.raises(ValueError):
        messages.render_content("S", "info", "x" * 5000, None, None, None)


# ---- TTL / expires_at -----------------------------------------------------------


def test_default_ttl_comes_from_message_ttl_days(monkeypatch):
    monkeypatch.setattr(messages, "MESSAGE_TTL_DAYS", 7)
    expires = messages.resolve_expires_at(None)
    assert timedelta(days=6.9) < expires - datetime.now(timezone.utc) <= timedelta(days=7.1)


def test_expires_at_accepts_future_iso_string():
    soon = datetime.now(timezone.utc) + timedelta(days=2)
    assert messages.resolve_expires_at(soon.isoformat()) == soon


def test_expires_at_must_be_in_the_future():
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    with pytest.raises(ValueError):
        messages.resolve_expires_at(past.isoformat())


def test_expires_at_may_be_at_most_thirty_days_out():
    limit = datetime.now(timezone.utc) + timedelta(days=30, minutes=5)
    with pytest.raises(ValueError):
        messages.resolve_expires_at(limit.isoformat())
    ok = datetime.now(timezone.utc) + timedelta(days=29)
    assert messages.resolve_expires_at(ok.isoformat()) == ok


def test_expires_at_rejects_unparseable_and_non_string():
    with pytest.raises(ValueError):
        messages.resolve_expires_at("not a date")
    with pytest.raises(ValueError):
        messages.resolve_expires_at(12345)


# ---- idempotency key ------------------------------------------------------------


def test_idempotency_key_max_128_chars():
    assert messages.validate_idempotency_key("  run-1 ") == "run-1"
    assert messages.validate_idempotency_key("x" * 128) == "x" * 128
    with pytest.raises(ValueError):
        messages.validate_idempotency_key("x" * 129)
    with pytest.raises(ValueError):
        messages.validate_idempotency_key("   ")
    assert messages.validate_idempotency_key(None) is None


# ---- purpose / status derivation ------------------------------------------------


def test_message_without_scope_requires_info_status():
    with pytest.raises(ValueError):
        messages.derive_purpose_and_check_status(None, "in_progress")
    assert messages.derive_purpose_and_check_status(None, "info") == "message"


def test_handoff_scope_requires_handoff_status():
    assert messages.derive_purpose_and_check_status("repo:github.com/o/r", "blocked") == "handoff"
    for bad in ("info", "done"):
        with pytest.raises(ValueError):
            messages.derive_purpose_and_check_status("repo:github.com/o/r", bad)


# ---- public row shape -------------------------------------------------------------


def _stored_row(**overrides):
    now = datetime.now(timezone.utc)
    row = {
        "id": uuid.uuid4(),
        "namespace": "default",
        "purpose": "message",
        "scope": None,
        "subject": "S",
        "subject_key": "s",
        "status": "info",
        "content": "# S",
        "author": "claude-code",
        "sender_key": "sender-key-hash",
        "idempotency_key": "run-1",
        "created_at": now,
        "claimed_at": None,
        "cancelled_at": None,
        "superseded_at": None,
        "expires_at": now + timedelta(days=1),
    }
    row.update(overrides)
    return row


def test_public_row_exposes_only_the_contracted_fields():
    public = messages.public_row(_stored_row())
    assert set(public) == {
        "id",
        "namespace",
        "purpose",
        "scope",
        "subject",
        "status",
        "delivery",
        "author",
        "created_at",
        "expires_at",
        "content",
    }
    row = _stored_row()
    public = messages.public_row(row)
    assert public["id"] == str(row["id"])
    assert public["created_at"] == row["created_at"].isoformat()
    assert public["expires_at"] == row["expires_at"].isoformat()


@pytest.mark.parametrize(
    ("overrides", "delivery"),
    [
        ({}, "pending"),
        ({"claimed_at": datetime.now(timezone.utc)}, "claimed"),
        ({"cancelled_at": datetime.now(timezone.utc)}, "cancelled"),
        ({"superseded_at": datetime.now(timezone.utc)}, "superseded"),
        ({"expires_at": datetime.now(timezone.utc) - timedelta(minutes=1)}, "expired"),
        # A delivered row stays delivered even once its expiry has passed.
        (
            {
                "claimed_at": datetime.now(timezone.utc) - timedelta(days=2),
                "expires_at": datetime.now(timezone.utc) - timedelta(minutes=1),
            },
            "claimed",
        ),
    ],
)
def test_delivery_is_derived_and_status_stays_the_report_status(overrides, delivery):
    public = messages.public_row(_stored_row(**overrides))
    assert public["status"] == "info"
    assert public["delivery"] == delivery


# ---- next rules -----------------------------------------------------------------


def test_next_rules_by_purpose_and_status():
    messages.check_next("message", "info", None)
    messages.check_next("message", "info", "optional next")
    messages.check_next("handoff", "in_progress", "must be nonblank")
    messages.check_next("handoff", "blocked", "must be nonblank")
    messages.check_next("handoff", "completed", None)
    with pytest.raises(ValueError):
        messages.check_next("handoff", "in_progress", "   ")
    with pytest.raises(ValueError):
        messages.check_next("handoff", "in_progress", None)
    with pytest.raises(ValueError):
        messages.check_next("handoff", "completed", "completed carries no next")
    with pytest.raises(ValueError):
        messages.check_next("message", "info", 42)
