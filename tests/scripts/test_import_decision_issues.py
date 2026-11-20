"""scripts/import_decision_issues.py against a fake gh runner and a fake REST layer.

No test here reaches gh or a memory-base server: both are fakes inside the process.
"""

from __future__ import annotations

import json
import re
import urllib.parse

import pytest

import import_decision_issues as imi

BASE = "http://memory.test"

ISSUES = [
    {
        "number": 1,
        "title": "Keep decisions in their own lane",
        "closedAt": "2026-01-05T10:00:00Z",
        "url": "https://github.com/acme/app/issues/1",
    },
    {
        "number": 2,
        "title": "Rerank before injection",
        "closedAt": "2026-01-06T10:00:00Z",
        "url": "https://github.com/acme/app/issues/2",
    },
    {
        "number": 3,
        "title": "One scope per repository",
        "closedAt": "2026-01-07T10:00:00Z",
        "url": "https://github.com/acme/app/issues/3",
    },
]

COMMENT_PAGES = {
    1: [[{"body": "early thought"}, {"body": "## Decision\nUse pgvector."}]],
    2: [[{"body": "first comment"}], [{"body": "second comment"}]],
    3: [[{"body": "## Decision\n" + "x" * 1600}]],
}


def _issue_of(args):
    """The issue number a gh comments call names."""
    return int(re.search(r"/issues/(\d+)/comments$", args[-1]).group(1))


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_BASE_API_KEY", "test-key")
    monkeypatch.setenv("MEMORY_BASE_AUTHOR", "claude-code")
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path))
    monkeypatch.delenv("CLAUDE_PLUGIN_AUTHOR", raising=False)
    (tmp_path / "status.json").write_text(json.dumps({"author": "claude-code"}))
    return tmp_path


@pytest.fixture
def gh_calls(monkeypatch):
    """The fake gh runner; the fixture value is the list of argv lines it saw."""
    calls = []

    def run_gh(args):
        calls.append(args)
        if args[0] == "issue":
            # gh returns newest first; the script must order ascending itself.
            return json.dumps(list(reversed(ISSUES)))
        return json.dumps(COMMENT_PAGES[_issue_of(args)])

    monkeypatch.setattr(imi, "run_gh", run_gh)
    return calls


class FakeRest:
    """The fake REST layer: records (method, url, payload) per call."""

    def __init__(self, monkeypatch):
        self.calls = []
        self.existing = []
        self.links = {}
        self.next_number = 0
        self.reply = None
        monkeypatch.setattr(imi, "rest", self)

    def __call__(self, url, *, api_key, author, payload=None):
        self.calls.append(("POST" if payload is not None else "GET", url, payload))
        if payload is not None:
            self.next_number += 1
            return self.reply or {"number": self.next_number, "id": "id", "superseded": None}
        if "/decisions?" in url:
            return self.existing
        number = int(url.split("/decisions/", 1)[1].split("?", 1)[0])
        return {"number": number, "links": self.links.get(number, [])}

    def posts(self):
        return [payload for method, _, payload in self.calls if method == "POST"]

    def urls(self, method):
        return [url for call_method, url, _ in self.calls if call_method == method]


@pytest.fixture
def rest(monkeypatch):
    return FakeRest(monkeypatch)


# ---- closing rule -------------------------------------------------------------


def test_closing_rule_takes_the_last_heading_comment_and_every_later_comment():
    comments = [
        {"body": "early"},
        {"body": "## Decision\nUse pgvector."},
        {"body": "Amendment: also index tags."},
        {"body": "Amendment 2."},
    ]
    assert imi.closing_text(comments, "u") == (
        "## Decision\nUse pgvector.\n\nAmendment: also index tags.\n\nAmendment 2."
    )


def test_closing_rule_takes_the_last_heading_comment_else_all_bodies():
    comments = [
        {"body": "## Decision\nUse pgvector."},
        {"body": "chatter without a heading"},
        {"body": "# decision\nThe final call."},
    ]
    assert imi.closing_text(comments, "u") == "# decision\nThe final call."

    fallback = [{"body": "first"}, {"body": "second"}]
    assert imi.closing_text(fallback, "u") == "first\n\nsecond"

    assert imi.closing_text([], "u") == ""


def test_cut_text_fits_1500_chars_with_the_suffix():
    url = "https://github.com/acme/app/issues/3"
    body = "## Decision\n" + "x" * 1600
    cut = imi.closing_text([{"body": body}], url)
    suffix = f" [cut; full text: {url}]"
    assert len(cut) == 1500
    assert cut.endswith(suffix)
    assert cut[: -len(suffix)] == body[: 1500 - len(suffix)]


def test_text_at_the_limit_is_not_cut():
    assert imi.closing_text([{"body": "d" * 1500}], "u") == "d" * 1500


# ---- run ----------------------------------------------------------------------


def test_records_issues_in_ascending_number_order(env, gh_calls, rest, capsys):
    assert imi.main(["--repo", "acme/app", "--url", BASE]) == 0
    assert [payload["links"][0] for payload in rest.posts()] == [i["url"] for i in ISSUES]
    assert rest.posts()[0] == {
        "scope": "repo:github.com/acme/app",
        "title": ISSUES[0]["title"],
        "decision": "## Decision\nUse pgvector.",
        "links": [ISSUES[0]["url"]],
        "decided_on": "2026-01-05",
    }
    out = capsys.readouterr().out
    assert "#1 -> D-1" in out
    assert "#2 -> D-2" in out
    assert "#3 -> D-3" in out
    assert out.rstrip().endswith("recorded 3, skipped 0")


def test_skips_issues_already_imported(env, gh_calls, rest, capsys):
    rest.existing = [{"number": 1, "title": "t", "decided_on": "2026-01-06", "status": "active"}]
    rest.links[1] = [ISSUES[1]["url"]]
    assert imi.main(["--repo", "acme/app", "--url", BASE]) == 0
    assert [payload["links"][0] for payload in rest.posts()] == [
        ISSUES[0]["url"],
        ISSUES[2]["url"],
    ]
    assert "#2 skipped (already imported)" in capsys.readouterr().out


def test_author_differing_from_the_plugins_status_refuses_before_any_request(
    env, gh_calls, rest, monkeypatch
):
    monkeypatch.setenv("MEMORY_BASE_AUTHOR", "other-persona")
    assert imi.main(["--repo", "acme/app", "--url", BASE]) == 2
    assert gh_calls == []
    assert rest.calls == []


@pytest.mark.parametrize(
    "status", [None, "not json", "{}"], ids=["missing", "garbage", "no-author"]
)
def test_an_unreadable_plugin_status_refuses_before_any_request(env, gh_calls, rest, status):
    (env / "status.json").unlink()
    if status is not None:
        (env / "status.json").write_text(status)
    assert imi.main(["--repo", "acme/app", "--url", BASE]) == 2
    assert gh_calls == []
    assert rest.calls == []


def test_plugin_data_option_names_the_status_directory(env, gh_calls, rest, monkeypatch, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "status.json").write_text(json.dumps({"author": "claude-code"}))
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path / "missing"))
    assert imi.main(["--repo", "acme/app", "--url", BASE, "--plugin-data", str(other)]) == 0


def test_a_full_decision_list_refuses_because_the_resume_check_would_miss_rows(env, gh_calls, rest):
    rest.existing = [
        {"number": n, "title": "t", "decided_on": "2026-01-06", "status": "active"}
        for n in range(200, 0, -1)
    ]
    assert imi.main(["--repo", "acme/app", "--url", BASE]) == 2
    assert rest.posts() == []


@pytest.mark.parametrize("existing", [{"detail": "x"}, [{"title": "no number"}]])
def test_a_malformed_list_reply_stops_the_run(env, gh_calls, rest, existing):
    rest.existing = existing
    assert imi.main(["--repo", "acme/app", "--url", BASE]) == 1
    assert rest.posts() == []


def test_a_record_reply_without_a_number_stops_the_run(env, gh_calls, rest):
    rest.reply = {"unexpected": True}
    assert imi.main(["--repo", "acme/app", "--url", BASE]) == 1
    assert len(rest.posts()) == 1


def test_gh_failure_on_comments_stops_before_writing_that_issue(env, rest, monkeypatch):
    def run_gh(args):
        if args[0] == "issue":
            return json.dumps(ISSUES)
        if _issue_of(args) == 2:
            raise imi.GhError("boom")
        return json.dumps(COMMENT_PAGES[_issue_of(args)])

    monkeypatch.setattr(imi, "run_gh", run_gh)
    assert imi.main(["--repo", "acme/app", "--url", BASE]) == 1
    assert [payload["links"][0] for payload in rest.posts()] == [ISSUES[0]["url"]]


def test_gh_failure_on_the_issue_list_writes_nothing(env, rest, monkeypatch):
    def run_gh(args):
        if args[0] == "issue":
            raise imi.GhError("boom")
        raise AssertionError("comments must not be fetched when the list fails")

    monkeypatch.setattr(imi, "run_gh", run_gh)
    assert imi.main(["--repo", "acme/app", "--url", BASE]) == 1
    assert rest.posts() == []


def test_dry_run_sends_no_post(env, gh_calls, rest, capsys):
    assert imi.main(["--repo", "acme/app", "--url", BASE, "--dry-run"]) == 0
    assert rest.posts() == []
    query = urllib.parse.urlencode(
        {"scope": "repo:github.com/acme/app", "status": "all", "limit": 200}
    )
    assert rest.urls("GET") == [f"{BASE}/decisions?{query}"]
    out = capsys.readouterr().out
    assert '#1 (dry run) "Keep decisions in their own lane": 26 chars' in out
    assert "Use pgvector" not in out
    assert "would record 3, skipped 0" in out
