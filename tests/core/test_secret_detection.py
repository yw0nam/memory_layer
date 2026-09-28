"""find_secret names the type of a fixed-format credential and ignores look-alikes."""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid

import pytest

from memory_base.core.secrets import DETECTORS, find_secret


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _jwt(payload: bytes) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    return f"{header}.{_b64(payload)}.{'s' * 43}"


# (secret_type, positive, near miss); every value is assembled at runtime.
CASES = [
    ("AWS Access Key", "AKIA" + "A" * 16, "AKIA" + "A" * 15),
    ("Basic Auth Credentials", "postgres://user:" + "p" * 8 + "@db.internal/app", "postgres://user@db.internal/app"),
    ("GitHub Token", "ghp_" + "a" * 36, "ghp_" + "a" * 35),
    ("GitLab Token", "glpat-" + "a" * 20, "glpat-" + "a" * 19),
    ("GitLab Token", "glcbt-" + "a" * 25, "glcbt-" + "a" * 19),
    ("JSON Web Token", _jwt(json.dumps({"sub": "agent"}).encode()), _jwt(b"plain words, not json")),
    ("OpenAI Token", "sk-" + "a" * 20 + "T3BlbkFJ" + "a" * 20, "sk-" + "a" * 10 + "T3BlbkFJ" + "a" * 10),
    ("Private Key", "-----BEGIN " + "RSA PRIVATE KEY-----", "-----BEGIN " + "PUBLIC KEY-----"),
    ("Slack Token", "xoxb-" + "1" * 10 + "-" + "a" * 12, "xoxb-" + "a" * 12),
    ("Stripe Access Key", "sk_live_" + "a" * 24, "sk_test_" + "a" * 24),
    ("SendGrid API Key", "SG." + "a" * 22 + "." + "a" * 43, "SG." + "a" * 21 + "." + "a" * 43),
    ("NPM tokens", "//registry.npmjs.org/:_authToken=npm_" + "a" * 36, "//registry.npmjs.org/:_authToken=${NPM_TOKEN}"),
    ("PyPI Token", "pypi-AgEIcHlwaS5vcmc" + "a" * 70, "pypi-AgEIcHlwaS5vcmc" + "a" * 69),
    ("OpenAI API Key", "sk-proj-" + "a" * 40, "sk-" + "a" * 31),
    ("Anthropic API Key", "sk-ant-api03-" + "a" * 20, "sk-ant-api03-" + "a" * 19),
    ("Google API Key", "AIza" + "a" * 35, "AIza" + "a" * 34),
]  # fmt: skip


def test_the_detector_tuple_is_exactly_the_fixed_format_set():
    assert [type(detector).__name__ for detector in DETECTORS] == [
        "AWSAccessKeyIdDetector",
        "BasicAuthDetector",
        "GitHubTokenDetector",
        "GitLabTokenDetector",
        "AnchoredJwtDetector",
        "OpenAILegacyKeyDetector",
        "PrivateKeyDetector",
        "SlackDetector",
        "StripeDetector",
        "SendGridDetector",
        "NpmAuthTokenDetector",
        "PypiTokenDetector",
        "ModernOpenAIKeyDetector",
        "AnthropicKeyDetector",
        "GoogleApiKeyDetector",
    ]


@pytest.mark.parametrize(
    ("secret_type", "positive", "near_miss"), CASES, ids=[f"{c[0]}-{c[1][:5]}" for c in CASES]
)
def test_each_detector_names_its_type_and_ignores_its_near_miss(secret_type, positive, near_miss):
    assert find_secret(positive) == secret_type
    assert find_secret(f"deploy notes:\nthe value {positive} lives in the vault\n") == secret_type
    assert find_secret(near_miss) is None
    assert find_secret(f"deploy notes:\nthe value {near_miss} lives in the vault\n") is None


@pytest.mark.parametrize("suffix", ["", "-proj", "-svcacct"])
def test_modern_openai_key_prefixes_are_detected(suffix):
    assert find_secret(f"sk{suffix}-" + "a" * 32) == "OpenAI API Key"


def test_anthropic_key_is_not_reported_as_an_openai_key():
    assert find_secret("sk-ant-admin01-" + "a" * 40) == "Anthropic API Key"


def test_local_detectors_do_not_match_inside_a_longer_token():
    assert find_secret("task-" + "a" * 40) is None
    assert find_secret("xAIza" + "a" * 35) is None
    assert find_secret("AIza" + "a" * 36) is None


@pytest.mark.parametrize(
    "text",
    [
        hashlib.sha1(b"commit").hexdigest(),
        "note:" + hashlib.sha256(b"note").hexdigest()[:16],
        str(uuid.UUID(int=0x1234_5678_9ABC_DEF0_1234_5678_9ABC_DEF0)),
        hashlib.sha256(b"content").hexdigest(),
        "postgres://db.internal:5432/app",
        "set password= in .env and restart the service",
        "prefer ruff for linting; the CI job runs ruff check and ruff format --check",
    ],
)
def test_ordinary_note_content_is_not_a_credential(text):
    assert find_secret(text) is None


_UUID_SHAPE = "-".join("0" * width for width in (8, 4, 4, 4, 12))
_JWT = _jwt(json.dumps({"sub": "agent"}).encode())


@pytest.mark.parametrize(
    ("text", "secret_type"),
    [
        ("//npm.pkg.github.com/acme/:_authToken=npm_" + "a" * 36, "NPM tokens"),
        ("//registry.npmjs.org/:_authToken=" + _UUID_SHAPE, "NPM tokens"),
        ("//localhost:4873/:_authToken=npm_" + "a" * 36, "NPM tokens"),
        ("add `//registry.npmjs.org/:_authToken=npm_" + "a" * 36 + "` to .npmrc", "NPM tokens"),
        (f"token={_JWT}", "JSON Web Token"),
        (f"https://app.example/callback?jwt={_JWT}", "JSON Web Token"),
        ("sk-proj-" + "a" * 20 + "T3BlbkFJ" + "a" * 20, "OpenAI Token"),
    ],
)
def test_real_shapes_match_the_linear_detectors(text, secret_type):
    assert find_secret(text) == secret_type


# Each detector's own anchors, repeated to fill one line, are its worst case.
_ANCHORS = {
    "AWSAccessKeyIdDetector": ["AKIA"],
    "BasicAuthDetector": ["://a:b@", "://a:", "://"],
    "GitHubTokenDetector": ["ghp_"],
    "GitLabTokenDetector": ["glpat-", "glcbt-", "GR1348941"],
    "AnchoredJwtDetector": ["eyJ", "eyJa.", "eyJ" + "a" * 11 + ".a."],
    "OpenAILegacyKeyDetector": ["sk-", "sk-a", "aT3BlbkFJ"],
    "PrivateKeyDetector": ["-----BEGIN "],
    "SlackDetector": ["xox", "xoxb-", "xoxb-1-"],
    "StripeDetector": ["sk_live_"],
    "SendGridDetector": ["SG.", "SG.a."],
    "NpmAuthTokenDetector": ["//", "//a/", " //a/"],
    "PypiTokenDetector": ["pypi-", "pypi-AgEIcHlwaS5vcmc"],
    "ModernOpenAIKeyDetector": ["sk-", " sk-a"],
    "AnthropicKeyDetector": ["sk-ant-", "sk-ant-a-"],
    "GoogleApiKeyDetector": ["AIza"],
}
_LINE_CHARS = 2_000_000


def test_every_detector_has_worst_case_anchors():
    assert set(_ANCHORS) == {type(detector).__name__ for detector in DETECTORS}


@pytest.mark.parametrize("detector", DETECTORS, ids=lambda detector: type(detector).__name__)
def test_every_detector_scans_a_crafted_two_million_char_line_in_linear_time(detector):
    for anchor in [*_ANCHORS[type(detector).__name__], "[x](https://e.com/a) "]:
        line = (anchor * (_LINE_CHARS // len(anchor) + 1))[:_LINE_CHARS]
        started = time.perf_counter()
        for _ in detector.analyze_string(line):
            pass
        assert time.perf_counter() - started < 1.0, anchor
