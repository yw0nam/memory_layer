"""Deterministic detection of fixed-format credentials and keyword-stated short codes."""

from __future__ import annotations

import re
from collections.abc import Generator

from detect_secrets.plugins.aws import AWSKeyDetector
from detect_secrets.plugins.base import RegexBasedDetector
from detect_secrets.plugins.basic_auth import BasicAuthDetector
from detect_secrets.plugins.github_token import GitHubTokenDetector
from detect_secrets.plugins.gitlab_token import GitLabTokenDetector
from detect_secrets.plugins.jwt import JwtTokenDetector
from detect_secrets.plugins.npm import NpmDetector
from detect_secrets.plugins.openai import OpenAIDetector
from detect_secrets.plugins.private_key import PrivateKeyDetector
from detect_secrets.plugins.pypi_token import PypiTokenDetector
from detect_secrets.plugins.sendgrid import SendGridDetector
from detect_secrets.plugins.slack import SlackDetector
from detect_secrets.plugins.stripe import StripeDetector

_TOKEN = "A-Za-z0-9_-"


class AWSAccessKeyIdDetector(AWSKeyDetector):
    """The access-key-id shape alone, without the keyword-context secret pattern."""

    denylist = (AWSKeyDetector.denylist[0],)


class AnchoredJwtDetector(JwtTokenDetector):
    """The JWT shape, started only at a token boundary so scanning stays linear."""

    # A header carrying the required "alg" member encodes to at least 14 characters.
    denylist = (
        re.compile(rf"(?<![{_TOKEN}])eyJ[{_TOKEN}]{{11,}}\.[A-Za-z0-9_=-]+\.?[A-Za-z0-9-_.+/=]*?"),
    )


class OpenAILegacyKeyDetector(OpenAIDetector):
    """The legacy OpenAI key shape, started only at a token boundary so scanning stays linear."""

    denylist = (
        re.compile(rf"(?<![{_TOKEN}])sk-[{_TOKEN}]*[A-Za-z0-9]{{20}}T3BlbkFJ[A-Za-z0-9]{{20}}"),
    )


class NpmAuthTokenDetector(NpmDetector):
    """An npmrc auth-token line, one start per whitespace- or quote-delimited run."""

    denylist = (
        re.compile(
            r"(?<![^\s`'\"])//[^\s`'\"]+/:_authToken=\s*(?:npm_[A-Za-z0-9_-]+|[A-Fa-f0-9-]{36})"
        ),
    )


class ModernOpenAIKeyDetector(RegexBasedDetector):
    secret_type = "OpenAI API Key"
    denylist = (
        re.compile(
            rf"(?<![{_TOKEN}])sk-(?!ant-)(?:proj-|svcacct-)?[{_TOKEN}]{{32,}}(?![{_TOKEN}])"
        ),
    )


class AnthropicKeyDetector(RegexBasedDetector):
    secret_type = "Anthropic API Key"
    denylist = (re.compile(rf"(?<![{_TOKEN}])sk-ant-[a-z0-9]+-[{_TOKEN}]{{20,}}(?![{_TOKEN}])"),)


class GoogleApiKeyDetector(RegexBasedDetector):
    secret_type = "Google API Key"
    denylist = (re.compile(rf"(?<![{_TOKEN}])AIza[{_TOKEN}]{{35}}(?![{_TOKEN}])"),)


_SENTENCE = re.compile(r"[^.;\n!?]+")
_CODE_NOISE = re.compile(
    r"(?i)\b(?:status|exit|error|http|response|zip|postal|country|area|return|result) code\b"
)
_CODE_KEYWORD = re.compile(
    rf"(?i)(?<![{_TOKEN}])(?:pin|passcode|password|passphrase|combination|otp"
    r"|(?:one-time|two-factor|verification|pick-?up|locker|access|door|gate|security|2fa|mfa"
    r"|login|auth|confirmation|unlock|lock|entry|parcel|bank)[ -]code)"
    rf"(?![{_TOKEN}])"
)
# 3-8 digits that are not a date, year, measured quantity, version, path, or phone number.
_CODE = re.compile(
    r"(?i)(?<![A-Za-z0-9_.+$#%/:-])(?<!\d[- ])(?!\d{4}-\d{2}-\d{2})(?!(?:19|20)\d{2}(?![- ]?\d))"
    r"\d(?:[- ]?\d){2,7}"
    r"(?![A-Za-z0-9_%]|[.:/-]\d|-?(?:second|minute|hour|day|week|month|year|ms|km|kg|ml|cm|mm|px)s?\b)"
)


class ShortCodeDetector(RegexBasedDetector):
    """A digit code stated in the same sentence as the keyword that names it."""

    secret_type = "Short Code"
    denylist = ()

    def analyze_string(self, string: str) -> Generator[str, None, None]:
        for sentence in _SENTENCE.findall(string):
            if (
                not _CODE_NOISE.search(sentence)
                and _CODE_KEYWORD.search(sentence)
                and _CODE.search(sentence)
            ):
                yield self.secret_type


DETECTORS: tuple[RegexBasedDetector, ...] = (
    AWSAccessKeyIdDetector(),
    BasicAuthDetector(),
    GitHubTokenDetector(),
    GitLabTokenDetector(),
    AnchoredJwtDetector(),
    OpenAILegacyKeyDetector(),
    PrivateKeyDetector(),
    SlackDetector(),
    StripeDetector(),
    SendGridDetector(),
    NpmAuthTokenDetector(),
    PypiTokenDetector(),
    ModernOpenAIKeyDetector(),
    AnthropicKeyDetector(),
    GoogleApiKeyDetector(),
    ShortCodeDetector(),
)


def find_secret(text: str) -> str | None:
    """Return the type of the first credential found in text, never the credential itself."""
    for detector in DETECTORS:
        if next(detector.analyze_string(text), None) is not None:
            return detector.secret_type
    return None
