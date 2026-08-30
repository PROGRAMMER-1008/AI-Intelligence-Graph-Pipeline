"""
Tests for AsyncHttpClient's certificate-fallback mechanism
(src/utils/http_client.py). Added after a real user-reported bug:
on some Windows Python installs, aiohttp's default SSL resolution
fails with `ClientConnectorCertificateError: unable to get local
issuer certificate` even though the target site's certificate is
valid (curl/requests succeed against the same host). The fix
(_rebuild_session_with_certifi_fallback) rebuilds the session with an
explicit certifi CA bundle ONLY after a real certificate error is
observed — not proactively — because forcing an explicit certifi-only
context unconditionally was tried first and found to break a
*different* environment (a sandboxed/proxied one where the default
aiohttp resolution succeeds but a certifi-only context fails with
"self-signed certificate in certificate chain", implying that
environment's egress path presents a certificate whose root isn't in
certifi's bundle but is trusted at the OS level).

These tests use mocks rather than live network calls specifically so
they pass regardless of which of the two failure modes (or neither)
the CI/test-runner's own environment happens to exhibit.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.http_client import AsyncHttpClient, RateLimitExceededError, RetriesExhaustedError


def test_fallback_not_applied_by_default():
    """The client must NOT proactively force a certifi-only SSL
    context — only after a real certificate failure is observed."""

    async def run():
        client = AsyncHttpClient()
        assert client._certifi_fallback_applied is False

    asyncio.run(run())


def test_rebuild_sets_flag_and_swaps_session():
    async def run():
        async with AsyncHttpClient() as client:
            original_session = client._session
            await client._rebuild_session_with_certifi_fallback()
            assert client._certifi_fallback_applied is True
            assert client._session is not original_session

    asyncio.run(run())


def test_rebuild_is_idempotent():
    """Multiple concurrent requests could hit a certificate error at
    roughly the same time; rebuilding the session twice must not
    double-swap or error."""

    async def run():
        async with AsyncHttpClient() as client:
            await client._rebuild_session_with_certifi_fallback()
            session_after_first = client._session
            await client._rebuild_session_with_certifi_fallback()
            session_after_second = client._session
            assert session_after_first is session_after_second

    asyncio.run(run())


def test_retries_exhausted_error_is_distinct_from_rate_limit_error():
    """
    Regression test for the exact bug reported by a user: a
    persistent ClientConnectorCertificateError (nothing to do with
    rate limiting) was being reported as 'RateLimitExceededError:
    429 persisted after 6 attempts', which sent debugging effort in
    completely the wrong direction. RetriesExhaustedError must be a
    distinct exception type carrying the real underlying exception,
    and must never claim a 429 occurred when one didn't.
    """
    err = RetriesExhaustedError("https://example.test", 6, ValueError("some real cause"))
    assert not isinstance(err, RateLimitExceededError)
    assert "429" not in str(err)
    assert "ValueError" in str(err)
    assert "some real cause" in str(err)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
