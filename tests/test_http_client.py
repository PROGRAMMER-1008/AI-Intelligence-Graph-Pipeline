"""
Tests for AsyncHttpClient's rate-limit detection.

Regression test for a real bug caught during development: GitHub's REST
API signals rate limiting via HTTP 403 + X-RateLimit-Remaining: 0,
NOT via 429. Verified against a live response on 2026-08-28 (see
docs/architecture.md, "Rate Limit Handling" section, for the raw
headers captured). A naive "only check for 429" implementation would
treat this as an unrecoverable auth failure and abandon the request
instead of backing off and retrying — this test locks in the fix.
"""

import time
import sys
from pathlib import Path
from unittest.mock import MagicMock

from multidict import CIMultiDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.http_client import AsyncHttpClient


def _mock_response(status: int, headers: dict) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.headers = CIMultiDict(headers)
    return resp


def test_github_style_403_is_detected_as_rate_limit():
    resp = _mock_response(
        403,
        {
            "x-ratelimit-limit": "60",
            "X-RateLimit-Remaining": "0",
            "x-ratelimit-used": "60",
            "x-ratelimit-reset": str(int(time.time()) + 45),
        },
    )
    assert AsyncHttpClient._is_rate_limited_403(resp) is True


def test_genuine_403_auth_failure_is_not_misclassified():
    resp = _mock_response(403, {})
    assert AsyncHttpClient._is_rate_limited_403(resp) is False


def test_403_with_nonzero_remaining_is_not_rate_limit():
    resp = _mock_response(403, {"X-RateLimit-Remaining": "12"})
    assert AsyncHttpClient._is_rate_limited_403(resp) is False


def test_retry_after_header_takes_priority_over_ratelimit_reset():
    resp = _mock_response(
        429,
        {"Retry-After": "5", "X-RateLimit-Reset": str(int(time.time()) + 999)},
    )
    delay = AsyncHttpClient._extract_retry_after_seconds(resp)
    assert delay == 5.0


def test_ratelimit_reset_used_when_retry_after_absent():
    reset_at = int(time.time()) + 45
    resp = _mock_response(403, {"x-ratelimit-reset": str(reset_at)})
    delay = AsyncHttpClient._extract_retry_after_seconds(resp)
    assert delay is not None
    assert 40 <= delay <= 46


def test_no_rate_limit_headers_returns_none():
    resp = _mock_response(500, {})
    assert AsyncHttpClient._extract_retry_after_seconds(resp) is None


def test_get_text_accepts_post_method_kwarg():
    """
    Regression test for a real bug caught during development: get_text
    and get_json hardcoded 'GET' as the first positional arg to
    _request(), so any caller passing method='POST' (as every LLM tier
    in orchestrator.py does) hit
    'got multiple values for argument method' and crashed before the
    request was ever sent. Caught via a live-ish integration check
    (verify_groq_live.py against a .env-loaded fake key) rather than
    a unit test originally — added here so it can't silently regress.
    This test only checks the method routes through correctly, not
    live network behavior.
    """
    import asyncio
    import inspect

    sig = inspect.signature(AsyncHttpClient.get_text)
    assert "method" in sig.parameters
    sig2 = inspect.signature(AsyncHttpClient.get_json)
    assert "method" in sig2.parameters

    # Confirm calling with method="POST" doesn't raise a TypeError at
    # the call-construction level (would raise before any network I/O
    # if the signature regressed).
    client = AsyncHttpClient()

    async def build_call():
        coro = client.get_text("https://example.invalid", method="POST", json={"a": 1})
        coro.close()  # never actually await / send it — just prove no TypeError on construction

    asyncio.run(build_call())


if __name__ == "__main__":
    # Lightweight runner so this can be executed without pytest installed,
    # since the sandbox network allowlist doesn't include pypi mirrors for
    # every dependency — see docs/architecture.md for the CI setup that
    # would run this via pytest in a real deployment.
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS: {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed")
