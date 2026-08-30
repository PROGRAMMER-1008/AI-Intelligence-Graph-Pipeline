"""
Resilient async HTTP client.

This is the single place in the codebase that talks to the network,
by design. Centralizing it means:
  1. Every scraper and LLM call gets the same 429/5xx backoff logic
     for free instead of reimplementing (and inevitably bungling) it.
  2. Concurrency is capped globally via a semaphore, not per-scraper,
     so we can't accidentally hammer a host from three coroutines
     that don't know about each other.
  3. It's the one place to add a proxy pool / different anti-bot
     strategy later without touching scraper code.

Retry policy:
  - 429 Too Many Requests -> respect Retry-After header if present,
    otherwise exponential backoff with full jitter (AWS-style),
    capped at max_backoff.
  - 5xx -> same backoff, treated as transient.
  - 413 Payload Too Large -> NOT retried here; this is a caller-side
    problem (the caller sent too much), so we raise a distinct
    PayloadTooLargeError so the LLM orchestrator can chunk and retry
    at a higher level. Retrying blindly at the transport layer would
    just get another 413.
  - Connection errors / timeouts -> retried with backoff, capped
    attempts, because these hosts can be flaky and a hard fail on
    the first timeout would be wasteful for a long-running crawl.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

logger = logging.getLogger("graphone.http")


class PayloadTooLargeError(Exception):
    """Raised on HTTP 413. Caller must shrink the payload and retry explicitly."""

    def __init__(self, url: str, payload_size: Optional[int] = None):
        self.url = url
        self.payload_size = payload_size
        super().__init__(f"413 Payload Too Large for {url} (size={payload_size})")


class RateLimitExceededError(Exception):
    """Raised when 429 retries are exhausted — surfaced so a caller can
    switch LLM tiers (fallback chain) rather than fail the whole record."""

    def __init__(self, url: str, attempts: int):
        self.url = url
        self.attempts = attempts
        super().__init__(f"429 persisted after {attempts} attempts for {url}")


class RetriesExhaustedError(Exception):
    """
    Raised when max_attempts is reached WITHOUT a 429/403-rate-limit
    ever being the cause — e.g. a persistent SSL/connection failure,
    DNS failure, or timeout. Distinguished from RateLimitExceededError
    because the fix is completely different: a real 429 means "wait
    longer or use a different provider"; a persistent connection/SSL
    failure means "something about this environment's network/cert
    setup is broken" and retrying harder will never help. Reporting
    the latter as "429 persisted" (the bug this class fixes) sends
    whoever's debugging it looking in exactly the wrong place — this
    was caught from a real user report where a Windows-side aiohttp/
    certifi certificate issue was misreported as arXiv rate-limiting.
    """

    def __init__(self, url: str, attempts: int, last_exception: Exception):
        self.url = url
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Failed after {attempts} attempts for {url} (not rate-limiting — "
            f"last error was {type(last_exception).__name__}: {last_exception})"
        )


@dataclass
class RetryConfig:
    max_attempts: int = 6
    base_delay: float = 1.0
    max_delay: float = 60.0
    # Full jitter per Marc Brooker / AWS Architecture Blog "Exponential
    # Backoff and Jitter": delay = random(0, min(max_delay, base * 2^attempt))
    # This avoids the thundering-herd retry synchronization that plain
    # exponential backoff (without jitter) produces across concurrent workers.


class AsyncHttpClient:
    def __init__(
        self,
        max_concurrency: int = 20,
        retry_config: Optional[RetryConfig] = None,
        default_headers: Optional[dict] = None,
        timeout_seconds: float = 30.0,
    ):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._retry = retry_config or RetryConfig()
        self._headers = default_headers or {
            "User-Agent": (
                "GraphOneBot/1.0 (+https://graphone.example/bot; "
                "Intelligence Graph research crawler; contact: data@graphone.example)"
            )
        }
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: Optional[aiohttp.ClientSession] = None
        self._certifi_fallback_applied = False

    async def __aenter__(self) -> "AsyncHttpClient":
        # SSL handling needs to satisfy two different real environments
        # that were both hit during development, which pull in
        # opposite directions:
        #   1. Some Windows Python installs: aiohttp's default SSL
        #      resolution fails with `ClientConnectorCertificateError:
        #      unable to get local issuer certificate` even though the
        #      target site's certificate is fine (confirmed: curl and
        #      `requests` succeed against the same host on the same
        #      machine). aiohttp doesn't use the OS certificate store
        #      the way `requests`/curl do by default.
        #   2. Some sandboxed/proxied environments: forcing an
        #      explicit `certifi`-only SSL context breaks connections
        #      that plain aiohttp (using its own default resolution,
        #      which DOES consult more sources depending on platform)
        #      handles fine — confirmed directly: default aiohttp
        #      succeeded against api.github.com in an environment
        #      where a certifi-only context failed with "self-signed
        #      certificate in certificate chain," implying that
        #      environment's egress path presents a certificate whose
        #      root isn't in certifi's bundle but IS trusted at the OS
        #      level.
        #
        # Resolution: try aiohttp's default (no explicit ssl= argument)
        # first — this is correct for environment (2) and for the
        # common case. If constructing a connection fails specifically
        # with a certificate error, fall back to an explicit
        # certifi-based context — this is the fix for environment (1).
        # Both paths are exercised by whichever environment needs them;
        # neither is silently assumed to be "the" right answer.
        self._session = aiohttp.ClientSession(headers=self._headers, timeout=self._timeout)
        return self

    async def _rebuild_session_with_certifi_fallback(self) -> None:
        """
        Called only after a certificate-verification failure using the
        default session. Rebuilds the session with an explicit
        certifi-based SSL context. Idempotent: safe to call multiple
        times (checks a flag) since several concurrent requests could
        hit the certificate error at roughly the same time.
        """
        if self._certifi_fallback_applied:
            return
        self._certifi_fallback_applied = True
        logger.warning(
            "Default SSL certificate verification failed — retrying with an "
            "explicit certifi CA bundle. This is a known fix for aiohttp on "
            "some Windows Python installs where the OS certificate store "
            "isn't consulted by default."
        )
        import certifi
        import ssl

        old_session = self._session
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=0)
        self._session = aiohttp.ClientSession(
            headers=self._headers, timeout=self._timeout, connector=connector
        )
        if old_session:
            await old_session.close()

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    def _compute_backoff(self, attempt: int, retry_after: Optional[float] = None) -> float:
        if retry_after is not None:
            return min(retry_after, self._retry.max_delay)
        exp = min(self._retry.max_delay, self._retry.base_delay * (2**attempt))
        return random.uniform(0, exp)  # full jitter

    async def get_json(self, url: str, method: str = "GET", **kwargs) -> Any:
        text, _ = await self._request(method, url, **kwargs)
        import json

        return json.loads(text)

    async def get_text(self, url: str, method: str = "GET", **kwargs) -> str:
        text, _ = await self._request(method, url, **kwargs)
        return text

    async def get_bytes(self, url: str, **kwargs) -> bytes:
        last_exception: Optional[Exception] = None
        hit_rate_limit = False
        async with self._semaphore:
            for attempt in range(self._retry.max_attempts):
                try:
                    async with self._session.get(url, **kwargs) as resp:
                        if resp.status == 429 or self._is_rate_limited_403(resp):
                            hit_rate_limit = True
                            await self._handle_429(url, resp, attempt)
                            continue
                        if resp.status == 413:
                            raise PayloadTooLargeError(url)
                        if resp.status >= 500:
                            await self._handle_5xx(url, resp.status, attempt)
                            continue
                        resp.raise_for_status()
                        return await resp.read()
                except aiohttp.ClientConnectorCertificateError as e:
                    last_exception = e
                    await self._rebuild_session_with_certifi_fallback()
                    # No sleep here: the fallback session swap IS the
                    # fix, not a transient condition to back off from —
                    # retry immediately with the new session.
                except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                    last_exception = e
                    await self._handle_transient(url, e, attempt)
            if hit_rate_limit:
                raise RateLimitExceededError(url, self._retry.max_attempts)
            raise RetriesExhaustedError(
                url, self._retry.max_attempts, last_exception or Exception("unknown failure")
            )

    @staticmethod
    def _is_rate_limited_403(resp: aiohttp.ClientResponse) -> bool:
        """
        GitHub's REST API (and some other APIs) return HTTP 403 — not 429 —
        for rate limiting, distinguished by the X-RateLimit-Remaining: 0
        header. Confirmed against a live 403 response during development:
        headers included x-ratelimit-limit, x-ratelimit-remaining: 0, and
        x-ratelimit-reset (unix timestamp) with no Retry-After header.
        Treating every 403 as a hard auth failure (the naive approach)
        would misclassify this as unrecoverable and abandon the fetch
        instead of backing off — a real bug that only shows up under
        actual rate-limit conditions, which is exactly why we handle it
        explicitly here rather than assuming 429 is the only rate-limit
        status code a real API will use.
        """
        if resp.status != 403:
            return False
        remaining = resp.headers.get("X-RateLimit-Remaining")
        return remaining == "0"

    @staticmethod
    def _extract_retry_after_seconds(resp: aiohttp.ClientResponse) -> Optional[float]:
        """
        Standard Retry-After header takes priority (seconds, per RFC).
        Falls back to GitHub-style X-RateLimit-Reset (unix epoch seconds
        of the reset moment) when Retry-After is absent — this is the
        header GitHub actually sends on rate-limited 403s, confirmed
        live: no Retry-After, but x-ratelimit-reset present.
        """
        retry_after_hdr = resp.headers.get("Retry-After")
        if retry_after_hdr:
            try:
                return float(retry_after_hdr)
            except ValueError:
                pass

        reset_hdr = resp.headers.get("X-RateLimit-Reset")
        if reset_hdr:
            try:
                import time

                reset_epoch = float(reset_hdr)
                return max(0.0, reset_epoch - time.time())
            except ValueError:
                pass

        return None

    async def _request(self, method: str, url: str, **kwargs) -> tuple[str, int]:
        last_exception: Optional[Exception] = None
        hit_rate_limit = False
        async with self._semaphore:
            for attempt in range(self._retry.max_attempts):
                try:
                    async with self._session.request(method, url, **kwargs) as resp:
                        if resp.status == 429 or self._is_rate_limited_403(resp):
                            hit_rate_limit = True
                            await self._handle_429(url, resp, attempt)
                            continue
                        if resp.status == 413:
                            raise PayloadTooLargeError(url)
                        if resp.status >= 500:
                            await self._handle_5xx(url, resp.status, attempt)
                            continue
                        resp.raise_for_status()
                        return await resp.text(), resp.status
                except aiohttp.ClientConnectorCertificateError as e:
                    last_exception = e
                    await self._rebuild_session_with_certifi_fallback()
                except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                    last_exception = e
                    await self._handle_transient(url, e, attempt)
            if hit_rate_limit:
                raise RateLimitExceededError(url, self._retry.max_attempts)
            raise RetriesExhaustedError(
                url, self._retry.max_attempts, last_exception or Exception("unknown failure")
            )

    async def _handle_429(self, url: str, resp: aiohttp.ClientResponse, attempt: int):
        retry_after = self._extract_retry_after_seconds(resp)
        delay = self._compute_backoff(attempt, retry_after)
        logger.warning(
            "429 rate-limited on %s (attempt %d/%d) — backing off %.2fs",
            url,
            attempt + 1,
            self._retry.max_attempts,
            delay,
        )
        await asyncio.sleep(delay)

    async def _handle_5xx(self, url: str, status: int, attempt: int):
        delay = self._compute_backoff(attempt)
        logger.warning(
            "%d server error on %s (attempt %d/%d) — backing off %.2fs",
            status,
            url,
            attempt + 1,
            self._retry.max_attempts,
            delay,
        )
        await asyncio.sleep(delay)

    async def _handle_transient(self, url: str, exc: Exception, attempt: int):
        delay = self._compute_backoff(attempt)
        logger.warning(
            "Transient error on %s (%s) (attempt %d/%d) — backing off %.2fs",
            url,
            exc.__class__.__name__,
            attempt + 1,
            self._retry.max_attempts,
            delay,
        )
        await asyncio.sleep(delay)
