"""
Date normalization for the freshness pipeline (Phase II).

Real-world date strings arriving from scraped HTML/RSS are a mess:
  - ISO-8601 with/without timezone: "2026-08-27T10:00:00Z"
  - RFC-822 (common in RSS): "Wed, 27 Aug 2026 10:00:00 GMT"
  - Relative: "2 hours ago", "yesterday", "3d ago", "just now"
  - Human: "August 27, 2026", "27/08/2026" (ambiguous vs US format!)
  - Missing entirely: many job boards and news aggregators strip
    dates from list pages; only the detail page has them, or
    sometimes NOTHING has them.

Strategy, in priority order:
  1. Try strict ISO-8601 / RFC-822 parsing (dateutil handles both).
  2. Pattern-match relative expressions against a fixed reference time
     (the time we made the request — NOT time.now() call-by-call,
     to keep all dates in a single fetch batch internally consistent).
  3. If nothing parses, return None and let the freshness heuristic
     (below) decide admission based on crawl-state rather than date.

The freshness heuristic (Phase II "Intelligent Heuristics" requirement):
  If a source gives no reliable date, we do NOT guess a date (that
  would be fabricating data — a straight path to the disqualification
  clause in the brief: "Hallucinated data ... will result in immediate
  disqualification"). Instead we track a content hash + first-seen
  timestamp per source in a local seen-store. An article is treated
  as "fresh" only if its content hash was NOT present in the previous
  crawl's seen-store. This trades date precision for correctness: we
  never claim a fabricated timestamp, and we still get useful
  freshness signal for undated sources.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dateutil import parser as dateutil_parser
from dateutil.tz import UTC


_RELATIVE_PATTERNS = [
    # (regex, unit -> timedelta kwargs multiplier)
    (re.compile(r"just now|moments? ago", re.I), None),
    (re.compile(r"(\d+)\s*s(ec(ond)?s?)?\s*ago", re.I), "seconds"),
    (re.compile(r"(\d+)\s*m(in(ute)?s?)?\s*ago", re.I), "minutes"),
    (re.compile(r"(\d+)\s*h(ou)?rs?\s*ago", re.I), "hours"),
    (re.compile(r"(\d+)\s*d(ays?)?\s*ago", re.I), "days"),
    (re.compile(r"(\d+)\s*w(eeks?)?\s*ago", re.I), "weeks"),
    (re.compile(r"yesterday", re.I), "yesterday"),
    (re.compile(r"today", re.I), "today"),
]


_MONTH_NAMES = re.compile(
    r"\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|"
    r"aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?)\b",
    re.I,
)
_WEEKDAY_NAMES = re.compile(
    r"\b(mon(day)?|tue(s(day)?)?|wed(nesday)?|thu(rs(day)?)?|fri(day)?|"
    r"sat(urday)?|sun(day)?)\b",
    re.I,
)
_NUMERIC_DATE_PATTERN = re.compile(
    r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b"  # 2026-08-27, 27/08/2026, etc.
)
_ISO_LIKE = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")


def _looks_like_a_real_date(raw: str) -> bool:
    """
    Guard against dateutil's fuzzy=True fabricating a date from a bare
    number in unstructured garbage text. Requires at least one actual
    date signal: a month name, weekday name, ISO-8601-shaped
    timestamp, or a slash/hyphen-delimited numeric date. A lone
    4-digit number (e.g. stray year mentioned in unrelated text) does
    NOT qualify on its own.
    """
    return bool(
        _MONTH_NAMES.search(raw)
        or _WEEKDAY_NAMES.search(raw)
        or _NUMERIC_DATE_PATTERN.search(raw)
        or _ISO_LIKE.search(raw)
    )


def normalize_date(raw: Optional[str], reference_time: Optional[datetime] = None) -> Optional[str]:
    """
    Best-effort normalization of an arbitrary date string to ISO-8601 UTC.
    Returns None if the string cannot be confidently parsed — callers
    must NOT fabricate a fallback date; None flows into the freshness
    heuristic instead.
    """
    if not raw or not raw.strip():
        return None

    raw = raw.strip()
    ref = reference_time or datetime.now(timezone.utc)

    # 1. Relative expressions
    for pattern, unit in _RELATIVE_PATTERNS:
        m = pattern.search(raw)
        if not m:
            continue
        if unit is None:  # "just now"
            return ref.astimezone(UTC).isoformat()
        if unit == "yesterday":
            return (ref - timedelta(days=1)).astimezone(UTC).isoformat()
        if unit == "today":
            return ref.astimezone(UTC).isoformat()
        qty = int(m.group(1))
        delta = timedelta(**{unit: qty})
        return (ref - delta).astimezone(UTC).isoformat()

    # 2. Strict / semi-strict parsing via dateutil (handles ISO-8601,
    #    RFC-822 from RSS feeds, and most human-readable formats).
    #
    #    fuzzy=True is necessary for real-world strings like
    #    "Posted on August 26, 2026 by Staff" but is dangerously
    #    over-eager on pure garbage: dateutil will happily extract a
    #    bare 4-digit number from ANYWHERE in a string and construct a
    #    full date from it (confirmed during testing:
    #    "not-a-date-at-all-2099" parses to 2099-08-28 by treating
    #    "2099" as a year and defaulting month/day to today's). That
    #    is fabrication by omission -- exactly what this function
    #    exists to prevent. Guard: require the input to contain
    #    *something* that looks like an actual date token (a month
    #    name, a weekday name, or a slash/hyphen-separated numeric
    #    date pattern) before trusting a fuzzy parse. A bare number
    #    with no such structure is rejected rather than guessed.
    if not _looks_like_a_real_date(raw):
        return None

    try:
        dt = dateutil_parser.parse(raw, fuzzy=True)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)  # assume UTC if unspecified —
            # explicit assumption, documented, rather than silent local-time guess
        return dt.astimezone(UTC).isoformat()
    except (ValueError, OverflowError):
        return None


def is_within_last_24h(iso_date: Optional[str], reference_time: Optional[datetime] = None) -> bool:
    if not iso_date:
        return False
    ref = reference_time or datetime.now(timezone.utc)
    try:
        dt = dateutil_parser.isoparse(iso_date)
        return (ref - dt) <= timedelta(hours=24)
    except (ValueError, OverflowError):
        return False


def content_hash(*parts: str) -> str:
    """Stable hash of content used for dedup / freshness-without-a-date."""
    joined = "||".join(p.strip().lower() for p in parts if p)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass
class SeenStore:
    """
    Tiny persistent set of content hashes, one file per source, used
    for the "intelligent heuristic" freshness fallback when a source
    gives no parseable date at all.

    This directly answers Phase VI question 3 ("How does your
    architecture ensure we never process the same article/job twice
    across distributed crawler nodes?") at small scale: the seen-store
    is the single source of truth for "have we emitted this before".
    In the architecture doc we describe how this generalizes to a
    shared Redis set for multi-node deployment — see architecture.md.
    """

    path: Path
    _seen: set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.path.exists():
            self._seen = set(json.loads(self.path.read_text()))

    def is_new(self, h: str) -> bool:
        return h not in self._seen

    def mark_seen(self, h: str) -> None:
        self._seen.add(h)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(self._seen)))
