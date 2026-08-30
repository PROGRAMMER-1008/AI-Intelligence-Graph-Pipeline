"""
Tests for date normalization (src/utils/date_normalizer.py).
Covers relative dates, RFC-822/ISO-8601, and the None-on-failure
contract that the freshness pipeline depends on (never fabricate a
date when one can't be parsed).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.date_normalizer import is_within_last_24h, normalize_date

REF = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)


def test_relative_hours_ago():
    result = normalize_date("2 hours ago", reference_time=REF)
    assert result == "2026-08-27T10:00:00+00:00"
    assert is_within_last_24h(result, reference_time=REF) is True


def test_relative_days_ago_beyond_24h():
    result = normalize_date("3 days ago", reference_time=REF)
    assert is_within_last_24h(result, reference_time=REF) is False


def test_yesterday():
    result = normalize_date("yesterday", reference_time=REF)
    assert result == "2026-08-26T12:00:00+00:00"


def test_just_now():
    result = normalize_date("just now", reference_time=REF)
    assert result == REF.isoformat()


def test_rfc822_format():
    result = normalize_date("Wed, 27 Aug 2026 10:00:00 GMT", reference_time=REF)
    assert result == "2026-08-27T10:00:00+00:00"


def test_iso8601_format():
    result = normalize_date("2026-08-27T05:00:00Z", reference_time=REF)
    assert result == "2026-08-27T05:00:00+00:00"


def test_empty_string_returns_none():
    assert normalize_date("", reference_time=REF) is None


def test_none_input_returns_none():
    assert normalize_date(None, reference_time=REF) is None


def test_unparseable_garbage_returns_none():
    assert normalize_date("garbage not a date xyz", reference_time=REF) is None


def test_never_fabricates_a_date_on_failure():
    # Contract test: the freshness pipeline depends on None (not a
    # guessed/default date) whenever parsing fails, per the brief's
    # anti-hallucination requirement.
    for bad_input in ["", None, "asdf", "not-a-date-at-all-2099"]:
        assert normalize_date(bad_input, reference_time=REF) is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
