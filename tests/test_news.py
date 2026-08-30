"""
Tests for RSS parsing and freshness filtering (src/scrapers/news.py).
Fixture matches real RSS 2.0 structure; the 5 real feed hosts are
unreachable from the dev sandbox (see module docstring in news.py).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scrapers.news import _parse_rss
from src.utils.date_normalizer import is_within_last_24h, normalize_date

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rss_sample_feed.xml"
REFERENCE_TIME = datetime(2026, 8, 28, 4, 30, 0, tzinfo=timezone.utc)


def test_parses_all_items():
    xml = FIXTURE_PATH.read_text()
    items = _parse_rss(xml)
    assert len(items) == 4


def test_html_entities_stripped_from_summary():
    xml = FIXTURE_PATH.read_text()
    items = _parse_rss(xml)
    first = items[0]
    assert "<p>" not in first["summary"]
    assert "Series A" in first["summary"]


def test_fresh_article_within_24h():
    xml = FIXTURE_PATH.read_text()
    items = _parse_rss(xml)
    fresh_item = next(i for i in items if "Series A" in i["title"])
    normalized = normalize_date(fresh_item["raw_date"], reference_time=REFERENCE_TIME)
    assert is_within_last_24h(normalized, reference_time=REFERENCE_TIME) is True


def test_stale_article_beyond_24h_filtered():
    xml = FIXTURE_PATH.read_text()
    items = _parse_rss(xml)
    stale_item = next(i for i in items if "Three Days Ago" in i["title"])
    normalized = normalize_date(stale_item["raw_date"], reference_time=REFERENCE_TIME)
    assert is_within_last_24h(normalized, reference_time=REFERENCE_TIME) is False


def test_undated_article_never_fabricates_freshness():
    xml = FIXTURE_PATH.read_text()
    items = _parse_rss(xml)
    undated = next(i for i in items if "No Date At All" in i["title"])
    assert undated["raw_date"] is None
    normalized = normalize_date(undated["raw_date"], reference_time=REFERENCE_TIME)
    assert normalized is None
    assert is_within_last_24h(normalized, reference_time=REFERENCE_TIME) is False


def test_end_to_end_freshness_filter_count():
    """Of 4 fixture items: 1 fresh, 1 stale, 1 undated, 1 fresh (2h old)
    -> exactly 2 should survive a real freshness filter pass."""
    xml = FIXTURE_PATH.read_text()
    items = _parse_rss(xml)
    surviving = []
    for item in items:
        normalized = normalize_date(item["raw_date"], reference_time=REFERENCE_TIME)
        if normalized and is_within_last_24h(normalized, reference_time=REFERENCE_TIME):
            surviving.append(item)
    assert len(surviving) == 2


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
