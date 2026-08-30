"""
Tests for the deterministic entity resolver (src/resolver/entity_resolver.py).
Includes the exact example from the assignment brief: "OpenAI",
"OpenAI, Inc.", "Open AI" -> "OpenAI".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.resolver.entity_resolver import EntityResolver, SeedEntity

SEEDS = [
    SeedEntity(canonical_name="OpenAI", aliases=["OpenAI, Inc.", "Open AI", "OpenAI Inc"]),
    SeedEntity(canonical_name="Anthropic", aliases=["Anthropic PBC", "Anthropic, Inc."]),
    SeedEntity(canonical_name="Y Combinator", aliases=["YC", "Y-Combinator"]),
]


def _resolver():
    return EntityResolver(SEEDS)


def test_brief_example_exact_aliases():
    """The exact example given in the assignment brief."""
    r = _resolver()
    for name in ["OpenAI", "OpenAI, Inc.", "Open AI"]:
        canonical, log = r.resolve(name)
        assert canonical == "OpenAI", f"{name} should resolve to OpenAI, got {canonical}"
        assert log.match_method == "exact"
        assert log.confidence == 1.0


def test_case_insensitive_match():
    r = _resolver()
    canonical, log = r.resolve("open ai")
    assert canonical == "OpenAI"


def test_legal_suffix_normalization():
    r = _resolver()
    canonical, log = r.resolve("Anthropic PBC")
    assert canonical == "Anthropic"
    assert log.match_method == "exact"  # PBC is a registered alias here


def test_fuzzy_match_above_threshold():
    r = _resolver()
    canonical, log = r.resolve("Antropic")  # misspelling, missing 'h'
    assert canonical == "Anthropic"
    assert log.match_method == "fuzzy"
    assert log.confidence >= 0.92


def test_near_miss_stays_unresolved_not_fabricated():
    """
    'OpenAl' (capital I swapped for lowercase L) scores 83.3 on
    token_sort_ratio against 'openai' -- below our 92 threshold.
    This must NOT be force-matched: better to leave two spellings
    unmerged than risk conflating a genuinely different (if similarly
    named) entity. Verified score via rapidfuzz directly during
    development.
    """
    r = _resolver()
    canonical, log = r.resolve("OpenAl")
    assert log.match_method == "unresolved"
    assert canonical == "OpenAl"  # emitted as-is, never fabricated


def test_unknown_entity_stays_unresolved():
    r = _resolver()
    canonical, log = r.resolve("Some Random Startup Inc.")
    assert log.match_method == "unresolved"
    assert log.confidence == 0.0


def test_alias_resolution():
    r = _resolver()
    canonical, log = r.resolve("YC")
    assert canonical == "Y Combinator"
    assert log.match_method == "exact"


def test_mapping_log_entry_shape():
    r = _resolver()
    _, log = r.resolve("OpenAI, Inc.", record_type="STARTUP")
    assert log.raw_name == "OpenAI, Inc."
    assert log.canonical_name == "OpenAI"
    assert log.record_type == "STARTUP"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
