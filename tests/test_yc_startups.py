"""
Tests for src/scrapers/yc_startups.py: company-record parsing and the
derive_product_records() transformation. yc-oss.github.io is reachable
from the dev sandbox (unlike arxiv/groq/news/jobs hosts) — this was
confirmed live during initial development (see chat history / README),
but these tests use a fixed sample dict rather than a live call so
they run deterministically and don't depend on YC's live catalog
contents on any given test run.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas import PricingModel
from src.scrapers.yc_startups import _company_to_record, derive_product_records

SAMPLE_COMPANY = {
    "id": 1,
    "name": "VectorForge",
    "one_liner": "AI-powered vector search for enterprises",
    "long_description": "VectorForge builds infrastructure for semantic search at scale.",
    "team_size": 45,
    "website": "https://vectorforge.example",
    "all_locations": "San Francisco, CA",
    "industries": ["B2B", "Developer Tools"],
    "launched_at": 1700000000,
    "status": "Active",
    "isHiring": True,
    "url": "https://www.ycombinator.com/companies/vectorforge",
}


def test_company_to_record_maps_core_fields():
    record = _company_to_record(SAMPLE_COMPANY)
    assert record.content.entity_name == "VectorForge"
    assert record.content.employee_count == 45
    assert record.content.website == "https://vectorforge.example"
    assert record.content.is_hiring is True


def test_company_to_record_converts_unix_launch_timestamp():
    record = _company_to_record(SAMPLE_COMPANY)
    assert record.content.founded_or_launched_at is not None
    assert record.content.founded_or_launched_at.startswith("2023-")  # 1700000000 epoch


def test_company_to_record_handles_missing_launched_at():
    company = dict(SAMPLE_COMPANY)
    del company["launched_at"]
    record = _company_to_record(company)
    assert record.content.founded_or_launched_at is None


def test_derive_product_records_count_matches_input():
    startup = _company_to_record(SAMPLE_COMPANY)
    products = derive_product_records([startup])
    assert len(products) == 1


def test_derive_product_records_pricing_always_unknown():
    """
    Contract test: YC's data has no pricing field, so pricing_model
    must NEVER be guessed as FREE/PAID/FREEMIUM/ENTERPRISE — only
    UNKNOWN is honest here.
    """
    startup = _company_to_record(SAMPLE_COMPANY)
    products = derive_product_records([startup])
    assert products[0].content.pricing_model == PricingModel.UNKNOWN


def test_derive_product_records_preserves_source_url():
    startup = _company_to_record(SAMPLE_COMPANY)
    products = derive_product_records([startup])
    assert products[0].source.url == startup.source.url


def test_derive_product_records_uses_one_liner_as_description():
    startup = _company_to_record(SAMPLE_COMPANY)
    products = derive_product_records([startup])
    assert products[0].content.description == "AI-powered vector search for enterprises"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
