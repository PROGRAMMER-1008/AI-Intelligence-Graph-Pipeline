"""
Tests for src/storage/csv_export.py. Verifies each of the 6 required
deliverable CSVs is written with correct headers and correctly
flattened content, using small in-memory record sets built the same
way the real pipeline builds them.
"""

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.resolver.entity_resolver import EntityResolver, SeedEntity
from src.scrapers.yc_startups import _company_to_record, derive_product_records
from src.storage.csv_export import export_all

SAMPLE_COMPANY = {
    "id": 1,
    "name": "VectorForge",
    "one_liner": "AI vector search",
    "long_description": "desc",
    "team_size": 45,
    "website": "https://vf.com",
    "all_locations": "SF",
    "industries": ["B2B"],
    "launched_at": 1700000000,
    "status": "Active",
    "isHiring": True,
    "url": "https://yc.com/vf",
}


def _build_test_data():
    startups = [_company_to_record(SAMPLE_COMPANY)]
    products = derive_product_records(startups)
    resolver = EntityResolver([SeedEntity(canonical_name="VectorForge", aliases=[])])
    _, log_entry = resolver.resolve("VectorForge")
    return startups, products, [log_entry]


def test_all_six_files_are_created():
    startups, products, mapping_log = _build_test_data()
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_all(Path(tmp), startups, products, [], [], [], mapping_log)
        assert len(paths) == 6
        for name, path in paths.items():
            assert path.exists(), f"{name} CSV was not created"


def test_startups_csv_has_correct_headers():
    startups, products, mapping_log = _build_test_data()
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_all(Path(tmp), startups, products, [], [], [], mapping_log)
        with open(paths["startups"]) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row["entity_name"] == "VectorForge"
            assert row["employee_count"] == "45"
            assert row["website"] == "https://vf.com"


def test_products_csv_pricing_model_is_unknown():
    startups, products, mapping_log = _build_test_data()
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_all(Path(tmp), startups, products, [], [], [], mapping_log)
        with open(paths["products"]) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row["pricing_model"] == "UNKNOWN"


def test_entity_mapping_log_csv_has_correct_columns():
    startups, products, mapping_log = _build_test_data()
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_all(Path(tmp), startups, products, [], [], [], mapping_log)
        with open(paths["entity_mapping_log"]) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row["raw_name"] == "VectorForge"
            assert row["canonical_name"] == "VectorForge"
            assert row["match_method"] == "exact"


def test_empty_record_lists_produce_header_only_csv():
    """Sources that found nothing (e.g. no fresh news in the last 24h)
    should still produce a valid, header-only CSV — not crash."""
    startups, products, mapping_log = _build_test_data()
    with tempfile.TemporaryDirectory() as tmp:
        paths = export_all(Path(tmp), startups, products, [], [], [], mapping_log)
        with open(paths["jobs"]) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert rows == []
            assert reader.fieldnames is not None  # header row still written


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
