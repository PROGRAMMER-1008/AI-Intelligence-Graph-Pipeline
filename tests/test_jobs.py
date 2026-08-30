"""
Tests for src/scrapers/jobs.py. Since all 5 job board hosts are
outside the dev sandbox's network allowlist (see module docstring in
jobs.py), each source's parsing/filtering logic is tested by mocking
AsyncHttpClient.get_json / get_text to return realistic fixture data
matching each board's documented schema, rather than hitting the
network. This proves the parsing and AI-filtering logic is correct;
it does NOT prove the live endpoints still return exactly this shape
today — see README "Verifying the jobs pipeline" for the live check
to run once outside the sandbox.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scrapers.jobs import (
    _infer_role_family,
    _matches_ai_keywords,
    fetch_arbeitnow_jobs,
    fetch_jobicy_jobs,
    fetch_remoteok_jobs,
    fetch_wwr_jobs,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _mock_client_returning_json(payload: dict | list) -> MagicMock:
    client = MagicMock()
    client.get_json = AsyncMock(return_value=payload)
    return client


def _mock_client_returning_text(text: str) -> MagicMock:
    client = MagicMock()
    client.get_text = AsyncMock(return_value=text)
    return client


# --- keyword / role inference (pure functions, no network) -----------------


def test_ai_keyword_matches_positive_cases():
    assert _matches_ai_keywords("Senior Machine Learning Engineer", "") is True
    assert _matches_ai_keywords("AI Research Scientist", "") is True
    assert _matches_ai_keywords("Full Stack Engineer", "react, node, ai integrations") is True
    assert _matches_ai_keywords("NLP Engineer", "") is True


def test_ai_keyword_matches_negative_cases():
    assert _matches_ai_keywords("Backend Developer", "python, django") is False
    assert _matches_ai_keywords("Marketing Manager", "seo, content") is False


def test_role_family_inference():
    assert _infer_role_family("Senior Software Engineer") == "Engineering"
    assert _infer_role_family("Data Scientist") == "Data Science"
    assert _infer_role_family("Research Scientist, NLP") == "Research"
    assert _infer_role_family("Chief Marketing Officer") is None


# --- RemoteOK ----------------------------------------------------------------


def test_remoteok_skips_legend_object():
    """The first array element (no 'id' field) must never become a job record."""
    fixture = json.loads((FIXTURES / "remoteok_sample_response.json").read_text())
    client = _mock_client_returning_json(fixture)

    async def run():
        return await fetch_remoteok_jobs(client)

    records = asyncio.run(run())
    # 2 real jobs per fixture, fetched once per tag (5 tags) but each
    # call returns the SAME fixture in this test, so we just check
    # no record was built from the legend object.
    for r in records:
        assert r.content.company != "" 
        assert r.content.title in ("Senior Machine Learning Engineer", "Backend Developer")


def test_remoteok_marks_all_jobs_remote():
    fixture = json.loads((FIXTURES / "remoteok_sample_response.json").read_text())
    client = _mock_client_returning_json(fixture)

    async def run():
        return await fetch_remoteok_jobs(client)

    records = asyncio.run(run())
    assert all(r.content.is_remote is True for r in records)


def test_remoteok_infers_role_family():
    fixture = json.loads((FIXTURES / "remoteok_sample_response.json").read_text())
    client = _mock_client_returning_json(fixture)

    async def run():
        return await fetch_remoteok_jobs(client)

    records = asyncio.run(run())
    ml_job = next(r for r in records if "Machine Learning" in r.content.title)
    assert ml_job.content.role_family == "Engineering"


# --- Jobicy ------------------------------------------------------------------


def test_jobicy_parses_documented_fields():
    fixture = json.loads((FIXTURES / "jobicy_sample_response.json").read_text())
    client = _mock_client_returning_json(fixture)

    async def run():
        return await fetch_jobicy_jobs(client)

    records = asyncio.run(run())
    assert len(records) >= 1
    nlp_job = next(r for r in records if "NLP" in r.content.title)
    assert nlp_job.content.company == "LinguaTech"
    assert nlp_job.content.is_remote is True


def test_jobicy_anywhere_geo_becomes_remote_location():
    fixture = json.loads((FIXTURES / "jobicy_sample_response.json").read_text())
    client = _mock_client_returning_json(fixture)

    async def run():
        return await fetch_jobicy_jobs(client)

    records = asyncio.run(run())
    nlp_job = next(r for r in records if "NLP" in r.content.title)
    assert nlp_job.content.location == "Remote"


# --- Arbeitnow -----------------------------------------------------------


def test_arbeitnow_filters_non_ai_jobs():
    fixture = json.loads((FIXTURES / "arbeitnow_sample_response.json").read_text())
    client = _mock_client_returning_json(fixture)

    async def run():
        return await fetch_arbeitnow_jobs(client, max_pages=1)

    records = asyncio.run(run())
    titles = [r.content.title for r in records]
    assert "AI Platform Engineer" in titles
    assert "Office Manager" not in titles  # correctly filtered out as non-AI


def test_arbeitnow_parses_unix_timestamp():
    fixture = json.loads((FIXTURES / "arbeitnow_sample_response.json").read_text())
    client = _mock_client_returning_json(fixture)

    async def run():
        return await fetch_arbeitnow_jobs(client, max_pages=1)

    records = asyncio.run(run())
    ai_job = next(r for r in records if r.content.title == "AI Platform Engineer")
    assert ai_job.content.date is not None
    assert ai_job.content.date.startswith("2026-")


def test_arbeitnow_respects_remote_flag():
    fixture = json.loads((FIXTURES / "arbeitnow_sample_response.json").read_text())
    client = _mock_client_returning_json(fixture)

    async def run():
        return await fetch_arbeitnow_jobs(client, max_pages=1)

    records = asyncio.run(run())
    ai_job = next(r for r in records if r.content.title == "AI Platform Engineer")
    assert ai_job.content.is_remote is True


# --- We Work Remotely ------------------------------------------------------


def test_wwr_filters_non_ai_jobs():
    xml = (FIXTURES / "wwr_sample_feed.xml").read_text()
    client = _mock_client_returning_text(xml)

    async def run():
        return await fetch_wwr_jobs(client)

    records = asyncio.run(run())
    titles = [r.content.title for r in records]
    assert any("Infrastructure Engineer" in t for t in titles)
    assert not any("Frontend Developer" in t for t in titles)


def test_wwr_splits_company_from_title():
    xml = (FIXTURES / "wwr_sample_feed.xml").read_text()
    client = _mock_client_returning_text(xml)

    async def run():
        return await fetch_wwr_jobs(client)

    records = asyncio.run(run())
    ml_job = records[0]
    assert ml_job.content.company == "NeuralForge"
    assert "Infrastructure Engineer" in ml_job.content.title


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
