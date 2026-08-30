"""
End-to-end test of src/pipeline.py's orchestration logic, with every
network-touching scraper function mocked out. This does NOT prove any
live source still returns the expected shape (each scraper module has
its own fixture-based tests for that, e.g. test_yc_startups.py,
test_jobs.py) — it proves the PIPELINE ITSELF correctly: calls each
source, derives products from startups, resolves entities, applies
job freshness filtering, and writes all 6 CSVs, INCLUDING when one or
more sources fail entirely (the graceful-degradation contract that
matters most in production, and the exact behavior observed live when
yc-oss.github.io's sandbox allowlist entry disappeared mid-development
— see conversation history / commit notes).
"""

import asyncio
import csv
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_pipeline
from src.scrapers.yc_startups import _company_to_record
from src.schemas import JobContent, JobRecord, NewsContent, NewsRecord, ResearchPaperContent, ResearchPaperRecord, Source

SAMPLE_COMPANY = {
    "id": 1, "name": "VectorForge", "one_liner": "AI vector search",
    "long_description": "desc", "team_size": 45, "website": "https://vf.com",
    "all_locations": "SF", "industries": ["B2B"], "launched_at": 1700000000,
    "status": "Active", "isHiring": True, "url": "https://yc.com/vf",
}


def _sample_startups():
    return [_company_to_record(SAMPLE_COMPANY)]


def _sample_papers():
    return [
        ResearchPaperRecord(
            source=Source(name="arXiv", url="https://arxiv.org/abs/1234.5678"),
            content=ResearchPaperContent(
                title="Test Paper", authors=["A. Author"],
                paper_url="https://arxiv.org/abs/1234.5678",
            ),
        )
    ]


def _sample_jobs(fresh: bool, stale: bool):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    jobs = []
    if fresh:
        jobs.append(
            JobRecord(
                source=Source(name="RemoteOK", url="https://remoteok.com/1"),
                content=JobContent(
                    company="TestCo", title="ML Engineer",
                    date=now.isoformat(), is_remote=True,
                ),
            )
        )
    if stale:
        jobs.append(
            JobRecord(
                source=Source(name="RemoteOK", url="https://remoteok.com/2"),
                content=JobContent(
                    company="OldCo", title="Old ML Job",
                    date=(now - timedelta(days=5)).isoformat(), is_remote=True,
                ),
            )
        )
    return jobs


def _sample_news():
    return [
        NewsRecord(
            source=Source(name="TechCrunch AI", url="https://example.test/1"),
            content=NewsContent(title="Test Article", url="https://example.test/1"),
        )
    ]


def test_pipeline_writes_all_csvs_with_normal_data():
    async def run():
        with (
            patch("src.pipeline.fetch_ai_startups", new=AsyncMock(return_value=_sample_startups())),
            patch("src.pipeline.fetch_papers", new=AsyncMock(return_value=_sample_papers())),
            patch("src.pipeline.fetch_all_ai_jobs", new=AsyncMock(return_value=_sample_jobs(fresh=True, stale=True))),
            patch("src.pipeline.fetch_fresh_news", new=AsyncMock(return_value=_sample_news())),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                stats = await run_pipeline(output_dir=Path(tmp))
                assert stats["startups"] == 1
                assert stats["products"] == 1
                assert stats["papers"] == 1
                assert stats["news"] == 1
                for fname in [
                    "startups.csv", "products.csv", "research_papers.csv",
                    "jobs.csv", "news.csv", "entity_mapping_log.csv",
                ]:
                    assert (Path(tmp) / fname).exists()

    asyncio.run(run())


def test_pipeline_applies_job_freshness_filter():
    """The pipeline-level freshness filter must drop stale jobs even
    though the scraper itself returned them (scraper-level filtering
    is a source's own responsibility; the pipeline enforces the final
    24h guarantee regardless)."""

    async def run():
        with (
            patch("src.pipeline.fetch_ai_startups", new=AsyncMock(return_value=[])),
            patch("src.pipeline.fetch_papers", new=AsyncMock(return_value=[])),
            patch("src.pipeline.fetch_all_ai_jobs", new=AsyncMock(return_value=_sample_jobs(fresh=True, stale=True))),
            patch("src.pipeline.fetch_fresh_news", new=AsyncMock(return_value=[])),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                stats = await run_pipeline(output_dir=Path(tmp))
                assert stats["jobs"] == 1  # only the fresh one survives

                with open(Path(tmp) / "jobs.csv") as f:
                    rows = list(csv.DictReader(f))
                    assert len(rows) == 1
                    assert rows[0]["company"] == "TestCo"

    asyncio.run(run())


def test_pipeline_survives_total_source_failure():
    """
    Regression test for exactly the scenario observed live during
    development: a source host becomes unreachable (sandbox allowlist
    change removed yc-oss.github.io access mid-session) and the
    scraper's own try/except returns an empty list rather than
    raising. The pipeline must still complete and write valid
    (empty) CSVs for every tab rather than crashing.
    """

    async def run():
        with (
            patch("src.pipeline.fetch_ai_startups", new=AsyncMock(return_value=[])),
            patch("src.pipeline.fetch_papers", new=AsyncMock(side_effect=Exception("host unreachable"))),
            patch("src.pipeline.fetch_all_ai_jobs", new=AsyncMock(side_effect=Exception("host unreachable"))),
            patch("src.pipeline.fetch_fresh_news", new=AsyncMock(side_effect=Exception("host unreachable"))),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                stats = await run_pipeline(output_dir=Path(tmp))
                assert stats["startups"] == 0
                assert stats["papers"] == 0
                assert stats["jobs"] == 0
                assert stats["news"] == 0
                # Every CSV must still exist, even if empty
                for fname in [
                    "startups.csv", "products.csv", "research_papers.csv",
                    "jobs.csv", "news.csv", "entity_mapping_log.csv",
                ]:
                    assert (Path(tmp) / fname).exists()

    asyncio.run(run())


def test_pipeline_respects_skip_argument():
    async def run():
        with (
            patch("src.pipeline.fetch_ai_startups", new=AsyncMock(return_value=_sample_startups())) as mock_startups,
            patch("src.pipeline.fetch_papers", new=AsyncMock(return_value=[])) as mock_papers,
            patch("src.pipeline.fetch_all_ai_jobs", new=AsyncMock(return_value=[])),
            patch("src.pipeline.fetch_fresh_news", new=AsyncMock(return_value=[])),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                await run_pipeline(output_dir=Path(tmp), skip={"papers"})
                mock_startups.assert_called_once()
                mock_papers.assert_not_called()

    asyncio.run(run())


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
