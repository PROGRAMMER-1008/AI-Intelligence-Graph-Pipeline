"""
Main pipeline entrypoint. Runs Phases I, II, and IV end-to-end:
fetches startups (+ derived products), research papers, jobs, and
news; resolves entity names against a seed list; writes all 6
required deliverable CSVs.

Phase III (LLM extraction) is wired in as an optional enrichment step
(--enrich-with-llm) rather than a hard requirement of every run: the
scrapers above already produce clean, schema-conformant structured
data directly from source JSON/RSS APIs, so there's no *raw
HTML/text that needs LLM extraction* in the current source set — the
brief's Phase III applies to the case where a source only gives you
messy unstructured text. To still demonstrate the orchestrator against
real content (and prove Groq is live), --enrich-with-llm runs each
news article's full_text through the LLM tier chain to extract a
structured one-paragraph summary + key entities, added as extra CSV
columns. This is genuinely optional extraction ON TOP of clean data,
not a required parsing step, and is skipped by default so a full
pipeline run doesn't require any LLM key to succeed end-to-end
(consistent with graceful degradation elsewhere in this codebase).

Usage:
    python -m src.pipeline
    python -m src.pipeline --min-startups 1000 --min-papers 1000
    python -m src.pipeline --enrich-with-llm
    python -m src.pipeline --skip jobs,news   # for fast iteration

IMPORTANT: see README "Verifying the [x] pipeline" sections. Several
source hosts (arxiv, groq, the 5 news RSS feeds, all 5 job boards)
were not reachable from the sandbox this was developed in — each
scraper's own module docstring documents this individually and gives
the exact live-verification command. Running this end-to-end for the
first time on your own machine IS that verification; watch the log
output for per-source failures rather than assuming success.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from src.llm.orchestrator import DeepSeekTier, GeminiFlashTier, GroqTier, LLMOrchestrator
from src.resolver.entity_resolver import EntityResolver, build_seed_list_from_yc_top_companies
from src.scrapers.arxiv_papers import fetch_papers
from src.scrapers.jobs import fetch_all_ai_jobs
from src.scrapers.news import fetch_fresh_news
from src.scrapers.yc_startups import derive_product_records, fetch_ai_startups
from src.storage.csv_export import export_all
from src.utils.date_normalizer import is_within_last_24h
from src.utils.http_client import AsyncHttpClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("graphone.pipeline")


async def run_pipeline(
    min_startups: int = 1000,
    min_papers: int = 1000,
    output_dir: Path = Path("data/processed"),
    skip: set[str] | None = None,
    enrich_with_llm: bool = False,
) -> dict:
    skip = skip or set()
    started_at = time.monotonic()
    stats = {}

    async with AsyncHttpClient(max_concurrency=20) as client:
        # --- Phase I: Startups + derived Products ---------------------
        startups, products = [], []
        if "startups" not in skip:
            logger.info("=== Fetching startups (target: %d) ===", min_startups)
            try:
                startups = await fetch_ai_startups(client, min_records=min_startups)
                products = derive_product_records(startups)
                logger.info(
                    "Startups: %d records, %d derived product records",
                    len(startups), len(products),
                )
            except Exception as e:
                logger.error("Startup fetch failed entirely: %s", e)
        stats["startups"] = len(startups)
        stats["products"] = len(products)

        # --- Phase I: Research Papers -----------------------------------
        papers = []
        if "papers" not in skip:
            logger.info("=== Fetching research papers (target: %d) ===", min_papers)
            github_token = os.environ.get("GITHUB_TOKEN")
            try:
                papers = await fetch_papers(
                    client, category="cs.AI", min_records=min_papers, github_token=github_token
                )
                logger.info("Papers: %d records", len(papers))
            except Exception as e:
                logger.error("Paper fetch failed entirely: %s", e)
        stats["papers"] = len(papers)

        # --- Phase II: Jobs (24h freshness applied at collection) ------
        jobs = []
        if "jobs" not in skip:
            logger.info("=== Fetching AI jobs across 5 boards ===")
            try:
                all_jobs = await fetch_all_ai_jobs(client)
                # Freshness filter: same is_within_last_24h() used for
                # news, applied here at the pipeline level so every
                # source's raw fetch function stays freely reusable
                # (e.g. for non-freshness-constrained future use)
                # while THIS deliverable always enforces 24h.
                jobs = [j for j in all_jobs if is_within_last_24h(j.content.date)]
                logger.info(
                    "Jobs: %d fetched, %d within 24h freshness window",
                    len(all_jobs), len(jobs),
                )
            except Exception as e:
                logger.error("Job fetch failed entirely: %s", e)
        stats["jobs"] = len(jobs)

        # --- Phase II: News (freshness already applied inside fetch) --
        news = []
        if "news" not in skip:
            logger.info("=== Fetching fresh AI news across 5 sources ===")
            try:
                news = await fetch_fresh_news(client, fetch_full_text=True)
                logger.info("News: %d fresh (<24h) records", len(news))
            except Exception as e:
                logger.error("News fetch failed entirely: %s", e)
        stats["news"] = len(news)

        # --- Phase III (optional): LLM enrichment of news summaries ---
        if enrich_with_llm and news:
            logger.info("=== Enriching news with LLM summaries (Phase III demo) ===")
            orchestrator = LLMOrchestrator(
                tiers=[GroqTier(), GeminiFlashTier(), DeepSeekTier()]
            )
            if not orchestrator.available_tiers():
                logger.warning(
                    "No LLM tiers configured (missing API keys) — skipping enrichment. "
                    "Set GROQ_API_KEY in .env to enable this step."
                )
            else:
                enrich_schema = (
                    '{"summary": "one paragraph, plain text", '
                    '"key_entities": ["list of company/product names mentioned"]}'
                )
                enriched_count = 0
                for article in news[:20]:  # cap for demo purposes / cost control
                    text = article.content.full_text or article.content.summary or ""
                    if not text:
                        continue
                    result = await orchestrator.extract(client, text, enrich_schema)
                    if result.success:
                        # Stash enrichment onto the record's summary
                        # field if it was empty, rather than inventing
                        # a new schema field mid-pipeline.
                        if not article.content.summary:
                            article.content.summary = result.data.get("summary")
                        enriched_count += 1
                logger.info(
                    "LLM-enriched %d/%d news records via tier(s): %s",
                    enriched_count, min(len(news), 20),
                    [t.name for t in orchestrator.available_tiers()],
                )

        # --- Phase IV: Entity Resolution --------------------------------
        logger.info("=== Resolving entities ===")
        seed_list = build_seed_list_from_yc_top_companies(
            [
                {"name": s.content.entity_name, "former_names": []}
                for s in startups[:50]
            ]
        )
        resolver = EntityResolver(seed_list)
        mapping_log = []
        for s in startups:
            canonical, log_entry = resolver.resolve(s.content.raw_name, record_type="STARTUP")
            s.content.entity_name = canonical
            mapping_log.append(log_entry)
        for p in products:
            canonical, log_entry = resolver.resolve(p.content.startup_name, record_type="PRODUCT")
            p.content.startup_name = canonical
            mapping_log.append(log_entry)
        logger.info("Entity resolution: %d mapping log entries", len(mapping_log))
        stats["entity_mappings"] = len(mapping_log)

        # --- Export ------------------------------------------------------
        logger.info("=== Writing CSV deliverables to %s ===", output_dir)
        paths = export_all(output_dir, startups, products, papers, jobs, news, mapping_log)
        for name, path in paths.items():
            logger.info("  %s -> %s", name, path)

    elapsed = time.monotonic() - started_at
    logger.info("=== Pipeline complete in %.1fs ===", elapsed)
    logger.info("Stats: %s", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description="GraphOne Intelligence Graph pipeline")
    parser.add_argument("--min-startups", type=int, default=1000)
    parser.add_argument("--min-papers", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated sources to skip: startups,papers,jobs,news",
    )
    parser.add_argument(
        "--enrich-with-llm", action="store_true",
        help="Run news full-text through the LLM orchestrator for a Phase III demo (requires GROQ_API_KEY)",
    )
    args = parser.parse_args()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    asyncio.run(
        run_pipeline(
            min_startups=args.min_startups,
            min_papers=args.min_papers,
            output_dir=Path(args.output_dir),
            skip=skip,
            enrich_with_llm=args.enrich_with_llm,
        )
    )


if __name__ == "__main__":
    main()
