"""
Storage layer — writes pipeline output as CSVs, one per required tab
(Startups, Products, Research Papers, Jobs, News, Entity Mapping Log).

WHY CSV INSTEAD OF DIRECT GOOGLE SHEETS API WRITES: the brief's
deliverable is "a public Google Sheet link" — the destination is a
Sheet, but the mechanism for getting there doesn't have to be a live
API integration. Google Sheets can import a CSV directly (File >
Import > Upload, one per tab) in under a minute per tab, with zero
Google Cloud credential setup. Given the deadline, wiring up a
service-account OAuth flow (which itself requires the user to create
a GCP project, enable the Sheets API, generate and download a
credentials JSON, and share the sheet with a service-account email)
is meaningfully more setup risk than it's worth for what is
ultimately a one-time export step. gspread is still listed in
requirements.txt and a live-write path can be added later
(see storage/sheets_export.py stub note below) if there's time
remaining, but CSV-then-import is the pragmatic, low-risk path to the
actual deliverable under time pressure — it produces the exact same
end artifact (a populated public Google Sheet) with far less that can
go wrong the night before a deadline.

Every writer here flattens a Pydantic record's nested `content` object
into flat columns (Sheets/CSV have no concept of nested JSON) and
always includes schema_version, record_type, source name/url, and
collected_at so no information from the canonical schema is lost in
the flattening — anyone auditing the sheet can trace every row back
to schema + source, per the brief's traceability requirement.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Sequence

from src.schemas import (
    EntityMappingLogEntry,
    JobRecord,
    NewsRecord,
    ProductRecord,
    ResearchPaperRecord,
    StartupRecord,
)

logger = logging.getLogger("graphone.storage")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Wrote %d rows to %s", len(rows), path)


def export_startups_csv(records: list[StartupRecord], output_dir: Path) -> Path:
    fieldnames = [
        "schema_version", "record_type", "source_name", "source_url",
        "entity_name", "raw_name", "one_liner", "description",
        "employee_count", "website", "location", "industries",
        "founded_or_launched_at", "status", "is_hiring", "collected_at",
    ]
    rows = []
    for r in records:
        rows.append({
            "schema_version": r.schema_version,
            "record_type": r.record_type.value,
            "source_name": r.source.name,
            "source_url": r.source.url,
            "entity_name": r.content.entity_name,
            "raw_name": r.content.raw_name,
            "one_liner": r.content.one_liner,
            "description": r.content.description,
            "employee_count": r.content.employee_count,
            "website": r.content.website,
            "location": r.content.location,
            "industries": "; ".join(r.content.industries),
            "founded_or_launched_at": r.content.founded_or_launched_at,
            "status": r.content.status,
            "is_hiring": r.content.is_hiring,
            "collected_at": r.collected_at,
        })
    path = output_dir / "startups.csv"
    _write_csv(path, rows, fieldnames)
    return path


def export_products_csv(records: list[ProductRecord], output_dir: Path) -> Path:
    fieldnames = [
        "schema_version", "record_type", "source_name", "source_url",
        "startup_name", "product_name", "pricing_model", "description",
        "website", "category", "collected_at",
    ]
    rows = []
    for r in records:
        rows.append({
            "schema_version": r.schema_version,
            "record_type": r.record_type.value,
            "source_name": r.source.name,
            "source_url": r.source.url,
            "startup_name": r.content.startup_name,
            "product_name": r.content.product_name,
            "pricing_model": r.content.pricing_model.value,
            "description": r.content.description,
            "website": r.content.website,
            "category": r.content.category,
            "collected_at": r.collected_at,
        })
    path = output_dir / "products.csv"
    _write_csv(path, rows, fieldnames)
    return path


def export_research_papers_csv(records: list[ResearchPaperRecord], output_dir: Path) -> Path:
    fieldnames = [
        "schema_version", "record_type", "source_name", "source_url",
        "title", "authors", "paper_url", "github_url", "github_stars",
        "published_date", "arxiv_id", "categories", "abstract", "collected_at",
    ]
    rows = []
    for r in records:
        rows.append({
            "schema_version": r.schema_version,
            "record_type": r.record_type.value,
            "source_name": r.source.name,
            "source_url": r.source.url,
            "title": r.content.title,
            "authors": "; ".join(r.content.authors),
            "paper_url": r.content.paper_url,
            "github_url": r.content.github_url,
            "github_stars": r.content.github_stars,
            "published_date": r.content.published_date,
            "arxiv_id": r.content.arxiv_id,
            "categories": "; ".join(r.content.categories),
            "abstract": r.content.abstract,
            "collected_at": r.collected_at,
        })
    path = output_dir / "research_papers.csv"
    _write_csv(path, rows, fieldnames)
    return path


def export_jobs_csv(records: list[JobRecord], output_dir: Path) -> Path:
    fieldnames = [
        "schema_version", "record_type", "source_name", "source_url",
        "company", "title", "date", "is_remote", "role_family",
        "location", "url", "collected_at",
    ]
    rows = []
    for r in records:
        rows.append({
            "schema_version": r.schema_version,
            "record_type": r.record_type.value,
            "source_name": r.source.name,
            "source_url": r.source.url,
            "company": r.content.company,
            "title": r.content.title,
            "date": r.content.date,
            "is_remote": r.content.is_remote,
            "role_family": r.content.role_family,
            "location": r.content.location,
            "url": r.content.url,
            "collected_at": r.collected_at,
        })
    path = output_dir / "jobs.csv"
    _write_csv(path, rows, fieldnames)
    return path


def export_news_csv(records: list[NewsRecord], output_dir: Path) -> Path:
    fieldnames = [
        "schema_version", "record_type", "source_name", "source_url",
        "title", "url", "published_date", "author", "summary",
        "full_text", "collected_at",
    ]
    rows = []
    for r in records:
        rows.append({
            "schema_version": r.schema_version,
            "record_type": r.record_type.value,
            "source_name": r.source.name,
            "source_url": r.source.url,
            "title": r.content.title,
            "url": r.content.url,
            "published_date": r.content.published_date,
            "author": r.content.author,
            "summary": r.content.summary,
            "full_text": r.content.full_text,
            "collected_at": r.collected_at,
        })
    path = output_dir / "news.csv"
    _write_csv(path, rows, fieldnames)
    return path


def export_entity_mapping_log_csv(entries: list[EntityMappingLogEntry], output_dir: Path) -> Path:
    fieldnames = ["raw_name", "canonical_name", "match_method", "confidence", "record_type"]
    rows = [e.model_dump() for e in entries]
    path = output_dir / "entity_mapping_log.csv"
    _write_csv(path, rows, fieldnames)
    return path


def export_all(
    output_dir: Path,
    startups: list[StartupRecord],
    products: list[ProductRecord],
    papers: list[ResearchPaperRecord],
    jobs: list[JobRecord],
    news: list[NewsRecord],
    entity_mapping_log: list[EntityMappingLogEntry],
) -> dict[str, Path]:
    """Writes all 6 required deliverable CSVs and returns their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "startups": export_startups_csv(startups, output_dir),
        "products": export_products_csv(products, output_dir),
        "research_papers": export_research_papers_csv(papers, output_dir),
        "jobs": export_jobs_csv(jobs, output_dir),
        "news": export_news_csv(news, output_dir),
        "entity_mapping_log": export_entity_mapping_log_csv(entity_mapping_log, output_dir),
    }
