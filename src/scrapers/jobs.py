"""
Job signal ingestion — Phase II ("5 AI job boards").

SOURCE DECISION: all 5 boards are read through their own free, public,
no-auth JSON/RSS endpoints — not scraped HTML. Same rationale as
yc_startups.py and news.py: these boards publish structured feeds
specifically for redistribution/aggregation, so using them is the
correct integration point, not a workaround. Endpoints and field
names below were confirmed via multiple independent, mutually
corroborating sources during development (see inline citations) —
NOT executed live from the dev sandbox, since all 5 hosts are outside
its network egress allowlist (same restriction documented in
arxiv_papers.py and news.py). Field-name assumptions are flagged
per-source below; run scripts/verify_jobs_live.py (see README) to
confirm against live responses on first real use.

Sources:
  1. RemoteOK        - https://remoteok.com/api  (JSON, tag-filterable)
  2. Jobicy          - https://jobicy.com/api/v2/remote-jobs (JSON,
                        tag/industry-filterable, documented OpenAPI
                        schema, confirmed field names)
  3. Arbeitnow       - https://arbeitnow.com/api/job-board-api (JSON,
                        page-paginated, has explicit `remote` boolean)
  4. We Work Remotely - https://weworkremotely.com/categories/
                        remote-programming-jobs.rss (RSS; used for its
                        programming category since WWR has no direct
                        "AI" category — filtered further by keyword)
  5. Himalayas       - https://himalayas.app/jobs/api (JSON,
                        offset-paginated per ever-jobs aggregator docs)

AI-relevance filtering: none of these boards has a strict "AI" tag
that every AI job uses consistently, so each source is queried with
AI-relevant tags/keywords (ai, machine-learning, llm, ml-engineer,
nlp, computer-vision, data-scientist) where the source supports
server-side filtering (RemoteOK tag, Jobicy tag/industry), and
filtered client-side by keyword match against title+tags+description
for sources that don't (Arbeitnow, WWR, Himalayas). This mirrors the
approach already used in yc_startups.py (AI_RELEVANT_TAGS).

Freshness/date handling: same normalize_date()/is_within_last_24h()
pipeline as news.py — dates are never fabricated; jobs with no
parseable posting date are dropped rather than guessed into or out
of the "fresh" bucket.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import feedparser
from bs4 import BeautifulSoup

from src.schemas import JobContent, JobRecord, Source
from src.utils.date_normalizer import normalize_date
from src.utils.http_client import AsyncHttpClient

logger = logging.getLogger("graphone.scrapers.jobs")

AI_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    " ai ",
    " ai/",
    "/ai ",
    "llm",
    "nlp",
    "computer vision",
    "deep learning",
    "data scientist",
    "ml engineer",
    "ml infra",
    "genai",
    "generative ai",
]


def _matches_ai_keywords(*texts: Optional[str]) -> bool:
    combined = " ".join(t.lower() for t in texts if t)
    combined = f" {combined} "  # pad so ' ai ' boundary check works at string edges too
    return any(kw in combined for kw in AI_KEYWORDS)


ROLE_FAMILY_KEYWORDS = {
    "Engineering": ["engineer", "developer", "swe", "sde", "architect"],
    "Data Science": ["data scientist", "data science", "ml engineer", "machine learning engineer"],
    "Research": ["research scientist", "researcher", "research engineer"],
    "Product": ["product manager", "product owner"],
    "Design": ["designer", "ux", "ui"],
}


def _infer_role_family(title: str) -> Optional[str]:
    title_lower = title.lower()
    for family, keywords in ROLE_FAMILY_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return family
    return None


# ---------------------------------------------------------------------------
# 1. RemoteOK
# ---------------------------------------------------------------------------

REMOTEOK_TAGS = ["ai", "machine-learning", "nlp", "data-science", "llm"]


async def fetch_remoteok_jobs(client: AsyncHttpClient) -> list[JobRecord]:
    """
    RemoteOK's /api endpoint returns a JSON array whose FIRST element
    is a legend/metadata object (no 'id' field) — a well-documented
    quirk of this API, confirmed across multiple independent
    third-party integration writeups. Every subsequent element is a
    real job. Field names used below (position/company/tags/date/
    epoch/url/description/location) are the consistently-documented
    RemoteOK schema; verify against a live response before relying on
    this in production (see README "Verifying the jobs pipeline").
    """
    records: list[JobRecord] = []
    reference_time = datetime.now(timezone.utc)

    for tag in REMOTEOK_TAGS:
        url = f"https://remoteok.com/api?tags={tag}"
        try:
            data = await client.get_json(url, headers={"User-Agent": "GraphOneBot/1.0"})
        except Exception as e:
            logger.error("RemoteOK fetch failed for tag=%s: %s", tag, e)
            continue

        jobs = [item for item in data if isinstance(item, dict) and "id" in item]
        logger.info("RemoteOK tag=%s -> %d jobs", tag, len(jobs))

        for job in jobs:
            raw_date = job.get("date")  # RemoteOK provides ISO-8601 in 'date'
            normalized = normalize_date(raw_date, reference_time=reference_time)
            title = job.get("position", "")
            content = JobContent(
                company=job.get("company", "Unknown"),
                title=title,
                date=normalized,
                is_remote=True,  # RemoteOK is remote-only by definition
                role_family=_infer_role_family(title),
                location=job.get("location"),
                url=job.get("url"),
            )
            records.append(
                JobRecord(
                    source=Source(name="RemoteOK", url=job.get("url", "https://remoteok.com")),
                    content=content,
                )
            )

    return records


# ---------------------------------------------------------------------------
# 2. Jobicy
# ---------------------------------------------------------------------------

JOBICY_TAGS = ["machine-learning", "data-science", "artificial-intelligence", "llm", "nlp"]
# NOTE: "ai" (bare, 2 characters) was in this list originally and
# returned a live 400 Bad Request from Jobicy's API — confirmed by a
# user running this pipeline. Jobicy's own docs describe `tag` as a
# free-text keyword search rather than a fixed taxonomy, so this is
# most likely a length/validation rule on their side rather than a
# documented restriction; rather than guess further at their internal
# validation, "ai" was replaced with the longer, unambiguous
# "artificial-intelligence" here. Jobicy's own docs (see module
# docstring) show `industry` as a first-class, independently
# documented filter dimension — used below alongside `tag` for
# broader, more reliable coverage.
JOBICY_INDUSTRIES = ["engineering", "data-science"]


async def fetch_jobicy_jobs(client: AsyncHttpClient) -> list[JobRecord]:
    """
    Field names confirmed via Jobicy's own published API docs and
    independent third-party documentation (APILayer marketplace
    listing, go-api-libs OpenAPI spec): jobTitle, companyName,
    jobIndustry, jobType, jobGeo, jobDescription, pubDate. Note:
    Jobicy intentionally delays publication by 6 hours (their own
    stated policy) — factor this into freshness expectations; a job
    "posted" per pubDate may already be up to 6h old when it first
    appears here, which is still within the 24h freshness window but
    worth knowing.
    """
    records: list[JobRecord] = []
    reference_time = datetime.now(timezone.utc)
    seen_urls: set[str] = set()

    async def _fetch_and_parse(url: str, label: str) -> None:
        try:
            data = await client.get_json(url)
        except Exception as e:
            logger.error("Jobicy fetch failed for %s: %s", label, e)
            return

        jobs = data.get("jobs", [])
        logger.info("Jobicy %s -> %d jobs", label, len(jobs))

        for job in jobs:
            job_url = job.get("url")
            if job_url and job_url in seen_urls:
                continue  # already collected via a different tag/industry query
            if job_url:
                seen_urls.add(job_url)

            raw_date = job.get("pubDate")
            normalized = normalize_date(raw_date, reference_time=reference_time)
            title = job.get("jobTitle", "")
            geo = job.get("jobGeo", "")

            content = JobContent(
                company=job.get("companyName", "Unknown"),
                title=title,
                date=normalized,
                is_remote=True,  # Jobicy is a remote-jobs board by definition
                role_family=_infer_role_family(title),
                location=geo if geo and geo.lower() != "anywhere" else "Remote",
                url=job_url,
            )
            records.append(
                JobRecord(
                    source=Source(name="Jobicy", url=job_url or "https://jobicy.com"),
                    content=content,
                )
            )

    for tag in JOBICY_TAGS:
        await _fetch_and_parse(f"https://jobicy.com/api/v2/remote-jobs?count=50&tag={tag}", f"tag={tag}")

    for industry in JOBICY_INDUSTRIES:
        await _fetch_and_parse(
            f"https://jobicy.com/api/v2/remote-jobs?count=50&industry={industry}",
            f"industry={industry}",
        )

    return records


# ---------------------------------------------------------------------------
# 3. Arbeitnow
# ---------------------------------------------------------------------------


async def fetch_arbeitnow_jobs(client: AsyncHttpClient, max_pages: int = 3) -> list[JobRecord]:
    """
    Arbeitnow has no tag/keyword server-side filter param documented
    (per theirstack.com's ATS API survey and the arbeitnow.com/blog
    post itself, which only documents `visa_sponsorship` and `page`),
    so we paginate and filter client-side by AI keyword match against
    title + tags. Field names (title, company_name, tags, remote,
    url, created_at) per the widely-mirrored Arbeitnow schema
    referenced across the ever-jobs aggregator and job-board-app repo.
    """
    records: list[JobRecord] = []
    reference_time = datetime.now(timezone.utc)

    for page in range(1, max_pages + 1):
        url = f"https://arbeitnow.com/api/job-board-api?page={page}"
        try:
            data = await client.get_json(url)
        except Exception as e:
            logger.error("Arbeitnow fetch failed for page=%d: %s", page, e)
            break

        jobs = data.get("data", [])
        if not jobs:
            break
        logger.info("Arbeitnow page=%d -> %d jobs", page, len(jobs))

        for job in jobs:
            title = job.get("title", "")
            tags = job.get("tags", []) or []
            description = job.get("description", "")
            if not _matches_ai_keywords(title, " ".join(tags), description[:500]):
                continue

            raw_date = job.get("created_at")  # Arbeitnow uses a unix timestamp here
            normalized = None
            if raw_date is not None:
                try:
                    normalized = datetime.fromtimestamp(
                        int(raw_date), tz=timezone.utc
                    ).isoformat()
                except (ValueError, TypeError, OSError):
                    normalized = normalize_date(str(raw_date), reference_time=reference_time)

            content = JobContent(
                company=job.get("company_name", "Unknown"),
                title=title,
                date=normalized,
                is_remote=bool(job.get("remote", False)),
                role_family=_infer_role_family(title),
                location=job.get("location"),
                url=job.get("url"),
            )
            records.append(
                JobRecord(
                    source=Source(name="Arbeitnow", url=job.get("url", "https://arbeitnow.com")),
                    content=content,
                )
            )

    return records


# ---------------------------------------------------------------------------
# 4. We Work Remotely (RSS, programming category, keyword-filtered)
# ---------------------------------------------------------------------------

WWR_FEED_URL = "https://weworkremotely.com/categories/remote-programming-jobs.rss"


async def fetch_wwr_jobs(client: AsyncHttpClient) -> list[JobRecord]:
    """
    WWR's RSS has no AI-specific category (confirmed against their
    published category list — see module docstring), so we pull the
    broader Programming category feed and filter client-side by AI
    keyword match against title + summary, same pattern as
    Arbeitnow above.
    """
    records: list[JobRecord] = []
    reference_time = datetime.now(timezone.utc)

    try:
        xml_text = await client.get_text(WWR_FEED_URL)
    except Exception as e:
        logger.error("WWR fetch failed: %s", e)
        return records

    parsed = feedparser.parse(xml_text)
    logger.info("WWR programming feed -> %d items", len(parsed.entries))

    for entry in parsed.entries:
        title = entry.get("title", "")
        summary_html = entry.get("summary", "")
        summary_text = (
            BeautifulSoup(summary_html, "lxml").get_text(" ", strip=True) if summary_html else ""
        )
        if not _matches_ai_keywords(title, summary_text[:500]):
            continue

        raw_date = entry.get("published")
        normalized = normalize_date(raw_date, reference_time=reference_time)

        # WWR RSS titles are conventionally "Company: Job Title"
        company = "Unknown"
        job_title = title
        if ":" in title:
            parts = title.split(":", 1)
            company, job_title = parts[0].strip(), parts[1].strip()

        content = JobContent(
            company=company,
            title=job_title,
            date=normalized,
            is_remote=True,  # WWR is fully-remote-only by definition
            role_family=_infer_role_family(job_title),
            location="Remote",
            url=entry.get("link"),
        )
        records.append(
            JobRecord(
                source=Source(name="We Work Remotely", url=entry.get("link", WWR_FEED_URL)),
                content=content,
            )
        )

    return records


# ---------------------------------------------------------------------------
# 5. Himalayas
# ---------------------------------------------------------------------------


async def fetch_himalayas_jobs(client: AsyncHttpClient, max_offset_pages: int = 3) -> list[JobRecord]:
    """
    Himalayas' public API (himalayas.app/jobs/api) is documented as
    offset-paginated with a max of 20 results per page (per the
    ever-jobs aggregator's source survey — this pipeline has not
    independently re-verified Himalayas' exact query-param names for
    offset/limit beyond that secondary source, so this is the
    lowest-confidence source in this file; confirm param names against
    a live response before depending on pagination working as written
    here). Filtered client-side by AI keyword, same as Arbeitnow/WWR.
    """
    records: list[JobRecord] = []
    reference_time = datetime.now(timezone.utc)
    page_size = 20

    for page in range(max_offset_pages):
        offset = page * page_size
        url = f"https://himalayas.app/jobs/api?limit={page_size}&offset={offset}"
        try:
            data = await client.get_json(url)
        except Exception as e:
            logger.error("Himalayas fetch failed at offset=%d: %s", offset, e)
            break

        jobs = data.get("jobs", data if isinstance(data, list) else [])
        if not jobs:
            break
        logger.info("Himalayas offset=%d -> %d jobs", offset, len(jobs))

        for job in jobs:
            title = job.get("title", "")
            description = job.get("description", "") or ""
            if not _matches_ai_keywords(title, description[:500]):
                continue

            raw_date = job.get("pubDate") or job.get("publishedAt") or job.get("createdAt")
            normalized = normalize_date(str(raw_date) if raw_date else None, reference_time=reference_time)

            content = JobContent(
                company=job.get("companyName", job.get("company", "Unknown")),
                title=title,
                date=normalized,
                is_remote=True,
                role_family=_infer_role_family(title),
                location=job.get("location", "Remote"),
                url=job.get("applicationLink", job.get("url")),
            )
            records.append(
                JobRecord(
                    source=Source(name="Himalayas", url=job.get("applicationLink", "https://himalayas.app")),
                    content=content,
                )
            )

    return records


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def fetch_all_ai_jobs(client: AsyncHttpClient) -> list[JobRecord]:
    """
    Runs all 5 sources. Each source's failure is isolated (logged,
    not raised) so one board being down doesn't kill the others —
    consistent with the production-readiness requirement. Returns the
    combined list; freshness (<24h) filtering is applied by the
    caller (pipeline runner) using the same is_within_last_24h()
    helper used for news, so both freshness policies stay identical
    and in one place.
    """
    import asyncio

    results = await asyncio.gather(
        fetch_remoteok_jobs(client),
        fetch_jobicy_jobs(client),
        fetch_arbeitnow_jobs(client),
        fetch_wwr_jobs(client),
        fetch_himalayas_jobs(client),
        return_exceptions=True,
    )

    all_jobs: list[JobRecord] = []
    source_names = ["RemoteOK", "Jobicy", "Arbeitnow", "We Work Remotely", "Himalayas"]
    for name, result in zip(source_names, results):
        if isinstance(result, Exception):
            logger.error("Source %s failed entirely: %s", name, result)
            continue
        all_jobs.extend(result)

    logger.info("Collected %d total AI-relevant job records across 5 boards", len(all_jobs))
    return all_jobs
