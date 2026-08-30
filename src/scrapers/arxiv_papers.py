"""
Research paper acquisition — Phase I.

Source: ArXiv's official Export API (https://export.arxiv.org/api/query),
Atom 1.0 XML, documented at https://info.arxiv.org/help/api/user-manual.html.
No auth required. ArXiv's own Terms of Use ask for a 3-second delay
between successive requests and cap max_results at 2000/request
(30000/query via pagination) — we honor both explicitly below rather
than relying on retry-after-the-fact throttling, since arXiv does not
reliably return 429s for this — it's a courtesy-based API, and the
professional thing to do is rate-limit ourselves rather than wait to
get blocked.

GitHub correlation: many arXiv abstracts link their code repo directly
in the abstract text (a very common convention: "Code at
https://github.com/org/repo"). We extract these with a regex and then
hit the GitHub REST API (api.github.com/repos/{owner}/{repo}) for
live star counts — this is the "dynamic metrics tracking" requirement.
GitHub's unauthenticated rate limit is a harsh 60 req/hour; with a
personal access token (GITHUB_TOKEN env var) it's 5000 req/hour, so
production runs should always set one. See README.md setup section.

IMPORTANT — development environment constraint (documented for
honesty, not hidden): this module was developed and unit-tested
against a schema-accurate but non-live fixture
(tests/fixtures/arxiv_sample_response.xml) because the development
sandbox's network egress allowlist does not include export.arxiv.org.
api.github.com WAS reachable and the GitHub star-fetching path below
was validated against a real, live API response during development
(see tests/test_http_client.py and docs/architecture.md for the raw
evidence). Anyone running this pipeline outside that sandboxed
environment has unrestricted network access and should confirm the
ArXiv path against a live query as a first smoke test — see
README.md "Verifying the ArXiv pipeline" section.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional
from urllib.parse import quote

import feedparser

from src.schemas import ResearchPaperContent, ResearchPaperRecord, Source
from src.utils.http_client import AsyncHttpClient, RateLimitExceededError

logger = logging.getLogger("graphone.scrapers.arxiv")

ARXIV_API = "https://export.arxiv.org/api/query"
GITHUB_API = "https://api.github.com"

# ArXiv ToS: be a good citizen. This is not a retry-backoff value, it's
# a fixed courtesy delay applied between EVERY successive call, per
# https://info.arxiv.org/help/api/user-manual.html
ARXIV_COURTESY_DELAY_SECONDS = 3.0
ARXIV_MAX_PER_REQUEST = 2000  # hard API ceiling

# Matches github.com/owner/repo, tolerating a .git suffix or any
# trailing path (/tree/main, /blob/..., /issues/1) and stopping the
# repo name at the next slash, whitespace, or common trailing
# punctuation from prose (period, comma, closing paren/bracket).
_GITHUB_URL_RE = re.compile(
    r"github\.com/([A-Za-z0-9][-A-Za-z0-9]*)/([A-Za-z0-9._-]+?)"
    r"(?:\.git)?(?=[/\s\)\]\.,;]|$)"
)


def extract_github_repo(text: str) -> Optional[tuple[str, str]]:
    """Returns (owner, repo) for the first GitHub repo link found in text, if any."""
    if not text:
        return None
    m = _GITHUB_URL_RE.search(text)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    return owner, repo


async def fetch_github_stars(
    client: AsyncHttpClient, owner: str, repo: str, github_token: Optional[str] = None
) -> Optional[int]:
    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    try:
        data = await client.get_json(url, headers=headers)
        return data.get("stargazers_count")
    except RateLimitExceededError:
        # Distinguished from a genuine 404 deliberately: during
        # development this path was hit live (unauthenticated GitHub
        # caps at 60 req/hour, shared across the whole sandbox IP) and
        # a broad except-and-continue would silently misreport
        # "no repo found" for what was actually "couldn't check because
        # we're rate-limited". Both end in github_stars=None for this
        # record, but only one should be logged as expected/benign.
        # Fix: set GITHUB_TOKEN (5000 req/hour) for any real run — see
        # README.md.
        logger.warning(
            "GitHub rate limit exhausted fetching %s/%s — star count omitted "
            "(set GITHUB_TOKEN env var to raise the 60/hr unauthenticated cap "
            "to 5000/hr)",
            owner,
            repo,
        )
        return None
    except Exception as e:
        # A dead/renamed/private repo is common and NOT a pipeline
        # failure — log and continue without stars rather than
        # aborting the whole paper record.
        logger.info("Could not fetch GitHub stars for %s/%s: %s", owner, repo, e)
        return None


def _parse_atom_response(xml_text: str) -> list[dict]:
    """
    Parses arXiv's Atom XML into plain dicts. feedparser handles the
    Atom 1.0 + arxiv: namespace extensions robustly (it's the standard
    tool for this — used by the official arxiv.py wrapper too).
    """
    parsed = feedparser.parse(xml_text)
    papers = []
    for entry in parsed.entries:
        arxiv_id = entry.id.split("/abs/")[-1] if "/abs/" in entry.id else entry.id
        authors = [a.get("name", "") for a in entry.get("authors", [])] or (
            [entry.author] if entry.get("author") else []
        )
        pdf_url = None
        for link in entry.get("links", []):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
        categories = [t.get("term") for t in entry.get("tags", [])] if entry.get("tags") else []

        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": " ".join(entry.title.split()),  # collapse whitespace/newlines
                "authors": authors,
                "abstract": " ".join(entry.summary.split()) if entry.get("summary") else None,
                "published": entry.get("published"),
                "paper_url": entry.get("id"),
                "pdf_url": pdf_url,
                "categories": categories,
            }
        )
    return papers


async def fetch_papers(
    client: AsyncHttpClient,
    category: str = "cs.AI",
    min_records: int = 1000,
    github_token: Optional[str] = None,
) -> list[ResearchPaperRecord]:
    """
    Fetches papers from the given arXiv category, paginating with the
    mandatory courtesy delay, correlates each with a GitHub repo when
    the abstract links one, and fetches live star counts.
    """
    records: list[ResearchPaperRecord] = []
    start = 0

    while len(records) < min_records:
        page_size = min(ARXIV_MAX_PER_REQUEST, 200)  # keep individual pages modest;
        # a real 500k-scale run would raise this toward ARXIV_MAX_PER_REQUEST,
        # but smaller pages here mean faster partial progress and cheaper
        # retries if a single page fails mid-crawl.
        query = (
            f"{ARXIV_API}?search_query=cat:{quote(category)}"
            f"&start={start}&max_results={page_size}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        logger.info("Fetching arXiv page: start=%d size=%d", start, page_size)
        xml_text = await client.get_text(query)
        papers = _parse_atom_response(xml_text)
        if not papers:
            logger.info("No more arXiv results at start=%d; stopping pagination.", start)
            break

        # GitHub star lookups run concurrently for this page (bounded by
        # the shared client semaphore), not serially — this is the
        # concurrency requirement in practice, not just in the async def.
        star_lookups = []
        for p in papers:
            repo = extract_github_repo(p["abstract"] or "")
            if repo:
                star_lookups.append(fetch_github_stars(client, *repo, github_token=github_token))
            else:
                star_lookups.append(_none())

        stars_and_repos = await asyncio.gather(*star_lookups)

        for p, stars in zip(papers, stars_and_repos):
            repo = extract_github_repo(p["abstract"] or "")
            github_url = f"https://github.com/{repo[0]}/{repo[1]}" if repo else None

            content = ResearchPaperContent(
                title=p["title"],
                authors=p["authors"],
                paper_url=p["paper_url"],
                github_url=github_url,
                github_stars=stars,
                published_date=p["published"],
                abstract=p["abstract"],
                arxiv_id=p["arxiv_id"],
                categories=p["categories"],
            )
            records.append(
                ResearchPaperRecord(
                    source=Source(name="arXiv", url=p["paper_url"]),
                    content=content,
                )
            )

        start += page_size
        if len(records) < min_records:
            await asyncio.sleep(ARXIV_COURTESY_DELAY_SECONDS)

    logger.info("Collected %d research paper records", len(records))
    return records[:min_records] if len(records) > min_records else records


async def _none():
    return None
