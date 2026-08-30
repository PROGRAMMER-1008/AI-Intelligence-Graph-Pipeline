"""
Canonical output schemas for the Intelligence Graph pipeline.

These map 1:1 onto the schemas specified in the assignment brief.
Every record produced by any scraper/extractor MUST pass through one
of these Pydantic models before it is allowed to reach storage. This
is the single choke point that guarantees schema conformance across
five very different source types (JSON APIs, RSS, HTML, LLM output).

Design note: we version the schema (`schema_version`) from day one.
Real ingestion pipelines live for years and the schema WILL change;
baking in a version field now means future consumers can branch on
it instead of guessing from field presence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


SCHEMA_VERSION = "1.0"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RecordType(str, Enum):
    STARTUP = "STARTUP"
    PRODUCT = "PRODUCT"
    RESEARCH_PAPER = "RESEARCH_PAPER"
    JOB = "JOB"
    NEWS = "NEWS"


class PricingModel(str, Enum):
    FREE = "FREE"
    FREEMIUM = "FREEMIUM"
    PAID = "PAID"
    ENTERPRISE = "ENTERPRISE"
    UNKNOWN = "UNKNOWN"  # not in spec, but real data is often ambiguous —
    # explicit UNKNOWN beats silently guessing FREE/PAID and polluting
    # downstream analytics with false precision.


class Source(BaseModel):
    name: str
    url: str


class StartupContent(BaseModel):
    entity_name: str = Field(..., description="Canonical startup name")
    raw_name: str = Field(..., description="Name as it appeared at the source, pre-resolution")
    one_liner: Optional[str] = None
    description: Optional[str] = None
    employee_count: Optional[int] = None
    website: Optional[str] = None
    location: Optional[str] = None
    industries: list[str] = Field(default_factory=list)
    founded_or_launched_at: Optional[str] = None  # ISO-8601, when known
    status: Optional[str] = None  # Active / Acquired / Inactive etc.
    is_hiring: Optional[bool] = None


class StartupRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.STARTUP
    source: Source
    content: StartupContent
    collected_at: str = Field(default_factory=utc_now_iso)


class ProductContent(BaseModel):
    startup_name: str = Field(..., description="Canonical startup name")
    product_name: Optional[str] = None
    pricing_model: PricingModel = PricingModel.UNKNOWN
    description: Optional[str] = None
    website: Optional[str] = None
    category: Optional[str] = None


class ProductRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.PRODUCT
    source: Source
    content: ProductContent
    collected_at: str = Field(default_factory=utc_now_iso)


class ResearchPaperContent(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    paper_url: str
    github_url: Optional[str] = None
    github_stars: Optional[int] = None
    published_date: Optional[str] = None  # ISO-8601
    abstract: Optional[str] = None
    arxiv_id: Optional[str] = None
    categories: list[str] = Field(default_factory=list)


class ResearchPaperRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.RESEARCH_PAPER
    source: Source
    content: ResearchPaperContent
    collected_at: str = Field(default_factory=utc_now_iso)


class JobContent(BaseModel):
    company: str
    title: Optional[str] = None
    date: Optional[str] = None  # ISO-8601 publication date
    is_remote: Optional[bool] = None
    role_family: Optional[str] = None
    location: Optional[str] = None
    url: Optional[str] = None


class JobRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.JOB
    source: Source
    content: JobContent
    collected_at: str = Field(default_factory=utc_now_iso)


class NewsContent(BaseModel):
    title: str
    url: str
    published_date: Optional[str] = None  # ISO-8601, normalized
    full_text: Optional[str] = None
    summary: Optional[str] = None
    author: Optional[str] = None


class NewsRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    record_type: RecordType = RecordType.NEWS
    source: Source
    content: NewsContent
    collected_at: str = Field(default_factory=utc_now_iso)


class EntityMappingLogEntry(BaseModel):
    """One row of the Entity Mapping Log deliverable: raw string -> canonical form."""
    raw_name: str
    canonical_name: str
    match_method: str  # exact | fuzzy | seed_list | unresolved
    confidence: float
    record_type: str  # STARTUP | PRODUCT
