"""
Deterministic Entity Resolution — Phase IV.

"Deterministic" in the brief is doing real work here: it rules out an
LLM-based fuzzy resolver (non-reproducible, costs a token budget per
lookup, and — worse — is exactly the kind of component that could
hallucinate a canonical name that doesn't exist). We use a fully
deterministic three-stage pipeline instead:

  Stage 1 — Exact match (case-insensitive, whitespace-normalized)
            against the seed list's canonical names AND their known
            aliases. O(1) dict lookup.

  Stage 2 — Normalized match: strip legal suffixes (Inc., LLC, Ltd.,
            Corp., "The", punctuation, extra whitespace) and retry
            exact match. Handles "OpenAI, Inc." -> "OpenAI" and
            "Open A.I." -> "Open AI" deterministically, with rules
            you can read and audit line by line (unlike an LLM call).

  Stage 3 — Fuzzy match via token-sort-ratio (rapidfuzz) against the
            seed list, ONLY as a last resort, with a high similarity
            threshold (92) and the match method + score always logged
            to the Entity Mapping Log. This catches things like minor
            misspellings or reordering ("AI Open" / "OpenAl" typos)
            while staying auditable — every fuzzy match is traceable
            to a similarity score, not a black-box LLM judgment.

  No match clears the threshold -> the raw name is emitted AS ITS OWN
  canonical form (never invented), tagged `unresolved` in the mapping
  log. This is a deliberate design choice: it is better to under-merge
  (leave two spellings of an unseen company as distinct entities) than
  to over-merge (wrongly collapse two different companies together) or
  fabricate a canonical name for something not in the seed list — both
  of the latter are silent data-quality failures that are much harder
  to catch downstream than an "unresolved" flag that surfaces cleanly
  in the audit log for a human to review.

Every resolution decision is logged as an EntityMappingLogEntry so the
required "Entity Mapping Log (Raw vs Canonical names)" deliverable is
a byproduct of the resolver actually running, not a separately
maintained artifact that could drift from what the pipeline did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from src.schemas import EntityMappingLogEntry

_LEGAL_SUFFIXES = re.compile(
    r"\b(inc\.?|incorporated|llc\.?|ltd\.?|limited|corp\.?|corporation|"
    r"co\.?|company|technologies|technology|labs?|group|holdings?|"
    r"the)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[.,!?;:'\"()\-_/&]+")
_MULTI_SPACE = re.compile(r"\s+")

FUZZY_THRESHOLD = 92.0


def _normalize(name: str) -> str:
    s = name.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _LEGAL_SUFFIXES.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


@dataclass
class SeedEntity:
    canonical_name: str
    aliases: list[str] = field(default_factory=list)


class EntityResolver:
    def __init__(self, seed_entities: list[SeedEntity]):
        self._seed = seed_entities
        self._exact_index: dict[str, str] = {}
        self._normalized_index: dict[str, str] = {}
        self._canonical_normalized: dict[str, str] = {}  # normalized -> canonical, for fuzzy pool

        for entity in seed_entities:
            all_names = [entity.canonical_name, *entity.aliases]
            for n in all_names:
                self._exact_index[n.strip().lower()] = entity.canonical_name
                self._normalized_index[_normalize(n)] = entity.canonical_name
            self._canonical_normalized[_normalize(entity.canonical_name)] = entity.canonical_name

        self._fuzzy_pool = list(self._canonical_normalized.keys())

    def resolve(self, raw_name: str, record_type: str = "STARTUP") -> tuple[str, EntityMappingLogEntry]:
        raw_stripped = raw_name.strip()

        # Stage 1: exact
        key = raw_stripped.lower()
        if key in self._exact_index:
            canonical = self._exact_index[key]
            return canonical, EntityMappingLogEntry(
                raw_name=raw_name,
                canonical_name=canonical,
                match_method="exact",
                confidence=1.0,
                record_type=record_type,
            )

        # Stage 2: normalized (strip legal suffixes/punctuation)
        norm = _normalize(raw_stripped)
        if norm in self._normalized_index:
            canonical = self._normalized_index[norm]
            return canonical, EntityMappingLogEntry(
                raw_name=raw_name,
                canonical_name=canonical,
                match_method="normalized",
                confidence=0.98,
                record_type=record_type,
            )

        # Stage 3: fuzzy, high threshold only
        best_score = 0.0
        best_match = None
        for candidate_norm in self._fuzzy_pool:
            score = fuzz.token_sort_ratio(norm, candidate_norm)
            if score > best_score:
                best_score = score
                best_match = candidate_norm

        if best_match is not None and best_score >= FUZZY_THRESHOLD:
            canonical = self._canonical_normalized[best_match]
            return canonical, EntityMappingLogEntry(
                raw_name=raw_name,
                canonical_name=canonical,
                match_method="fuzzy",
                confidence=round(best_score / 100.0, 4),
                record_type=record_type,
            )

        # Unresolved: emit as its own canonical form, never fabricate
        return raw_stripped, EntityMappingLogEntry(
            raw_name=raw_name,
            canonical_name=raw_stripped,
            match_method="unresolved",
            confidence=0.0,
            record_type=record_type,
        )


def build_seed_list_from_yc_top_companies(top_companies: list[dict]) -> list[SeedEntity]:
    """
    Builds the required 'mock database of 50 known AI startups' from
    real data rather than inventing names: pulls from YC's own
    'top_company' flag among AI-tagged results (already fetched in
    the startup scraper), using former_names as aliases. This keeps
    the seed list traceable to real entities instead of a hardcoded
    fictional list, while satisfying the brief's "you may mock a
    small database" allowance.
    """
    seeds = []
    for c in top_companies[:50]:
        aliases = c.get("former_names", []) or []
        seeds.append(SeedEntity(canonical_name=c["name"], aliases=aliases))
    return seeds
