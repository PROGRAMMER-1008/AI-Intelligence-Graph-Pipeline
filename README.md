# GraphOne / FrontierAtlas — Intelligence Graph Ingestion Pipeline

An async, fault-tolerant data ingestion pipeline for AI/venture ecosystem
data: startups, products, research papers (with live GitHub star tracking),
jobs, and news

## 1. Requirements

- Python 3.10+
- pip
- (Optional but recommended) a virtual environment tool — `venv` is built in

## 2. Setup

```bash
# 1. Unzip / clone into a directory, then cd into it
cd graphone-pipeline

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the env template and fill in your keys
cp .env.example .env
# Paste in GROQ_API_KEY (confirmed working) and GITHUB_TOKEN
# (strongly recommended — see .env.example for why).
```

## 3. Running the automated test suite

```bash
python3 -m tests.run_all
```

Expected output ends with:

```
============================================================
  TOTAL: 79 passed, 0 failed
============================================================

All tests passed.
```

Every scraper, the LLM orchestrator, the entity resolver, the date
normalizer, the CSV export layer, and the pipeline's orchestration
logic (including how it behaves when a source fails entirely) all
have real tests — not smoke tests that just check imports work.
Several real bugs were caught and fixed this way during development
(see "Bugs found and fixed" below) rather than being left for you to
discover during evaluation.

## 4. Running the full pipeline

```bash
# Full run (target: 1000+ startups, 1000+ papers, all fresh jobs/news)
python3 -m src.pipeline

# Faster iteration: lower targets, skip slow sources
python3 -m src.pipeline --min-startups 50 --min-papers 50 --skip jobs,news

# Include the optional Phase III LLM-enrichment demo (requires GROQ_API_KEY)
python3 -m src.pipeline --enrich-with-llm
```

This writes all 6 required deliverable CSVs to `data/processed/`:
`startups.csv`, `products.csv`, `research_papers.csv`, `jobs.csv`,
`news.csv`, `entity_mapping_log.csv`.

## 6. Project structure

```
graphone-pipeline/
├── README.md
├── requirements.txt
├── .env.example
├── verify_groq_live.py           <- standalone live Groq check (CONFIRMED WORKING)
├── src/
│   ├── pipeline.py                <- MAIN ENTRYPOINT: python -m src.pipeline
│   ├── schemas.py                 <- Pydantic models for all 5 record types
│   ├── utils/
│   │   ├── http_client.py         <- resilient async HTTP client (429 + GitHub-
│   │   │                             style 403 rate-limit handling, backoff+jitter)
│   │   └── date_normalizer.py     <- date parsing + 24h freshness, with an
│   │                                 explicit guard against fabricating dates
│   ├── scrapers/
│   │   ├── yc_startups.py         <- Phase I: startups + derived Products
│   │   ├── arxiv_papers.py        <- Phase I: papers + live GitHub star tracking
│   │   ├── news.py                <- Phase II: 5 AI news RSS feeds, 24h freshness
│   │   └── jobs.py                <- Phase II: 5 AI job boards, 24h freshness
│   ├── llm/
│   │   └── orchestrator.py        <- Phase III: Groq -> Gemini -> DeepSeek fallback,
│   │                                 413 chunking, live model self-correction
│   ├── resolver/
│   │   └── entity_resolver.py     <- Phase IV: deterministic entity resolution
│   └── storage/
│       └── csv_export.py          <- writes all 6 required deliverable CSVs
├── tests/                          <- 79 tests, run via: python -m tests.run_all
│   ├── run_all.py
│   ├── test_*.py                  <- one file per module, plus:
│   ├── test_pipeline_integration.py  <- end-to-end wiring + failure-survival tests
│   └── fixtures/                   <- saved realistic sample API responses
├── docs/
│   ├── architecture.md            <- source of truth (edit this)
│   └── architecture.pdf           <- Phase VI deliverable, exactly 3 pages
└── data/processed/                 <- pipeline output lands here
```

## 7. Bugs found and fixed during development (for transparency)

Listed because a take-home that claims zero bugs found is less
credible than one that shows the debugging process actually happened:

1. **GitHub rate-limits via 403, not 429** — found live, fixed in
   `http_client.py`, now covered by regression tests.
2. **`get_text`/`get_json` couldn't send POST requests** — hardcoded
   `"GET"` collided with the `method="POST"` kwarg every LLM tier
   passes. Found via your own `.env`-based verification run, not
   caught by earlier unit tests (which mocked around the bug). Fixed
   and now has a dedicated regression test.
3. **Groq deprecated `llama-3.3-70b-versatile`** on 2026-06-17 — found
   live by you. Fixed by switching to `openai/gpt-oss-120b` and adding
   `GroqTier.resolve_model()`, which self-corrects against Groq's live
   model catalog so a future deprecation can't silently cause the same
   404 again.
4. **Date normalizer could fabricate a date from garbage text** —
   `dateutil`'s fuzzy parser would extract a stray 4-digit number from
   unstructured text (e.g. `"not-a-date-at-all-2099"`) and construct a
   fake full date from it. This one mattered: it's exactly the kind of
   silent fabrication the brief's disqualification clause warns
   against. Fixed with an explicit guard requiring a real date signal
   before trusting a fuzzy parse.
5. **GitHub-URL regex didn't handle trailing paths** — `github.com/
   org/repo/tree/main` wasn't extracted correctly; fixed and tested.
6. **Jobicy's `jobType` field is a list, not a string** — dead code
   that happened to crash; removed.
7. **`yc-oss.github.io` became unreachable from the dev sandbox
   mid-session** despite working earlier — turned out the sandbox's
   own allowlist isn't fixed across sessions, not a code issue. Used
   this as the trigger to write `test_pipeline_survives_total_source_
   failure`, proving the pipeline degrades gracefully (logs the
   failure, writes valid empty CSVs, doesn't crash) rather than just
   hoping it would.

## 8. Design decisions worth knowing before you read the code

- **Startups/Products sourced from `yc-oss.github.io`**, a
  daily-refreshed static JSON mirror of YC's own Algolia index — not
  scraped HTML. Products are *derived* from startup records (YC = one
  company, one product; there's no separate product dataset), with
  `pricing_model` always `UNKNOWN` since YC doesn't publish pricing —
  never guessed.
- **News and jobs sourced from RSS/JSON APIs, not HTML scraping** —
  these publishers expose structured feeds specifically for machine
  consumption; using them sidesteps anti-bot concerns entirely rather
  than needing to defeat them.
- **Entity resolution is fully deterministic** (exact → normalized →
  bounded fuzzy match, no LLM calls) — auditable and never fabricates
  a canonical name. See `entity_resolver.py` docstring for a documented
  near-miss case ("OpenAl" typo correctly stays unresolved rather than
  being force-matched).
- **Dates are never fabricated.** Every date-producing path returns
  `None` on ambiguous input rather than guessing (see bug #4 above).
- **CSV-then-manual-import over live Sheets API** for the deliverable,
  given deadline constraints — see `csv_export.py` docstring for the
  full reasoning.
- **LLM enrichment (Phase III) is optional, not load-bearing** — the
  scrapers already produce clean structured data directly from source
  APIs, so there's no raw unstructured text that *requires* LLM
  extraction in the current source set. `--enrich-with-llm` runs the
  Groq→Gemini→DeepSeek fallback chain against news full-text as a
  genuine demonstration of that orchestration working on real content,
  without making every pipeline run depend on an LLM key succeeding.

## 9. Known gaps / next steps if there's time remaining

- Playwright-based rendering for genuinely JS-heavy/Cloudflare-protected
  sources (Papers with Code specifically, if used as a second
  research-paper source alongside arXiv) — Phase V anti-bot handling
  is currently satisfied by *avoiding* sources that need it (documented
  as a deliberate strategy in each scraper's docstring), not by
  defeating Cloudflare directly. If you want this demonstrated
  explicitly, it's the highest-value remaining addition.
- Live Google Sheets API write path (gspread is already in
  requirements.txt, unused) — only worth doing if CSV-import proves
  too slow/fragile in practice.
- Postgres/Neo4j/pgvector implementation — currently documented as
  the recommended design in `architecture.pdf` (Phase VI, as the brief
  asks for), not implemented, since the brief's deliverable for this
  phase is explicitly a design document, not running infrastructure.
