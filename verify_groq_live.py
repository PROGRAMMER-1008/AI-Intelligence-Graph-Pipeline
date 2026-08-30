"""
Run this YOURSELF, on your own machine, once you have a Groq API key.

The development sandbox used to build this pipeline has a restricted
network egress allowlist that does NOT include api.groq.com (confirmed:
it returns HTTP 403 with header `x-deny-reason: host_not_allowed`,
which is the sandbox's proxy rejecting the connection before it ever
reaches Groq — not a code bug and not an auth failure). So this exact
call could not be executed end-to-end during development. Every other
piece of this file's dependencies (chunking, tier fallback, JSON
parsing) IS unit tested against real/live data elsewhere in tests/ —
this script closes the one remaining gap.

Usage (either works — both reach the same os.environ variable):
    # Option A: shell export
    export GROQ_API_KEY="your-key-here"
    python3 verify_groq_live.py

    # Option B: .env file (cp .env.example .env, fill in the key)
    python3 verify_groq_live.py

Expected output: a real JSON object extracted by Llama 3.3 70B via
Groq, proving the GroqTier class in src/llm/orchestrator.py works
against the live API, not just against mocks.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()  # loads .env from the current directory if present, so
# GROQ_API_KEY works whether it's set via `export` in your shell or
# via a .env file — either path reaches os.environ the same way.

from src.llm.orchestrator import GroqTier, LLMOrchestrator
from src.utils.http_client import AsyncHttpClient


SAMPLE_TEXT = """
Nanonets is a San Francisco-based startup founded to solve automatic
data extraction. The company has grown to over 100 employees and was
part of Y Combinator's Winter 2017 batch. Nanonets builds AI agents
that extract structured data from invoices, purchase orders, and
clinical documents, with a focus on traceability -- every extraction
can be traced back to exactly what the agent read and why it made the
decision it did.
"""

SCHEMA_HINT = """
{
  "entity_name": "string - the company name",
  "employee_count": "integer or null - approximate headcount if mentioned",
  "yc_batch": "string or null - YC batch if mentioned, e.g. 'Winter 2017'",
  "one_liner": "string - a one-sentence description of what they do"
}
"""


async def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY environment variable first.")
        print('  export GROQ_API_KEY="your-key-here"')
        sys.exit(1)

    orchestrator = LLMOrchestrator(tiers=[GroqTier(api_key=api_key)])

    print(f"Available tiers: {[t.name for t in orchestrator.available_tiers()]}")
    print("Sending live extraction request to Groq...\n")

    async with AsyncHttpClient() as client:
        result = await orchestrator.extract(client, SAMPLE_TEXT, SCHEMA_HINT)

    print(f"Success: {result.success}")
    print(f"Tier used: {result.tier_used}")
    if result.success:
        import json

        print("Extracted data:")
        print(json.dumps(result.data, indent=2))
        print("\n✅ Live Groq integration confirmed working.")
    else:
        print(f"Error: {result.error}")
        print("\n❌ Something is wrong — check your API key and Groq account status.")


if __name__ == "__main__":
    asyncio.run(main())
