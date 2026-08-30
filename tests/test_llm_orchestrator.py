"""
Tests for the LLM orchestrator (src/llm/orchestrator.py): chunking
correctness (no content loss, budget respected) and fallback-chain
tier-availability logic (skip tiers with no configured key).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.orchestrator import (
    DeepSeekTier,
    GeminiFlashTier,
    GroqTier,
    LLMOrchestrator,
    chunk_text,
    estimate_tokens,
)


def test_short_text_stays_single_chunk():
    text = "This is a short paragraph.\n\nAnother short one."
    chunks = chunk_text(text, max_tokens=1000)
    assert len(chunks) == 1


def test_long_text_splits_within_budget():
    paragraphs = [
        f"This is paragraph number {i} with padding text to make it realistic."
        for i in range(50)
    ]
    long_text = "\n\n".join(paragraphs)
    chunks = chunk_text(long_text, max_tokens=500)
    assert len(chunks) > 1
    for c in chunks:
        assert estimate_tokens(c) <= 500


def test_no_content_lost_across_chunks():
    paragraphs = [f"This is paragraph number {i} unique marker." for i in range(50)]
    long_text = "\n\n".join(paragraphs)
    chunks = chunk_text(long_text, max_tokens=500)
    all_text = " ".join(chunks)
    for i in range(50):
        assert f"paragraph number {i} " in all_text


def test_pathological_giant_sentence_does_not_crash():
    giant = "word " * 5000  # no sentence-ending punctuation at all
    chunks = chunk_text(giant, max_tokens=200)
    assert len(chunks) >= 1  # must not raise


def test_no_tiers_configured_returns_clean_failure():
    orch = LLMOrchestrator(
        tiers=[GroqTier(api_key=None), GeminiFlashTier(api_key=None), DeepSeekTier(api_key=None)]
    )
    assert orch.available_tiers() == []

    async def run():
        return await orch.extract(None, "text", "schema")

    result = asyncio.run(run())
    assert result.success is False
    assert "No LLM tiers configured" in result.error


def test_only_configured_tier_is_available():
    orch = LLMOrchestrator(
        tiers=[
            GroqTier(api_key="fake-test-key"),
            GeminiFlashTier(api_key=None),
            DeepSeekTier(api_key=None),
        ]
    )
    available = [t.name for t in orch.available_tiers()]
    assert available == ["groq-llama3"]


def test_tier_order_is_preserved_when_multiple_available():
    orch = LLMOrchestrator(
        tiers=[
            GroqTier(api_key="k1"),
            GeminiFlashTier(api_key="k2"),
            DeepSeekTier(api_key="k3"),
        ]
    )
    available = [t.name for t in orch.available_tiers()]
    assert available == ["groq-llama3", "gemini-flash", "deepseek"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
