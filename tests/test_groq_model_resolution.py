"""
Tests for GroqTier.resolve_model() — added after a real user-reported
failure: the pipeline's original default model, llama-3.3-70b-versatile,
was deprecated by Groq on 2026-06-17 (after this pipeline was
originally written against it), causing every live extraction call to
fail with 404 Not Found. See GroqTier's class docstring in
src/llm/orchestrator.py for the full incident writeup.

resolve_model() queries Groq's live /openai/v1/models endpoint and
self-corrects if the configured model is no longer active, rather
than hardcoding a model name that will inevitably go stale again.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.orchestrator import GroqTier


def test_configured_model_confirmed_active():
    async def run():
        tier = GroqTier(api_key="fake")
        mock_client = MagicMock()
        mock_client.get_json = AsyncMock(
            return_value={
                "data": [
                    {"id": "openai/gpt-oss-120b", "active": True},
                    {"id": "qwen/qwen3.6-27b", "active": True},
                ]
            }
        )
        return await tier.resolve_model(mock_client)

    resolved = asyncio.run(run())
    assert resolved == "openai/gpt-oss-120b"


def test_stale_model_falls_back_to_live_gpt_oss():
    """
    Regression test for the exact deprecated-model scenario: a
    hardcoded/configured model that is no longer in Groq's active
    list should NOT be sent as-is (it would 404) — it should resolve
    to a currently active model instead.
    """

    async def run():
        tier = GroqTier(api_key="fake", model="llama-3.3-70b-versatile")
        mock_client = MagicMock()
        mock_client.get_json = AsyncMock(
            return_value={
                "data": [
                    {"id": "openai/gpt-oss-120b", "active": True},
                    {"id": "qwen/qwen3.6-27b", "active": True},
                ]
            }
        )
        return await tier.resolve_model(mock_client)

    resolved = asyncio.run(run())
    assert resolved == "openai/gpt-oss-120b"
    assert resolved != "llama-3.3-70b-versatile"


def test_models_endpoint_failure_falls_back_safely():
    """
    If the /models lookup itself fails (network issue, auth issue,
    etc.), resolve_model must not crash — it should proceed with
    whatever model was configured, exactly as before this feature
    existed, rather than blocking extraction entirely.
    """

    async def run():
        tier = GroqTier(api_key="fake", model="some-configured-model")
        mock_client = MagicMock()
        mock_client.get_json = AsyncMock(side_effect=Exception("network error"))
        return await tier.resolve_model(mock_client)

    resolved = asyncio.run(run())
    assert resolved == "some-configured-model"


def test_default_model_is_current_groq_recommendation():
    """
    Locks in that DEFAULT_MODEL was updated away from the deprecated
    llama-3.3-70b-versatile. This test will need updating if Groq
    deprecates openai/gpt-oss-120b in turn — that's expected and
    healthy; it's a deliberate tripwire, not a bug.
    """
    assert GroqTier.DEFAULT_MODEL == "openai/gpt-oss-120b"
    assert GroqTier.DEFAULT_MODEL != "llama-3.3-70b-versatile"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
