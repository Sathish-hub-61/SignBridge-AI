"""Unit tests for the single-call Compose LLM pipeline (one-way signer output)."""

import json
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.llm.compose import ComposePipeline, _sanitize_word
from app.llm.memory import ConversationMemory
from app.llm.openrouter_client import OpenRouterClient
from app.schemas import ComposeRequest, IntentEnum, RecognizedToken


def test_sanitize_word_injection_neutralization() -> None:
    """Verify that untrusted injection attempts in glosses are neutralized."""
    # Dangerous punctuation and scripts stripped; no truncation for short input
    assert _sanitize_word("IGNORE;DROP") == "IGNOREDROP"
    assert _sanitize_word("<script>alert(1)</script>") == "SCRIPTALERT1SCRIPT"
    assert _sanitize_word("NORMAL_SIGN") == "NORMAL_SIGN"
    assert _sanitize_word("WATER") == "WATER"


def test_compose_pipeline_instantiation() -> None:
    """Verify ComposePipeline instantiates with default settings."""
    settings = Settings(OPENROUTER_API_KEY="")
    pipeline = ComposePipeline(settings=settings)
    assert pipeline is not None


@pytest.mark.asyncio
async def test_offline_compose_fallback() -> None:
    """Verify offline rule-based fallback when API key is missing."""
    settings = Settings(OPENROUTER_API_KEY="")
    pipeline = ComposePipeline(settings=settings)

    tokens = [
        RecognizedToken(token="WATER", confidence=0.9),
        RecognizedToken(token="WANT", confidence=0.85),
    ]
    res = await pipeline.compose(ComposeRequest(tokens=tokens))
    assert res.intent == IntentEnum.REQUEST
    assert "WATER" in res.sentence or "water" in res.sentence.lower()


@pytest.mark.asyncio
async def test_offline_greeting_fallback() -> None:
    """Verify greeting rule when offline."""
    settings = Settings(OPENROUTER_API_KEY="")
    pipeline = ComposePipeline(settings=settings)

    tokens = [RecognizedToken(token="HELLO", confidence=0.95)]
    res = await pipeline.compose(ComposeRequest(tokens=tokens))
    assert "Hello" in res.sentence
    assert res.intent == IntentEnum.GREETING


@pytest.mark.asyncio
async def test_compose_json_repair_retry() -> None:
    """Verify that Compose retries with repair when first attempt returns malformed JSON."""
    settings = Settings(OPENROUTER_API_KEY="sk-or-v1-mock-test-key")
    mock_client = AsyncMock(spec=OpenRouterClient)

    valid_json = json.dumps({
        "sentence": "I need assistance right now.",
        "alternatives": ["Can I get help please?"],
        "intent": "request",
        "tone": "polite",
        "needs_confirmation": False,
        "ambiguity_note": None,
    })
    mock_client.create_completion.side_effect = [
        "THIS IS NOT VALID JSON",
        valid_json,
    ]

    mem = ConversationMemory()
    pipeline = ComposePipeline(client=mock_client, memory=mem, settings=settings)

    tokens = [RecognizedToken(token="HELP", confidence=0.9)]
    res = await pipeline.compose(ComposeRequest(tokens=tokens))

    assert res.sentence == "I need assistance right now."
    assert res.intent == IntentEnum.REQUEST
    assert mock_client.create_completion.call_count == 2


@pytest.mark.asyncio
async def test_compose_needs_confirmation_low_confidence() -> None:
    """Low-confidence glosses should set needs_confirmation=true."""
    settings = Settings(OPENROUTER_API_KEY="")
    pipeline = ComposePipeline(settings=settings)

    tokens = [RecognizedToken(token="UNCLEAR", confidence=0.30)]
    res = await pipeline.compose(ComposeRequest(tokens=tokens))
    assert res.needs_confirmation is True


def test_memory_signer_only_one_way() -> None:
    """Verify memory stores only signer turns (one-way, no assistant replies)."""
    mem = ConversationMemory()
    session_id = "sess-one-way"

    mem.add_signer_turn(session_id, "I need water.", intent="request")
    mem.add_signer_turn(session_id, "Thank you.", intent="thanks")

    history = mem.get_signer_history(session_id)
    assert history == ["I need water.", "Thank you."]

    # No assistant turns should exist
    session = mem.get_or_create_session(session_id)
    assistant_turns = [t for t in session.turns if t.role != "user"]
    assert len(assistant_turns) == 0
