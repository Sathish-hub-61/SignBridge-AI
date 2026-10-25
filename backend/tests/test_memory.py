"""Unit tests for conversation session memory (one-way signer output)."""

from app.llm.memory import ConversationMemory


def test_memory_turn_sliding_window() -> None:
    """Verify that conversation memory respects sliding window limit."""
    mem = ConversationMemory(max_history_turns=2)
    session_id = "test-session-1"

    # Add 5 signer turns (window holds 4, so 1 overflows into rolling summary)
    for i in range(1, 6):
        mem.add_signer_turn(session_id, f"Signer sentence {i}", intent="statement")

    session = mem.get_or_create_session(session_id)
    # max_history_turns=2 means max 4 turns kept; turn 1 overflowed into summary
    assert len(session.turns) <= 4
    # Latest turns preserved
    assert "Signer sentence 5" in session.turns[-1].content
    assert "Signer sentence 4" in session.turns[-2].content
    # Older context preserved in rolling summary
    assert "Signer sentence 1" in session.rolling_summary


def test_memory_reset() -> None:
    """Verify resetting session memory clears turns."""
    mem = ConversationMemory(max_history_turns=3)
    session_id = "test-session-reset"

    mem.add_signer_turn(session_id, "Hello", intent="greeting")

    assert len(mem.get_or_create_session(session_id).turns) == 1
    mem.reset_session(session_id)
    assert len(mem.get_or_create_session(session_id).turns) == 0


def test_context_formatting() -> None:
    """Verify context string formatting (signer-only, one-way)."""
    mem = ConversationMemory()
    session_id = "test-context"

    formatted_empty = mem.get_context_formatted(session_id)
    assert "No previous conversation" in formatted_empty

    mem.add_signer_turn(session_id, "Need doctor", intent="request")
    formatted = mem.get_context_formatted(session_id)
    assert "Signer [Intent: request]: Need doctor" in formatted
