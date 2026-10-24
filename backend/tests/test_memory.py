"""Unit tests for conversation session memory."""

from app.llm.memory import ConversationMemory


def test_memory_turn_sliding_window() -> None:
    """Verify that conversation memory respects sliding window limit."""
    mem = ConversationMemory(max_history_turns=2)
    session_id = "test-session-1"

    # Add 4 turns (8 messages)
    for i in range(1, 5):
        mem.add_turn(session_id, "user", f"User question {i}", intent="question")
        mem.add_turn(session_id, "assistant", f"Assistant answer {i}")

    session = mem.get_or_create_session(session_id)
    # max_history_turns=2 means max 4 messages kept in active list
    assert len(session.messages) <= 4
    # Latest turns preserved
    assert "User question 4" in session.messages[-2].content
    assert "Assistant answer 4" in session.messages[-1].content
    # Older context preserved in rolling summary
    assert "User question" in session.rolling_summary


def test_memory_reset() -> None:
    """Verify resetting session memory clears turns."""
    mem = ConversationMemory(max_history_turns=3)
    session_id = "test-session-reset"

    mem.add_turn(session_id, "user", "Hello", intent="greeting")
    mem.add_turn(session_id, "assistant", "Hi there")

    assert len(mem.get_or_create_session(session_id).messages) == 2
    mem.reset_session(session_id)
    assert len(mem.get_or_create_session(session_id).messages) == 0


def test_context_formatting() -> None:
    """Verify context string formatting."""
    mem = ConversationMemory()
    session_id = "test-context"

    formatted_empty = mem.get_context_formatted(session_id)
    assert "No previous conversation" in formatted_empty

    mem.add_turn(session_id, "user", "Need doctor", intent="request")
    formatted = mem.get_context_formatted(session_id)
    assert "User (Signer) [Intent: request]: Need doctor" in formatted
