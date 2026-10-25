"""Conversation session memory and context management for SignBridge AI.

One-way memory: stores only the signer's composed sentences (no assistant turns,
no audio, no video).
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger("signbridge.llm.memory")


@dataclass
class SessionTurn:
    """A single signer turn in session memory."""
    role: str = "user"
    content: str = ""
    intent: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


@dataclass
class SessionState:
    """State for a single conversation session."""
    session_id: str
    turns: list[SessionTurn] = field(default_factory=list)
    rolling_summary: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


class ConversationMemory:
    """Manages multi-turn conversation memory with a sliding window and rolling summary.

    Only the signer's output is stored (one-way rule). No assistant replies are kept.
    """

    def __init__(self, max_history_turns: int = 6) -> None:
        self.max_history_turns = max_history_turns
        self._sessions: dict[str, SessionState] = {}

    def get_or_create_session(self, session_id: str) -> SessionState:
        """Retrieve existing session state or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str | None = None,
    ) -> None:
        """Add a new message turn to the session history."""
        session = self.get_or_create_session(session_id)
        turn = SessionTurn(
            role=role,
            content=content,
            intent=intent,
            timestamp=datetime.now(UTC).isoformat(),
        )
        session.turns.append(turn)

        # Enforce sliding window on turns
        if len(session.turns) > self.max_history_turns * 2:
            overflow = session.turns[: -self.max_history_turns * 2]
            session.turns = session.turns[-self.max_history_turns * 2:]
            # Update rolling summary with older context
            user_points = [t.content for t in overflow if t.role == "user"]
            if user_points:
                combined = " Previous topics discussed: " + "; ".join(user_points[-3:])
                session.rolling_summary = (session.rolling_summary + combined).strip()[:300]

    def add_signer_turn(self, session_id: str, sentence: str, intent: str | None = None) -> None:
        """Record only the signer's composed sentence (one-way memory, no assistant turns)."""
        self.add_turn(session_id, "user", sentence, intent)

    def get_signer_history(self, session_id: str, limit: int = 6) -> list[str]:
        """Return the signer's recent spoken sentences (one-way history only)."""
        session = self.get_or_create_session(session_id)
        return [t.content for t in session.turns[-limit:] if t.role == "user"]

    def get_context_formatted(self, session_id: str) -> str:
        """Format the recent history and rolling summary for inclusion in LLM prompt."""
        session = self.get_or_create_session(session_id)
        lines: list[str] = []

        if session.rolling_summary:
            lines.append(f"Earlier conversation summary: {session.rolling_summary}")

        for turn in session.turns[-self.max_history_turns * 2:]:
            intent_str = f" [Intent: {turn.intent}]" if turn.intent else ""
            lines.append(f"Signer{intent_str}: {turn.content}")

        return "\n".join(lines) if lines else "No previous conversation turns."

    def reset_session(self, session_id: str) -> None:
        """Clear all conversation history for the given session id."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Session '%s' memory was reset.", session_id)


# Global singleton instance
memory_manager = ConversationMemory()
