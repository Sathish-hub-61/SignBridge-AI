"""Two-stage language interpretation and response pipeline for SignBridge AI."""

import json
import logging
import re
from collections.abc import AsyncGenerator
from pathlib import Path

from app.config import Settings, get_settings
from app.llm.memory import ConversationMemory, memory_manager
from app.llm.openrouter_client import OpenRouterClient
from app.schemas import (
    IntentEnum,
    InterpretResponse,
    PipelineResponse,
    RecognizedToken,
    RespondResponse,
)

logger = logging.getLogger("signbridge.llm.pipeline")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(filename: str) -> str:
    """Load prompt template text from prompts directory."""
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    logger.warning("Prompt file '%s' not found. Using embedded fallback prompt.", filename)
    return "You are SignBridge AI. Process the provided sign language glosses."


def sanitize_glosses(tokens: list[RecognizedToken]) -> tuple[list[RecognizedToken], str]:
    """Sanitize and delimit untrusted sign tokens to prevent prompt injection."""
    sanitized: list[RecognizedToken] = []
    # Cap total tokens to 30
    for t in tokens[:30]:
        clean_word = re.sub(r"[^A-Za-z0-9_\-]", "", t.token.strip().upper())[:24]
        if clean_word:
            conf = max(0.0, min(1.0, float(t.confidence)))
            sanitized.append(RecognizedToken(token=clean_word, confidence=conf))

    if not sanitized:
        sanitized = [RecognizedToken(token="UNCLEAR", confidence=0.0)]

    # Form formatted representation with confidence
    formatted_str = ", ".join(f"{t.token} (conf: {t.confidence:.2f})" for t in sanitized)
    return sanitized, f"<glosses>{formatted_str}</glosses>"


class LLMPipeline:
    """Orchestrates Stage A (Interpret) and Stage B (Respond) with offline fallbacks."""

    def __init__(
        self,
        client: OpenRouterClient | None = None,
        memory: ConversationMemory | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or OpenRouterClient(self.settings)
        self.memory = memory or memory_manager

        self.interpret_system_prompt = load_prompt("interpret_v1.txt")
        self.respond_system_prompt = load_prompt("respond_v1.txt")
        self.merged_system_prompt = load_prompt("merged_v1.txt")

    def _offline_interpret_fallback(self, tokens: list[RecognizedToken]) -> InterpretResponse:
        """Deterministic rule-based interpretation when offline or when LLM fails."""
        words = [t.token.upper() for t in tokens]
        words_set = set(words)
        min_conf = min([t.confidence for t in tokens], default=1.0)
        needs_confirm = min_conf < 0.6

        # Emergency keywords
        if any(w in words_set for w in ("HELP", "PAIN", "DOCTOR", "EMERGENCY", "HURT", "BLOOD", "ACCIDENT")):
            return InterpretResponse(
                sentence="I need urgent medical help and a doctor immediately.",
                intent=IntentEnum.EMERGENCY,
                intent_confidence=0.95,
                needs_confirmation=needs_confirm,
                ambiguity_note="Low token confidence detected" if needs_confirm else None,
            )

        # Greetings
        if any(w in words_set for w in ("HELLO", "HI", "NAMASTE", "MORNING", "WELCOME")):
            return InterpretResponse(
                sentence="Hello, greetings to you.",
                intent=IntentEnum.GREETING,
                intent_confidence=0.92,
                needs_confirmation=needs_confirm,
            )

        # Questions
        if any(w in words_set for w in ("WHERE", "WHAT", "WHEN", "HOW", "WHY")):
            return InterpretResponse(
                sentence=f"Could you please tell me: {' '.join(words)}?",
                intent=IntentEnum.QUESTION,
                intent_confidence=0.85,
                needs_confirmation=needs_confirm,
            )

        # Confirmations
        if any(w in words_set for w in ("YES", "OK", "AGREE", "CORRECT")):
            return InterpretResponse(
                sentence="Yes, that is correct.",
                intent=IntentEnum.CONFIRMATION,
                intent_confidence=0.90,
                needs_confirmation=needs_confirm,
            )

        # Negations
        if any(w in words_set for w in ("NO", "NOT", "WRONG", "STOP")):
            return InterpretResponse(
                sentence="No, that is not right.",
                intent=IntentEnum.NEGATION,
                intent_confidence=0.90,
                needs_confirmation=needs_confirm,
            )

        # Smalltalk
        if any(w in words_set for w in ("THANK", "THANKS", "GOOD")):
            return InterpretResponse(
                sentence="Thank you very much.",
                intent=IntentEnum.SMALLTALK,
                intent_confidence=0.88,
                needs_confirmation=needs_confirm,
            )

        # General / Unclear
        sentence_body = " ".join(words).capitalize() + "."
        return InterpretResponse(
            sentence=f"I am indicating: {sentence_body}",
            intent=IntentEnum.REQUEST if min_conf >= 0.6 else IntentEnum.UNCLEAR,
            intent_confidence=round(min_conf, 2),
            needs_confirmation=needs_confirm,
            ambiguity_note="Inferred via deterministic rule fallback" if needs_confirm else None,
        )

    def _offline_respond_fallback(self, interpretation: InterpretResponse) -> RespondResponse:
        """Deterministic conversational reply when offline."""
        intent = interpretation.intent

        if intent == IntentEnum.EMERGENCY:
            return RespondResponse(
                reply="I hear your emergency. Please contact emergency services immediately at 112. Where are you located?",
                quick_replies=["Call 112", "Chest pain", "Need ambulance"],
            )
        if intent == IntentEnum.GREETING:
            return RespondResponse(
                reply="Hello! I am SignBridge AI. How can I help you today?",
                quick_replies=["Need doctor", "Hospital desk", "Directions"],
            )
        if intent == IntentEnum.QUESTION:
            return RespondResponse(
                reply="I received your question. Let me verify the details for you.",
                quick_replies=["Thank you", "Show location", "Ask staff"],
            )
        if intent == IntentEnum.CONFIRMATION:
            return RespondResponse(
                reply="Thank you for confirming. Proceeding with your request.",
                quick_replies=["Continue", "Start over"],
            )
        if intent == IntentEnum.NEGATION:
            return RespondResponse(
                reply="Understood. Let us stop and correct what was misunderstood.",
                quick_replies=["Try sign again", "Type gloss", "Cancel"],
            )
        if intent == IntentEnum.SMALLTALK:
            return RespondResponse(
                reply="You are very welcome! Please let me know if you need anything else.",
                quick_replies=["All set", "New question"],
            )

        # Unclear or general
        return RespondResponse(
            reply="I recognized your signs, but could you confirm if this is what you meant?",
            quick_replies=["Yes, correct", "No, re-sign", "Type words"],
        )

    async def interpret(
        self,
        tokens: list[RecognizedToken],
        session_id: str = "default",
    ) -> InterpretResponse:
        """Stage A: Convert sign glosses into a validated English sentence and intent."""
        clean_tokens, delimited_str = sanitize_glosses(tokens)

        # Check if offline fallback is required
        if not self.settings.is_api_key_configured:
            return self._offline_interpret_fallback(clean_tokens)

        history_context = self.memory.get_context_formatted(session_id)
        user_prompt = (
            f"Conversation context:\n{history_context}\n\n"
            f"Current recognized input:\n{delimited_str}\n\n"
            "Produce the JSON interpretation object:"
        )

        messages = [
            {"role": "system", "content": self.interpret_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # First attempt
        raw_output = ""
        try:
            raw_output = await self.client.create_completion(
                messages=messages,
                temperature=self.settings.LLM_TEMPERATURE_INTERPRET,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw_output)
            return InterpretResponse.model_validate(data)
        except Exception as first_err:
            logger.warning("Stage A initial parse error: %s. Output was: %s", first_err, raw_output[:100])

            # One repair attempt
            try:
                repair_prompt = (
                    f"Your previous response was not valid JSON matching the schema. Error: {first_err}.\n"
                    "Fix it and return strictly a valid JSON object matching the required schema."
                )
                repair_messages = messages + [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": repair_prompt},
                ]
                repaired_output = await self.client.create_completion(
                    messages=repair_messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                data = json.loads(repaired_output)
                return InterpretResponse.model_validate(data)
            except Exception as repair_err:
                logger.error("Stage A repair attempt failed: %s. Falling back to offline rule.", repair_err)
                return self._offline_interpret_fallback(clean_tokens)

    async def respond(
        self,
        interpretation: InterpretResponse,
        session_id: str = "default",
    ) -> RespondResponse:
        """Stage B: Generate a short, plain conversational reply and quick replies."""
        if not self.settings.is_api_key_configured:
            fallback = self._offline_respond_fallback(interpretation)
            self.memory.add_turn(session_id, "user", interpretation.sentence, interpretation.intent)
            self.memory.add_turn(session_id, "assistant", fallback.reply)
            return fallback

        history_context = self.memory.get_context_formatted(session_id)
        user_prompt = (
            f"Conversation history:\n{history_context}\n\n"
            f"User's interpreted sentence: \"{interpretation.sentence}\"\n"
            f"Intent: {interpretation.intent.value}\n"
            f"Needs confirmation: {interpretation.needs_confirmation}\n\n"
            "Provide your supportive response and quick replies as a JSON object:"
        )

        messages = [
            {"role": "system", "content": self.respond_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_output = await self.client.create_completion(
                messages=messages,
                temperature=self.settings.LLM_TEMPERATURE_RESPOND,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw_output)
            resp = RespondResponse.model_validate(data)

            # Safety enforcement for emergency intent: ensure 112 is present
            if interpretation.intent == IntentEnum.EMERGENCY and "112" not in resp.reply:
                resp.reply = f"{resp.reply} Please call emergency services at 112 immediately."

            # Record turn in session memory
            self.memory.add_turn(session_id, "user", interpretation.sentence, interpretation.intent)
            self.memory.add_turn(session_id, "assistant", resp.reply)
            return resp
        except Exception as e:
            logger.warning("Stage B response failed (%s). Using offline response fallback.", e)
            fallback = self._offline_respond_fallback(interpretation)
            self.memory.add_turn(session_id, "user", interpretation.sentence, interpretation.intent)
            self.memory.add_turn(session_id, "assistant", fallback.reply)
            return fallback

    async def stream_respond(
        self,
        interpretation: InterpretResponse,
        session_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """Stream reply text chunk by chunk for the UI."""
        if not self.settings.is_api_key_configured:
            # Yield offline reply word by word to emulate streaming
            fallback = self._offline_respond_fallback(interpretation)
            words = fallback.reply.split(" ")
            for w in words:
                yield w + " "
            self.memory.add_turn(session_id, "user", interpretation.sentence, interpretation.intent)
            self.memory.add_turn(session_id, "assistant", fallback.reply)
            return

        history_context = self.memory.get_context_formatted(session_id)
        user_prompt = (
            f"Conversation history:\n{history_context}\n\n"
            f"User's interpreted sentence: \"{interpretation.sentence}\"\n"
            f"Intent: {interpretation.intent.value}\n\n"
            "Provide ONLY the plain-text reply (max 2 short sentences, emergency rule applied if relevant):"
        )

        messages = [
            {"role": "system", "content": self.respond_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        full_reply_chunks: list[str] = []
        try:
            async for chunk in self.client.stream_completion(
                messages=messages,
                temperature=self.settings.LLM_TEMPERATURE_RESPOND,
            ):
                full_reply_chunks.append(chunk)
                yield chunk

            accumulated = "".join(full_reply_chunks).strip()
            self.memory.add_turn(session_id, "user", interpretation.sentence, interpretation.intent)
            self.memory.add_turn(session_id, "assistant", accumulated)
        except Exception as e:
            logger.warning("Streaming failed (%s). Yielding fallback reply.", e)
            fallback = self._offline_respond_fallback(interpretation)
            for w in fallback.reply.split(" "):
                yield w + " "

    async def run_full_pipeline(
        self,
        tokens: list[RecognizedToken],
        session_id: str = "default",
    ) -> PipelineResponse:
        """Run end-to-end interpretation followed by response generation."""
        interpretation = await self.interpret(tokens, session_id=session_id)
        response = await self.respond(interpretation, session_id=session_id)
        return PipelineResponse(
            interpretation=interpretation,
            reply=response.reply,
            quick_replies=response.quick_replies,
        )


# Global singleton instance
pipeline = LLMPipeline()
