"""Single-call Compose language pipeline for SignBridge AI (one-way signer output).

Replaces the two-stage Interpret/Respond design. One Compose call turns recognized
sign glosses into a fluent first-person English sentence spoken AS the signer.

Design rationale (see docs/design-decisions.md):
- A single Compose call minimizes latency, cost, and error compounding.
- It trades away modular evaluation (one call instead of two), which is acceptable
  for a one-way signer-output system where there is no assistant reply to validate.
"""

import json
import logging
import re
from pathlib import Path

from app.config import Settings, get_settings
from app.llm.memory import ConversationMemory, memory_manager
from app.llm.openrouter_client import OpenRouterClient
from app.schemas import (
    ComposeRequest,
    ComposeResponse,
    IntentEnum,
    RecognizedToken,
    SignerProfile,
)

logger = logging.getLogger("signbridge.llm.compose")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

MAX_TOKENS = 30
MAX_GLOSS_LEN = 24
MAX_SCENARIO_LEN = 32
MAX_HISTORY_SENTENCES = 6


def load_prompt(filename: str) -> str:
    """Load prompt template text from prompts directory."""
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    logger.warning("Prompt file '%s' not found. Using embedded fallback prompt.", filename)
    return (
        "You are SignBridge AI. Compose one short first-person English sentence "
        "spoken as the signer from the recognized sign glosses. Output valid JSON only."
    )


def _sanitize_word(word: str) -> str:
    """Sanitize a single gloss token (injection hygiene, length-capped)."""
    clean = re.sub(r"[^A-Za-z0-9_\-]", "", word.strip().upper())[:MAX_GLOSS_LEN]
    return clean


def _format_glosses(tokens: list[RecognizedToken]) -> str:
    """Delimit glosses in <glosses> tags with confidence (length-capped)."""
    items = []
    for t in tokens[:MAX_TOKENS]:
        clean = _sanitize_word(t.token)
        if not clean:
            continue
        conf = max(0.0, min(1.0, float(t.confidence)))
        items.append(f"{clean}({conf:.2f})")
    if not items:
        items = ["UNCLEAR(0.00)"]
    return f"<glosses>{', '.join(items)}</glosses>"


def _scrub_text(value: str, max_len: int) -> str:
    """Scrub an untrusted text field: strip control chars, cap length."""
    clean = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value)
    return clean.strip()[:max_len]


class ComposePipeline:
    """Single Compose call: glosses + scenario + profile + history -> sentence."""

    def __init__(
        self,
        client: OpenRouterClient | None = None,
        memory: ConversationMemory | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or OpenRouterClient(self.settings)
        self.memory = memory or memory_manager
        self.system_prompt = load_prompt("compose_v1.md")

    # ------------------------------------------------------------------
    # Offline fallback (deterministic rule-based, no LLM)
    # ------------------------------------------------------------------

    def _offline_fallback(self, tokens: list[RecognizedToken]) -> ComposeResponse:
        """Build a simple rule-based sentence so the demo never hard-fails."""
        clean_tokens = [_sanitize_word(t.token) for t in tokens[:MAX_TOKENS]]
        clean_tokens = [t for t in clean_tokens if t]
        if not clean_tokens:
            clean_tokens = ["UNCLEAR"]
        words = " ".join(clean_tokens)
        min_conf = min([t.confidence for t in tokens], default=1.0) if tokens else 1.0
        needs_confirm = min_conf < 0.6

        sentence = f"I want {words}." if len(clean_tokens) > 1 else f"I am indicating {words}."
        if clean_tokens[0] in {"HELLO", "HI", "NAMASTE"}:
            sentence = "Hello."
            intent = IntentEnum.GREETING
        elif clean_tokens[0] in {"YES", "OK"}:
            sentence = "Yes, that is correct."
            intent = IntentEnum.STATEMENT
        elif clean_tokens[0] in {"NO", "NOT", "STOP"}:
            sentence = "No, that is not right."
            intent = IntentEnum.NEGATION
        elif clean_tokens[0] in {"THANK", "THANKS"}:
            sentence = "Thank you very much."
            intent = IntentEnum.THANKS
        elif clean_tokens[0] in {"WHERE", "WHAT", "WHEN", "HOW", "WHY"}:
            sentence = f"Could you tell me about {words}?"
            intent = IntentEnum.QUESTION
        else:
            intent = IntentEnum.REQUEST if min_conf >= 0.6 else IntentEnum.UNCLEAR

        return ComposeResponse(
            sentence=sentence,
            alternatives=[] if not needs_confirm else [f"Could you repeat {words}?"],
            intent=intent,
            tone="polite",
            needs_confirmation=needs_confirm,
            ambiguity_note="Deterministic rule fallback" if needs_confirm else None,
        )

    # ------------------------------------------------------------------
    # Single Compose call
    # ------------------------------------------------------------------

    async def compose(
        self,
        request: ComposeRequest,
    ) -> ComposeResponse:
        """Compose one first-person sentence from recognized glosses."""
        tokens = list(request.tokens)[:MAX_TOKENS]
        scenario = _scrub_text(request.scenario, MAX_SCENARIO_LEN)
        profile = request.profile or SignerProfile()
        name = _scrub_text(profile.name, 32)
        tone = profile.tone.value if hasattr(profile.tone, "value") else str(profile.tone)
        if tone not in ("casual", "polite"):
            tone = "polite"

        glosses_str = _format_glosses(tokens)
        history = self.memory.get_signer_history(request.session_id)[:MAX_HISTORY_SENTENCES]
        history_str = "<history>" + (" | ".join(history) if history else "none") + "</history>"

        user_prompt = (
            f"{glosses_str}\n"
            f"<scenario>{scenario}</scenario>\n"
            f"<signer_name>{name}</signer_name>\n"
            f"<signer_tone>{tone}</signer_tone>\n"
            f"{history_str}\n\n"
            "Compose one first-person sentence spoken as the signer. Output valid JSON only:"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Offline mode: no API key -> deterministic fallback
        if not self.settings.is_api_key_configured:
            return self._offline_fallback(tokens)

        # First attempt
        raw_output = ""
        try:
            raw_output = await self.client.create_completion(
                messages=messages,
                temperature=self.settings.LLM_TEMPERATURE_INTERPRET,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw_output)
            response = ComposeResponse.model_validate(data)
            self.memory.add_signer_turn(request.session_id, response.sentence, response.intent.value)
            return response
        except Exception as first_err:
            logger.warning("Compose initial parse error: %s. Output was: %s", first_err, raw_output[:100])

            # One repair attempt
            try:
                repair_prompt = (
                    f"Your previous response was not valid JSON matching the schema. Error: {first_err}.\n"
                    "Fix it and return strictly a valid JSON object with keys: sentence, alternatives, "
                    "intent, tone, needs_confirmation, ambiguity_note."
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
                response = ComposeResponse.model_validate(data)
                self.memory.add_signer_turn(request.session_id, response.sentence, response.intent.value)
                return response
            except Exception as repair_err:
                logger.error("Compose repair attempt failed: %s. Falling back to offline rule.", repair_err)
                return self._offline_fallback(tokens)


# Global singleton instance
pipeline = ComposePipeline()
