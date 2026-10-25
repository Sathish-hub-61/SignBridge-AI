"""Pydantic data models and validation schemas for SignBridge AI.

Strictly adheres to the One-Way Compose pipeline specification.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class ServiceStatus(StrEnum):
    """Health status enum."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: ServiceStatus = Field(default=ServiceStatus.HEALTHY, description="Backend service status")
    version: str = Field(default="0.1.0", description="API version")
    environment: str = Field(..., description="Runtime environment (development/production)")
    recognizer: str = Field(..., description="Active recognition engine (mock/model)")
    llm_configured: bool = Field(..., description="Whether OpenRouter API key is configured")
    primary_model: str = Field(..., description="Configured primary LLM model")


class RecognizedToken(BaseModel):
    """A single recognized sign token with confidence."""

    token: str = Field(..., description="ISL gloss identifier (e.g. HELP, WATER)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score (0 to 1)")
    timestamp_ms: int | None = Field(default=None, description="Detection timestamp in ms")


class IntentEnum(StrEnum):
    """Supported intent categories for ISL sentence generation."""

    STATEMENT = "statement"
    QUESTION = "question"
    REQUEST = "request"
    GREETING = "greeting"
    THANKS = "thanks"
    APOLOGY = "apology"
    NEGATION = "negation"
    UNCLEAR = "unclear"


class ToneEnum(StrEnum):
    """Signer's preferred speaking tone."""

    CASUAL = "casual"
    POLITE = "polite"
    NEUTRAL = "neutral"


class ScenarioEnum(StrEnum):
    """Predefined conversation scenarios for disambiguation."""

    GENERAL = "general"
    SHOP = "shop"
    RESTAURANT = "restaurant"
    BUS = "bus"
    HOSPITAL_RECEPTION = "hospital_reception"
    OFFICE = "office"


class SignerProfile(BaseModel):
    """Signer profile preferences for personalized voice synthesis."""

    name: str = Field(default="Signer", description="Signer name or identifier")
    tone: ToneEnum = Field(default=ToneEnum.POLITE, description="Preferred conversational tone")


class ComposeRequest(BaseModel):
    """Request payload for the single Compose LLM call."""

    tokens: list[RecognizedToken] = Field(..., description="Recognized ISL tokens with confidences")
    scenario: ScenarioEnum = Field(default=ScenarioEnum.GENERAL, description="Active scenario")
    profile: SignerProfile = Field(default_factory=SignerProfile, description="Signer profile settings")
    session_id: str = Field(default="default", description="Conversation session ID")


class ComposeResponse(BaseModel):
    """Validated structured output from LLM Compose call."""

    sentence: str = Field(..., description="Composed first-person English sentence spoken as the signer")
    alternatives: list[str] = Field(
        default_factory=list,
        max_length=2,
        description="Up to 2 alternative phrasing options if ambiguity exists",
    )
    intent: IntentEnum = Field(default=IntentEnum.STATEMENT, description="Classified intent")
    tone: str = Field(default="polite", description="Detected/applied tone")
    needs_confirmation: bool = Field(
        default=False,
        description="Whether human confirmation is recommended (e.g. low confidence or ambiguity)",
    )
    ambiguity_note: str | None = Field(
        default=None,
        description="Explanation note if glosses were ambiguous or missing information",
    )


class ErrorResponse(BaseModel):
    """Standardized API error response."""

    error: str = Field(..., description="Short error code")
    message: str = Field(..., description="Human-readable error description")
    detail: str | None = Field(default=None, description="Optional extra detail")
