"""Pydantic data models and validation schemas for SignBridge AI."""

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
    token: str = Field(..., description="ISL gloss identifier (e.g. HELP, PAIN)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score (0 to 1)")
    timestamp_ms: int | None = Field(default=None, description="Detection timestamp in ms")


class IntentEnum(StrEnum):
    """Supported intent categories for ISL conversation."""
    GREETING = "greeting"
    QUESTION = "question"
    REQUEST = "request"
    EMERGENCY = "emergency"
    CONFIRMATION = "confirmation"
    NEGATION = "negation"
    SMALLTALK = "smalltalk"
    UNCLEAR = "unclear"


class InterpretRequest(BaseModel):
    """Request payload for Stage A Interpret."""
    tokens: list[RecognizedToken] = Field(..., description="Sequence of recognized or typed glosses")
    session_id: str = Field(default="default", description="Conversation session ID")


class InterpretResponse(BaseModel):
    """Stage A structured interpretation output."""
    sentence: str = Field(..., description="Synthesized fluent English sentence")
    intent: IntentEnum = Field(..., description="Classified intent enum")
    intent_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in intent classification")
    needs_confirmation: bool = Field(default=False, description="Whether human confirmation is recommended")
    ambiguity_note: str | None = Field(default=None, description="Explanation if ambiguous or low confidence")


class RespondRequest(BaseModel):
    """Request payload for Stage B Respond."""
    sentence: str = Field(..., description="Interpreted sentence")
    intent: IntentEnum = Field(default=IntentEnum.UNCLEAR, description="Interpreted intent")
    session_id: str = Field(default="default", description="Conversation session ID")


class RespondResponse(BaseModel):
    """Stage B conversational response output."""
    reply: str = Field(..., description="Short, plain-language conversational reply (max ~2 sentences)")
    quick_replies: list[str] = Field(default_factory=list, description="Up to 3 quick response suggestions")


class PipelineRequest(BaseModel):
    """End-to-end request payload for Stage A + Stage B."""
    tokens: list[RecognizedToken] = Field(..., description="Sequence of sign glosses")
    session_id: str = Field(default="default", description="Conversation session ID")


class PipelineResponse(BaseModel):
    """End-to-end pipeline response combining Stage A and Stage B."""
    interpretation: InterpretResponse = Field(..., description="Stage A interpretation result")
    reply: str = Field(..., description="Stage B conversational reply")
    quick_replies: list[str] = Field(default_factory=list, description="Quick tap suggestions")


class ChatMessage(BaseModel):
    """A message entry in the conversation history."""
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text content")
    intent: str | None = Field(default=None, description="Intent for user turns")
    tokens: list[RecognizedToken] | None = Field(default=None, description="Source tokens if available")


class ErrorResponse(BaseModel):
    """Standardized API error response."""
    error: str = Field(..., description="Short error code")
    message: str = Field(..., description="Human-readable error description")
    detail: str | None = Field(default=None, description="Optional extra detail")
