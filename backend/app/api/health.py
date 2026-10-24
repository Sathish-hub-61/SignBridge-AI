"""Health check and diagnostic endpoints."""

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas import HealthResponse, ServiceStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return backend health status, recognizer configuration, and environment info."""
    return HealthResponse(
        status=ServiceStatus.HEALTHY,
        version="0.1.0",
        environment=settings.ENVIRONMENT,
        recognizer=settings.RECOGNIZER,
        llm_configured=settings.is_api_key_configured,
        primary_model=settings.OPENROUTER_MODEL,
    )
