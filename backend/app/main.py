"""Main FastAPI application entrypoint for SignBridge AI."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.collect import router as collect_router
from app.api.compose import router as compose_router
from app.api.health import router as health_router
from app.api.stream_ws import router as stream_router
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("signbridge.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown procedures."""
    settings = get_settings()
    logger.info("Starting SignBridge AI Backend (v0.1.0)...")
    logger.info("Environment: %s", settings.ENVIRONMENT)
    logger.info("Recognizer Engine: %s", settings.RECOGNIZER)
    logger.info("Primary LLM Model: %s", settings.OPENROUTER_MODEL)

    # Validate startup configuration without leaking secrets
    api_ready = settings.validate_startup()
    if not api_ready:
        logger.warning(
            "Running with offline fallback enabled. LLM calls will use deterministic mock responses."
        )

    yield

    logger.info("Shutting down SignBridge AI Backend...")


app = FastAPI(
    title="SignBridge AI API",
    description="Multimodal conversational sign language assistant backend.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(compose_router)
app.include_router(collect_router)
app.include_router(stream_router)


@app.get("/")
async def root_index() -> dict[str, str]:
    """Root landing endpoint."""
    return {
        "app": "SignBridge AI Backend",
        "status": "online",
        "health_check": "/health",
    }
