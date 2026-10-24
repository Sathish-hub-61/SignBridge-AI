"""Application configuration and environment settings for SignBridge AI.

Uses Pydantic BaseSettings to read settings from environment variables and .env files.
Validates configurations safely on startup without echoing secrets.
"""

import logging
import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("signbridge.config")


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    ENVIRONMENT: str = Field(default="development", description="Runtime environment")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    BACKEND_HOST: str = Field(default="127.0.0.1", description="Backend bind host")
    BACKEND_PORT: int = Field(default=8000, description="Backend bind port")

    # OpenRouter API (Stage 5-7 Language Layer)
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key")
    OPENROUTER_MODEL: str = Field(default="openai/gpt-4o-mini", description="Primary LLM model")
    OPENROUTER_FALLBACK_MODELS: str = Field(
        default="meta-llama/llama-3.2-3b-instruct:free,deepseek/deepseek-chat",
        description="Comma-separated fallback model names",
    )
    LLM_TIMEOUT_S: float = Field(default=15.0, description="LLM call timeout in seconds")
    LLM_TEMPERATURE_INTERPRET: float = Field(default=0.1, description="Stage A temperature")
    LLM_TEMPERATURE_RESPOND: float = Field(default=0.7, description="Stage B temperature")

    # Recognizer engine ('mock' or 'model')
    RECOGNIZER: str = Field(default="mock", description="Recognition engine type")

    # Test toggles
    RUN_LIVE_TESTS: int = Field(default=0, description="1 to run live network smoke tests")

    @property
    def fallback_models_list(self) -> list[str]:
        """Return fallback models as a list."""
        return [m.strip() for m in self.OPENROUTER_FALLBACK_MODELS.split(",") if m.strip()]

    @property
    def is_api_key_configured(self) -> bool:
        """Check if OPENROUTER_API_KEY is present and not a placeholder."""
        key = self.OPENROUTER_API_KEY.strip()
        if not key or key == "your_openrouter_api_key_here":
            return False
        return True

    def validate_startup(self) -> bool:
        """Validate startup configuration safely without logging key values.

        Returns True if API key is configured, False otherwise.
        """
        if not self.is_api_key_configured:
            logger.warning(
                "[CONFIG] OPENROUTER_API_KEY is not configured. "
                "SignBridge will run in mock/offline mode. "
                "To enable live Generative AI features, set OPENROUTER_API_KEY in your local .env file."
            )
            return False
        logger.info("[CONFIG] OPENROUTER_API_KEY is detected and configured.")
        return True


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
