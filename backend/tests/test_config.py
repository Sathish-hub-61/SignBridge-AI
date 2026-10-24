"""Unit tests for configuration loading and validation."""

from app.config import Settings


def test_default_settings() -> None:
    """Verify default settings values when unconfigured."""
    settings = Settings(
        OPENROUTER_API_KEY="",
        ENVIRONMENT="development",
        RECOGNIZER="mock",
    )
    assert settings.ENVIRONMENT == "development"
    assert settings.RECOGNIZER == "mock"
    assert not settings.is_api_key_configured
    assert not settings.validate_startup()


def test_configured_api_key() -> None:
    """Verify that a valid non-placeholder API key is detected without leaking."""
    settings = Settings(
        OPENROUTER_API_KEY="sk-or-v1-mock-test-key-do-not-leak",
    )
    assert settings.is_api_key_configured
    assert settings.validate_startup()


def test_placeholder_api_key_treated_as_unconfigured() -> None:
    """Verify that placeholder strings from .env.example are treated as unconfigured."""
    settings = Settings(
        OPENROUTER_API_KEY="your_openrouter_api_key_here",
    )
    assert not settings.is_api_key_configured
    assert not settings.validate_startup()


def test_fallback_models_parsing() -> None:
    """Verify fallback models list parsing."""
    settings = Settings(
        OPENROUTER_FALLBACK_MODELS="model-a, model-b ,model-c",
    )
    assert settings.fallback_models_list == ["model-a", "model-b", "model-c"]
