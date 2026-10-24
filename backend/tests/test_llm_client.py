"""Unit tests for OpenRouterClient with mocked HTTP requests."""

import httpx
import pytest

from app.config import Settings
from app.llm.openrouter_client import OpenRouterClient, OpenRouterError


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing test configuration."""
    return Settings(
        OPENROUTER_API_KEY="sk-or-v1-mock-test-key-for-unit-tests",
        OPENROUTER_MODEL="test-primary-model",
        OPENROUTER_FALLBACK_MODELS="test-fallback-model-1,test-fallback-model-2",
        LLM_TIMEOUT_S=1.0,
    )


@pytest.mark.asyncio
async def test_client_success(monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> None:
    """Test successful completion call."""
    client = OpenRouterClient(test_settings)

    async def mock_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        data = {
            "choices": [
                {"message": {"content": '{"sentence": "I need help", "intent": "request"}'}}
            ]
        }
        return httpx.Response(200, json=data)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await client.create_completion(
        messages=[{"role": "user", "content": "test"}],
    )
    assert "I need help" in result


@pytest.mark.asyncio
async def test_client_429_retry_then_success(
    monkeypatch: pytest.MonkeyPatch, test_settings: Settings
) -> None:
    """Test 429 rate limit triggers backoff retry and succeeds on next attempt."""
    client = OpenRouterClient(test_settings)
    call_count = 0

    async def mock_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, json={"error": "Rate limit exceeded"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Success after 429"}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await client.create_completion(
        messages=[{"role": "user", "content": "test"}],
    )
    assert result == "Success after 429"
    assert call_count == 2


@pytest.mark.asyncio
async def test_client_fallback_model(
    monkeypatch: pytest.MonkeyPatch, test_settings: Settings
) -> None:
    """Test fallback model is used when primary model returns 404/error."""
    client = OpenRouterClient(test_settings)
    models_called = []

    async def mock_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        json_body = kwargs.get("json", {})
        model = json_body.get("model") if isinstance(json_body, dict) else None
        models_called.append(model)
        if model == "test-primary-model":
            return httpx.Response(404, json={"error": "Model not found"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Fallback success"}}]})

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await client.create_completion(
        messages=[{"role": "user", "content": "test"}],
    )
    assert result == "Fallback success"
    assert "test-primary-model" in models_called
    assert "test-fallback-model-1" in models_called


@pytest.mark.asyncio
async def test_client_timeout_raises(
    monkeypatch: pytest.MonkeyPatch, test_settings: Settings
) -> None:
    """Test client raises OpenRouterError when timeout happens on all retries."""
    client = OpenRouterClient(test_settings)

    async def mock_post(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        raise httpx.TimeoutException("Request timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with pytest.raises(OpenRouterError, match="All models failed"):
        await client.create_completion(
            messages=[{"role": "user", "content": "test"}],
        )


@pytest.mark.asyncio
async def test_model_availability_check(
    monkeypatch: pytest.MonkeyPatch, test_settings: Settings
) -> None:
    """Test model availability query against /models."""
    client = OpenRouterClient(test_settings)

    async def mock_get(self: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
        data = {
            "data": [
                {"id": "test-primary-model"},
                {"id": "other-model"},
            ]
        }
        return httpx.Response(200, json=data)

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    available = await client.check_model_availability()
    assert available is True
