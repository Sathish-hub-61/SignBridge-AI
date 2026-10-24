"""OpenRouter API client with exponential backoff, model fallbacks, and streaming."""

import asyncio
import json
import logging
import random
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger("signbridge.llm.client")


class OpenRouterError(Exception):
    """Base exception for OpenRouter API client errors."""


class OpenRouterClient:
    """Robust client for OpenRouter chat completions with model fallback and offline recovery."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _get_headers(self) -> dict[str, str]:
        """Construct required headers for OpenRouter."""
        return {
            "Authorization": f"Bearer {self.settings.OPENROUTER_API_KEY.strip()}",
            "HTTP-Referer": f"http://{self.settings.BACKEND_HOST}:{self.settings.BACKEND_PORT}",
            "X-Title": "SignBridge AI",
            "Content-Type": "application/json",
        }

    async def check_model_availability(self) -> bool:
        """Query the OpenRouter /models endpoint to verify model access.

        Logs a clear warning if unavailable instead of crashing.
        """
        if not self.settings.is_api_key_configured:
            logger.info("Skipping model check: OPENROUTER_API_KEY not configured.")
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(
                    f"{self.BASE_URL}/models",
                    headers={"Authorization": f"Bearer {self.settings.OPENROUTER_API_KEY.strip()}"},
                )
                if res.status_code == 200:
                    models_data = res.json().get("data", [])
                    available_ids = {m.get("id") for m in models_data if isinstance(m, dict)}
                    if self.settings.OPENROUTER_MODEL not in available_ids:
                        logger.warning(
                            "Configured primary model '%s' not found in OpenRouter available models. "
                            "Will attempt fallback models: %s",
                            self.settings.OPENROUTER_MODEL,
                            self.settings.fallback_models_list,
                        )
                        return False
                    logger.info("Primary model '%s' verified on OpenRouter.", self.settings.OPENROUTER_MODEL)
                    return True
                logger.warning("OpenRouter /models returned status %d", res.status_code)
                return False
        except Exception as e:
            logger.warning("Unable to reach OpenRouter /models endpoint (%s). Proceeding with configured models.", e)
            return False

    async def create_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 400,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Call OpenRouter completions with exponential backoff and fallback model progression.

        Raises OpenRouterError if all models and retries fail.
        """
        if not self.settings.is_api_key_configured:
            raise OpenRouterError("OPENROUTER_API_KEY is not configured.")

        # Candidate model chain: primary model followed by fallback models
        models_to_try = [self.settings.OPENROUTER_MODEL] + self.settings.fallback_models_list

        headers = self._get_headers()
        last_error: Exception | None = None

        for model in models_to_try:
            # Up to 3 attempts per model with exponential backoff + jitter
            for attempt in range(1, 4):
                payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format:
                    payload["response_format"] = response_format

                try:
                    async with httpx.AsyncClient(timeout=self.settings.LLM_TIMEOUT_S) as client:
                        resp = await client.post(
                            f"{self.BASE_URL}/chat/completions",
                            headers=headers,
                            json=payload,
                        )

                        if resp.status_code == 200:
                            data = resp.json()
                            choices = data.get("choices", [])
                            if choices:
                                content = choices[0].get("message", {}).get("content", "")
                                return content
                            raise OpenRouterError(f"Empty choices in OpenRouter response from model {model}")

                        # 429 Rate Limit or 5xx Server Errors -> retry with backoff
                        if resp.status_code in (429, 500, 502, 503, 504):
                            delay = (0.5 * (2 ** (attempt - 1))) + random.uniform(0.05, 0.2)
                            logger.warning(
                                "OpenRouter %d on model '%s' (attempt %d/3). Retrying in %.2fs...",
                                resp.status_code,
                                model,
                                attempt,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue

                        # Other status codes (400, 401, 404) -> stop retrying this model
                        logger.error(
                            "OpenRouter error %d on model '%s': %s",
                            resp.status_code,
                            model,
                            resp.text[:200],
                        )
                        last_error = OpenRouterError(f"HTTP {resp.status_code} from {model}")
                        break

                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    delay = (0.5 * (2 ** (attempt - 1))) + random.uniform(0.05, 0.2)
                    logger.warning(
                        "Network error '%s' on model '%s' (attempt %d/3). Retrying in %.2fs...",
                        e,
                        model,
                        attempt,
                        delay,
                    )
                    last_error = e
                    await asyncio.sleep(delay)

        raise OpenRouterError(f"All models failed on OpenRouter. Last error: {last_error}")

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 300,
    ) -> AsyncGenerator[str, None]:
        """Stream reply tokens from OpenRouter via Server-Sent Events."""
        if not self.settings.is_api_key_configured:
            raise OpenRouterError("OPENROUTER_API_KEY is not configured.")

        models_to_try = [self.settings.OPENROUTER_MODEL] + self.settings.fallback_models_list
        headers = self._get_headers()

        for model in models_to_try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            try:
                async with httpx.AsyncClient(timeout=self.settings.LLM_TIMEOUT_S) as client:
                    async with client.stream(
                        "POST",
                        f"{self.BASE_URL}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status_code != 200:
                            logger.warning(
                                "Streaming request returned %d for model '%s'",
                                response.status_code,
                                model,
                            )
                            continue

                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                        return  # Stream completed successfully
            except Exception as e:
                logger.warning("Stream failed on model '%s': %s. Trying next fallback...", model, e)

        raise OpenRouterError("Streaming failed on all configured models.")
