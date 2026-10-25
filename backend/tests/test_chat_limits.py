"""Tests for compose endpoint security limits (input size + rate limiting)."""

import sys
import time
from collections import deque
from pathlib import Path

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

import pytest

from app.api import compose as compose_mod
from app.api.compose import (
    MAX_TOKENS_PER_REQUEST,
    MAX_TOTAL_CHARS_PER_REQUEST,
    RATE_LIMIT_MAX_REQUESTS,
    _enforce_input_limits,
    _enforce_rate_limit,
)


def make_tokens(count: int, text: str = "HELP") -> list[dict]:
    return [{"token": text, "confidence": 0.9} for _ in range(count)]


def test_input_limits_accepts_normal_payload():
    _enforce_input_limits(make_tokens(5))  # should not raise


def test_input_limits_rejects_too_many_tokens():
    with pytest.raises(HTTPException):
        _enforce_input_limits(make_tokens(MAX_TOKENS_PER_REQUEST + 1))


def test_input_limits_rejects_too_many_chars():
    # Each token "HELP" = 4 chars. MAX_TOTAL_CHARS_PER_REQUEST = 2048
    over = MAX_TOTAL_CHARS_PER_REQUEST // 4 + 10
    with pytest.raises(HTTPException):
        _enforce_input_limits(make_tokens(over, text="HELP"))


def test_rate_limit_allows_within_window():
    session_id = "sess-test-1"
    compose_mod._request_timestamps[session_id] = deque()
    for _ in range(5):
        _enforce_rate_limit(session_id)  # should not raise


def test_rate_limit_rejects_over_limit():
    session_id = "sess-test-2"
    compose_mod._request_timestamps[session_id] = deque()
    # Set all timestamps to now so they're within the window
    now = time.time()
    compose_mod._request_timestamps[session_id] = deque([now] * RATE_LIMIT_MAX_REQUESTS)
    with pytest.raises(HTTPException):
        _enforce_rate_limit(session_id)


def test_rate_limit_resets_after_window():
    session_id = "sess-test-3"
    compose_mod._request_timestamps[session_id] = deque()
    # Old timestamps outside the window should be dropped
    old = time.time() - compose_mod.RATE_LIMIT_WINDOW_SECONDS - 10
    compose_mod._request_timestamps[session_id] = deque([old] * RATE_LIMIT_MAX_REQUESTS)
    _enforce_rate_limit(session_id)  # should not raise (old ones dropped)
