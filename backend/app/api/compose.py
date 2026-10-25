"""Compose API endpoint for SignBridge AI (one-way signer output).

Security: per-request input size limits and a lightweight in-memory rate
limiter per session id protect the Compose chain from abuse. CORS is locked
to dev origins in main.py.
"""

import json
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.llm.compose import pipeline
from app.schemas import ComposeRequest, ComposeResponse

router = APIRouter(prefix="/compose", tags=["compose"])

# ---------------------------------------------------------------------------
# Security limits
# ---------------------------------------------------------------------------

MAX_TOKENS_PER_REQUEST = 64          # gloss tokens per Compose call
MAX_TOTAL_CHARS_PER_REQUEST = 2048   # total characters across all token strings
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 30         # max compose requests per window per session

# Sliding-window request timestamps per session id
_request_timestamps: dict[str, deque[float]] = defaultdict(deque)


def _enforce_input_limits(tokens: list) -> None:
    """Reject oversized or over-tokened payloads before hitting the LLM chain."""
    if len(tokens) > MAX_TOKENS_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many tokens: {len(tokens)} (max {MAX_TOKENS_PER_REQUEST})",
        )
    total_chars = sum(len(str(t.get("token", ""))) for t in tokens)
    if total_chars > MAX_TOTAL_CHARS_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Payload too large: {total_chars} chars (max {MAX_TOTAL_CHARS_PER_REQUEST})",
        )


def _enforce_rate_limit(session_id: str) -> None:
    """Lightweight sliding-window rate limiter per session id."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    dq = _request_timestamps[session_id]
    while dq and dq[0] < window_start:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down.",
        )
    dq.append(now)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=ComposeResponse)
async def compose_tokens(request: ComposeRequest) -> ComposeResponse:
    """Single Compose call: recognized glosses -> first-person sentence spoken as the signer."""
    if not request.tokens:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")
    _enforce_input_limits(request.tokens)
    _enforce_rate_limit(request.session_id)
    return await pipeline.compose(request)


@router.post("/stream")
async def stream_compose_response(request: ComposeRequest, http_request: Request) -> StreamingResponse:
    """Stream the composed sentence via Server-Sent Events (SSE)."""
    if not request.tokens:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")
    _enforce_input_limits(request.tokens)
    _enforce_rate_limit(request.session_id)

    response = await pipeline.compose(request)

    async def event_generator() -> AsyncGenerator[str, None]:
        yield f"event: compose\ndata: {response.model_dump_json()}\n\n"
        # Stream the sentence word-by-word for the UI to emulate TTS pacing
        words = response.sentence.split(" ")
        for w in words:
            data_payload = json.dumps({"delta": w + " "})
            yield f"event: chunk\ndata: {data_payload}\n\n"
        yield f"event: done\ndata: {json.dumps({'sentence': response.sentence})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/reset")
async def reset_conversation(session_id: str = "default") -> dict[str, str]:
    """Reset session conversation memory (signer history only)."""
    from app.llm.memory import memory_manager

    memory_manager.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@router.get("/history/{session_id}", response_model=list[str])
async def get_session_history(session_id: str = "default") -> list[str]:
    """Inspect the signer's recent spoken sentences for a given session."""
    from app.llm.memory import memory_manager

    return memory_manager.get_signer_history(session_id)
