"""Chat and language endpoints for SignBridge AI.

Security: per-request input size limits and a lightweight in-memory rate
limiter per session id protect the LLM chain from abuse. CORS is locked to
dev origins in main.py.
"""

import json
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.llm.memory import memory_manager
from app.llm.pipeline import pipeline
from app.schemas import (
    ChatMessage,
    InterpretRequest,
    InterpretResponse,
    PipelineRequest,
    PipelineResponse,
    RespondRequest,
    RespondResponse,
)

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------------------------------------------------------------------------
# Security limits
# ---------------------------------------------------------------------------

MAX_TOKENS_PER_REQUEST = 64          # gloss tokens per Interpret/Respond call
MAX_TOTAL_CHARS_PER_REQUEST = 2048   # total characters across all token strings
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 30         # max chat requests per window per session

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
    # Drop timestamps outside the window
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


@router.post("/interpret", response_model=InterpretResponse)
async def interpret_tokens(request: InterpretRequest) -> InterpretResponse:
    """Stage A: Convert sign glosses into a structured English sentence and intent."""
    if not request.tokens:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")
    _enforce_input_limits(request.tokens)
    return await pipeline.interpret(tokens=request.tokens, session_id=request.session_id)


@router.post("/respond", response_model=RespondResponse)
async def respond_to_sentence(request: RespondRequest) -> RespondResponse:
    """Stage B: Generate conversational reply and quick replies for an interpreted sentence."""
    interpretation = InterpretResponse(
        sentence=request.sentence,
        intent=request.intent,
        intent_confidence=1.0,
        needs_confirmation=False,
    )
    return await pipeline.respond(interpretation=interpretation, session_id=request.session_id)


@router.post("/pipeline", response_model=PipelineResponse)
async def execute_full_pipeline(request: PipelineRequest) -> PipelineResponse:
    """End-to-End: Process sign tokens through Stage A and Stage B."""
    if not request.tokens:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")
    _enforce_input_limits(request.tokens)
    _enforce_rate_limit(request.session_id)
    return await pipeline.run_full_pipeline(tokens=request.tokens, session_id=request.session_id)


@router.post("/stream")
async def stream_chat_response(request: PipelineRequest, http_request: Request) -> StreamingResponse:
    """Stream conversation reply using Server-Sent Events (SSE)."""
    if not request.tokens:
        raise HTTPException(status_code=400, detail="Tokens list cannot be empty.")
    _enforce_input_limits(request.tokens)
    _enforce_rate_limit(request.session_id)

    # 1. Run Stage A interpretation first
    interpretation = await pipeline.interpret(tokens=request.tokens, session_id=request.session_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        # Emit Stage A interpretation event
        yield f"event: interpretation\ndata: {interpretation.model_dump_json()}\n\n"

        # Emit streaming reply chunks
        full_chunks: list[str] = []
        async for chunk in pipeline.stream_respond(interpretation, session_id=request.session_id):
            full_chunks.append(chunk)
            data_payload = json.dumps({"delta": chunk})
            yield f"event: chunk\ndata: {data_payload}\n\n"

        # Emit completion with quick replies
        quick_resp = await pipeline.respond(interpretation, session_id=request.session_id)
        done_payload = json.dumps({
            "full_reply": "".join(full_chunks) or quick_resp.reply,
            "quick_replies": quick_resp.quick_replies,
        })
        yield f"event: done\ndata: {done_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/reset")
async def reset_conversation(session_id: str = "default") -> dict[str, str]:
    """Reset session conversation memory."""
    memory_manager.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@router.get("/history/{session_id}", response_model=list[ChatMessage])
async def get_session_history(session_id: str = "default") -> list[ChatMessage]:
    """Inspect current message history for a given session."""
    session = memory_manager.get_or_create_session(session_id)
    return session.messages
