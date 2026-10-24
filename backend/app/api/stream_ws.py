"""WebSocket streaming endpoint for landmark ingestion (Phase 2 + Phase 4).

Accepts compact landmark frame arrays from the browser, runs them through the
per-session segmenter + recognizer, and pushes recognized tokens back to the
client so they can be reviewed as editable confidence-shaded chips before
synthesis (confirm-before-send).

Message format (client -> server):
    {
      "session_id": "sess-123",
      "frames": [
        {
          "left_hand": [{"x":..,"y":..,"z":..}, ... 21],
          "right_hand": [{"x":..,"y":..,"z":..}, ... 21],
          "pose": [{"x":..,"y":..,"z":..}, ... 33],
          "face": [{"x":..,"y":..,"z":..}, ... 478]
        }
      ]
    }

Server replies with a JSON count echo per batch, plus a ``token`` event when
a recognition window completes.
"""

import json
import logging
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.recognition.features import extract_frame_features
from app.recognition.segmenter import segmenter
from app.schemas import RecognizedToken

logger = logging.getLogger("signbridge.stream_ws")

router = APIRouter(prefix="/stream", tags=["stream"])

# In-memory buffer counts per session (echo only; recognition in segmenter)
_buffer_counts: defaultdict[str, int] = defaultdict(int)


def _frame_to_features(frame: dict) -> list[float] | None:
    """Extract 193-dim features from a raw frame dict; None on failure."""
    if not isinstance(frame, dict):
        return None
    try:
        feats = extract_frame_features(frame)  # list[float] length 193
        return list(feats)
    except Exception:
        return None


@router.websocket("/ws")
async def stream_landmarks(websocket: WebSocket) -> None:
    """Accept compact landmark frame arrays, segment, recognize, echo tokens."""
    await websocket.accept()
    session_id = "default"
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid_json"}))
                continue

            if not isinstance(payload, dict):
                await websocket.send_text(json.dumps({"error": "payload must be an object"}))
                continue

            session_id = str(payload.get("session_id", "default"))
            frames = payload.get("frames", [])
            if not isinstance(frames, list):
                await websocket.send_text(json.dumps({"error": "frames must be a list"}))
                continue

            extracted = 0
            emitted_token: RecognizedToken | None = None
            for frame in frames:
                feats = _frame_to_features(frame)
                if feats is None:
                    continue
                extracted += 1
                token = segmenter.add_frame(session_id, feats)
                if token is not None:
                    emitted_token = token  # last completed window in this batch

            _buffer_counts[session_id] += extracted

            reply: dict = {
                "session_id": session_id,
                "frames_received": len(frames),
                "frames_extracted": extracted,
                "buffered_total": _buffer_counts[session_id],
                "buffer_len": segmenter.buffer_len(session_id),
                "status": "ok",
            }
            if emitted_token is not None:
                reply["token"] = emitted_token.model_dump()

            await websocket.send_text(json.dumps(reply))
    except WebSocketDisconnect:
        # Flush any remaining buffered frames for this session on disconnect.
        try:
            final_token = segmenter.flush(session_id)
            if final_token is not None:
                await websocket.send_text(json.dumps({
                    "session_id": session_id,
                    "token": final_token.model_dump(),
                    "flushed": True,
                    "status": "ok",
                }))
        except Exception:
            pass
        logger.info("WebSocket client disconnected: %s", session_id)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
