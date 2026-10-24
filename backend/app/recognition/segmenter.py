"""Sliding-window segmenter for landmark streams (Phase 4).

Buffers per-session landmark frame feature vectors and, whenever a full
window is available, runs the configured recognizer (mock or model) to emit
a single :class:`app.schemas.RecognizedToken`.

The segmenter is session-isolated: each ``session_id`` maintains its own
frame buffer. A global singleton ``segmenter`` is exposed for use by the
WebSocket stream endpoint.
"""

import logging
import time
from collections import defaultdict, deque

from app.recognition.inference import get_recognizer
from app.schemas import RecognizedToken

logger = logging.getLogger("signbridge.segmenter")

WINDOW_SIZE = 32
STRIDE = 16


class SessionSegmenter:
    """Sliding-window segmenter that turns frame features into tokens."""

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        recognizer: object | None = None,
    ) -> None:
        self.window_size = window_size
        self.stride = max(1, stride)
        self.recognizer = recognizer or get_recognizer()
        self._buffers: dict[str, deque[list[float]]] = defaultdict(deque)
        logger.info(
            "Segmenter initialized (window=%d, stride=%d, recognizer=%s)",
            window_size,
            self.stride,
            type(self.recognizer).__name__,
        )

    def add_frame(self, session_id: str, frame_features: list[float]) -> RecognizedToken | None:
        """Append one frame's 193-dim features; return a token when a window completes."""
        buf = self._buffers[session_id]
        buf.append(list(frame_features))
        if len(buf) >= self.window_size:
            window = [buf[i] for i in range(self.window_size)]
            # Advance the buffer by `stride` (overlapping windows).
            for _ in range(min(self.stride, len(buf))):
                buf.popleft()
            token = self.recognizer.recognize_window(window)
            now_ms = int(time.time() * 1000)
            if token.timestamp_ms is None:
                token.timestamp_ms = now_ms
            logger.debug(
                "Segmenter emitted token=%s confidence=%.2f session=%s",
                token.token,
                token.confidence,
                session_id,
            )
            return token
        return None

    def flush(self, session_id: str) -> RecognizedToken | None:
        """Emit a final token for any remaining buffered frames (end of stream)."""
        buf = self._buffers.get(session_id)
        if buf and len(buf) > 0:
            window = [buf[i] for i in range(len(buf))]
            buf.clear()
            token = self.recognizer.recognize_window(window)
            now_ms = int(time.time() * 1000)
            if token.timestamp_ms is None:
                token.timestamp_ms = now_ms
            return token
        return None

    def reset(self, session_id: str) -> None:
        """Drop the buffered frames for a session."""
        self._buffers.pop(session_id, None)

    def buffer_len(self, session_id: str) -> int:
        return len(self._buffers.get(session_id, []))


# Global singleton used by the WebSocket stream endpoint.
segmenter = SessionSegmenter()
