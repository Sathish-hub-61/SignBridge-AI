"""Tests for the sliding-window segmenter (Phase 4).

Covers window completion, stride overlap, flush, reset, and the recognizer
selection via get_recognizer() (mock vs model).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from app.recognition.inference import ModelRecognizer, get_recognizer
from app.recognition.segmenter import STRIDE, WINDOW_SIZE, SessionSegmenter
from app.schemas import RecognizedToken


class _FakeRecognizer:
    """Deterministic fake recognizer returning a fixed token."""

    def __init__(self, token: str = "HELP", confidence: float = 0.9) -> None:
        self.token = token
        self.confidence = confidence
        self.calls = 0
        self.windows_seen = []

    def recognize_window(self, frames_tensor):
        self.calls += 1
        self.windows_seen.append([list(f) for f in frames_tensor])
        return RecognizedToken(
            token=self.token,
            confidence=self.confidence,
            timestamp_ms=None,
        )


def test_segmenter_emits_token_on_window_complete():
    rec = _FakeRecognizer()
    seg = SessionSegmenter(window_size=4, stride=2, recognizer=None)
    seg.recognizer = rec

    frames = [[float(i)] for i in range(6)]  # 6 frames, window 4

    emitted = []
    for f in frames:
        tok = seg.add_frame("s1", f)
        if tok:
            emitted.append(tok)

    # Windows: [0,1,2,3] then stride-2 -> [2,3,4,5] => 2 emissions
    assert len(emitted) == 2
    assert emitted[0].token == "HELP"
    assert rec.calls == 2


def test_segmenter_stride_overlap():
    rec = _FakeRecognizer()
    seg = SessionSegmenter(window_size=4, stride=2)
    seg.recognizer = rec

    # 8 frames -> windows [0,1,2,3], [2,3,4,5], [4,5,6,7] => 3 emissions
    frames = [[float(i)] for i in range(8)]
    emitted = []
    for f in frames:
        tok = seg.add_frame("s1", f)
        if tok:
            emitted.append(tok)

    assert len(emitted) == 3
    assert rec.calls == 3


def test_segmenter_flush_emits_remaining():
    rec = _FakeRecognizer()
    seg = SessionSegmenter(window_size=4, stride=2)
    seg.recognizer = rec

    for i in range(3):
        seg.add_frame("s1", [float(i)])

    # No window complete yet -> nothing emitted during add_frame
    assert rec.calls == 0

    tok = seg.flush("s1")
    assert tok is not None
    assert tok.token == "HELP"
    assert rec.calls == 1


def test_segmenter_reset_clears_buffer():
    rec = _FakeRecognizer()
    seg = SessionSegmenter(window_size=4, stride=2)
    seg.recognizer = rec

    for i in range(4):
        seg.add_frame("s1", [float(i)])  # emits on 4th

    assert rec.calls == 1
    assert seg.buffer_len("s1") == 2  # stride 2 left 2 frames

    seg.reset("s1")
    assert seg.buffer_len("s1") == 0

    # After reset, need a full new window to emit
    for i in range(4):
        seg.add_frame("s1", [float(i)])
    assert rec.calls == 2


def test_segmenter_session_isolation():
    rec = _FakeRecognizer()
    seg = SessionSegmenter(window_size=3, stride=1)
    seg.recognizer = rec

    # Session A completes a window
    for i in range(3):
        seg.add_frame("sess-a", [float(i)])
    # Session B only 2 frames -> no emit
    for i in range(2):
        seg.add_frame("sess-b", [float(i)])

    assert rec.calls == 1
    assert seg.buffer_len("sess-a") == 2  # stride 1 left 2 frames
    assert seg.buffer_len("sess-b") == 2


def test_get_recognizer_mock_default(monkeypatch):
    monkeypatch.setenv("RECOGNIZER", "mock")
    from app.recognition.mock import MockRecognizer

    assert isinstance(get_recognizer(), MockRecognizer)


def test_get_recognizer_model(monkeypatch):
    monkeypatch.setenv("RECOGNIZER", "model")
    rec = get_recognizer()
    assert isinstance(rec, ModelRecognizer)


def test_get_recognizer_invalid(monkeypatch) -> None:
    monkeypatch.setenv("RECOGNIZER", "bogus")
    with pytest.raises(ValueError):
        get_recognizer()


def test_segmenter_defaults():
    seg = SessionSegmenter()
    assert seg.window_size == WINDOW_SIZE
    assert seg.stride == STRIDE
