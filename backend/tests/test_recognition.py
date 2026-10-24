"""Tests for sign recognition model architectures and inference.

Covers model creation, forward shapes, recognizer factory selection
(RECOGNIZER env), and the ModelRecognizer rejection threshold behavior.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from app.recognition.inference import ModelRecognizer, get_recognizer
from app.recognition.model import (
    BiGRUSignRecognizer,
    CNN1DSignRecognizer,
    TransformerSignRecognizer,
    create_model,
    list_architectures,
)

WINDOW = 32
INPUT_DIM = 193


def random_window(batch=2):
    return torch.randn(batch, WINDOW, INPUT_DIM)


def test_list_architectures():
    assert set(list_architectures()) == {"cnn1d", "bigru", "transformer"}


def test_create_model_unknown_arch() -> None:
    with pytest.raises(ValueError):
        create_model("nonexistent", num_classes=5)


def test_cnn1d_forward_shape():
    model = CNN1DSignRecognizer(num_classes=5, input_dim=INPUT_DIM, window=WINDOW)
    model.eval()
    with torch.no_grad():
        out = model(random_window())
    assert out.shape == (2, 5)


def test_bigru_forward_shape():
    model = BiGRUSignRecognizer(num_classes=5, input_dim=INPUT_DIM, window=WINDOW)
    model.eval()
    with torch.no_grad():
        out = model(random_window())
    assert out.shape == (2, 5)


def test_transformer_forward_shape():
    model = TransformerSignRecognizer(num_classes=5, input_dim=INPUT_DIM, window=WINDOW)
    model.eval()
    with torch.no_grad():
        out = model(random_window())
    assert out.shape == (2, 5)


def test_all_architectures_same_interface():
    for arch in list_architectures():
        model = create_model(arch, num_classes=4, input_dim=INPUT_DIM, window=WINDOW)
        model.eval()
        with torch.no_grad():
            out = model(random_window())
        assert out.shape == (2, 4), f"{arch} output shape mismatch"


def test_model_recognizer_rejection_threshold():
    labels = ["A", "B", "C"]
    rec = ModelRecognizer(
        labels=labels, input_dim=INPUT_DIM, window=WINDOW, rejection_threshold=0.99
    )
    # Random init model -> low confidence -> should reject to UNCLEAR
    result = rec.recognize_window([np.zeros(INPUT_DIM, dtype=np.float32)] * WINDOW)
    assert len(result) == 1
    assert result[0].token == "UNCLEAR"


def test_model_recognizer_high_confidence_token():
    labels = ["A", "B", "C"]
    rec = ModelRecognizer(
        labels=labels,
        input_dim=INPUT_DIM,
        window=WINDOW,
        rejection_threshold=0.0,
    )
    result = rec.recognize_window([np.zeros(INPUT_DIM, dtype=np.float32)] * WINDOW)
    assert len(result) == 1
    assert result[0].token in labels
    assert 0.0 <= result[0].confidence <= 1.0


def test_get_recognizer_mock_default(monkeypatch):
    monkeypatch.setenv("RECOGNIZER", "mock")
    rec = get_recognizer()
    from app.recognition.mock import MockRecognizer

    assert isinstance(rec, MockRecognizer)


def test_get_recognizer_model(monkeypatch):
    monkeypatch.setenv("RECOGNIZER", "model")
    rec = get_recognizer()
    assert isinstance(rec, ModelRecognizer)


def test_get_recognizer_invalid(monkeypatch) -> None:
    monkeypatch.setenv("RECOGNIZER", "bogus")
    with pytest.raises(ValueError):
        get_recognizer()
