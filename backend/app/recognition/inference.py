"""Model inference backend (PyTorch / ONNX) for SignBridge AI.

Implements ModelRecognizer with the same interface as mock.py:
    recognize_window(frames_tensor) -> list[RecognizedToken]

Selectable via env RECOGNIZER=mock|model through get_recognizer().
"""

import os

import numpy as np
import torch

from app.recognition.model import create_model
from app.schemas import RecognizedToken


class ModelRecognizer:
    """PyTorch-based sign recognizer using a trained sequence model."""

    def __init__(
        self,
        model_path: str | None = None,
        labels: list[str] | None = None,
        input_dim: int = 193,
        window: int = 32,
        architecture: str = "cnn1d",
        rejection_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.window = window
        self.architecture = architecture
        self.rejection_threshold = rejection_threshold
        self.device = device
        self.labels = labels or []
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

        self.model = create_model(
            architecture,
            num_classes=len(self.labels),
            input_dim=input_dim,
            window=window,
        )
        self.model.to(device)
        self.model.eval()

        if model_path:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Load trained weights from a checkpoint file."""
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)

    def _preprocess(self, frames_tensor: list) -> torch.Tensor:
        """Convert a list of 193-dim feature vectors to (1, window, 193) tensor."""
        arr = np.array(frames_tensor, dtype=np.float32)  # (T, 193)
        if arr.shape[0] >= self.window:
            arr = arr[: self.window]
        else:
            pad = np.zeros(
                (self.window - arr.shape[0], self.input_dim), dtype=np.float32
            )
            arr = np.vstack([arr, pad])
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, window, 193)

    def recognize_window(self, frames_tensor: list) -> list[RecognizedToken]:
        """Run inference on a window of frame features.

        Returns the top token with calibrated confidence. Below
        rejection_threshold returns an "UNCLEAR" token with low confidence.
        """
        if not self.labels:
            return [RecognizedToken(token="UNCLEAR", confidence=0.0, timestamp_ms=0)]

        x = self._preprocess(frames_tensor)
        with torch.no_grad():
            logits = self.model(x)  # (1, num_classes)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_classes,)

        top_prob, top_idx = torch.max(probs, dim=0)
        top_prob = float(top_prob.item())
        top_idx = int(top_idx.item())

        if top_prob < self.rejection_threshold:
            return [RecognizedToken(token="UNCLEAR", confidence=top_prob, timestamp_ms=0)]

        label = self.idx_to_label.get(top_idx, f"CLASS_{top_idx}")
        return [RecognizedToken(token=label, confidence=top_prob, timestamp_ms=0)]


def get_recognizer(recognizer_type: str | None = None):
    """Factory: return a recognizer based on RECOGNIZER env (mock|model)."""
    rtype = (recognizer_type or os.environ.get("RECOGNIZER", "mock")).lower()
    if rtype == "mock":
        from app.recognition.mock import MockRecognizer

        return MockRecognizer()
    if rtype == "model":
        return ModelRecognizer()
    raise ValueError(f"Unknown RECOGNIZER value: {rtype}. Use 'mock' or 'model'.")
