"""PyTorch sequence model architectures for ISL sign recognition.

Three architectures behind one interface:
  (a) 1D-CNN baseline
  (b) BiGRU
  (c) small Transformer encoder

All share the same forward signature: (batch, window, input_dim) -> (batch, num_classes).
"""

import math

import torch
import torch.nn as nn


class SignRecognizerBase(nn.Module):
    """Base class for ISL sign recognizers."""

    def __init__(self, num_classes: int, input_dim: int = 193, window: int = 32) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.window = window

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class CNN1DSignRecognizer(SignRecognizerBase):
    """1D-CNN baseline over the temporal feature sequence."""

    def __init__(self, num_classes: int, input_dim: int = 193, window: int = 32) -> None:
        super().__init__(num_classes, input_dim, window)
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, input_dim) -> (batch, input_dim, window)
        x = x.transpose(1, 2)
        return self.net(x)


class BiGRUSignRecognizer(SignRecognizerBase):
    """Bidirectional GRU sequence model."""

    def __init__(self, num_classes: int, input_dim: int = 193, window: int = 32) -> None:
        super().__init__(num_classes, input_dim, window)
        self.gru = nn.GRU(
            input_dim,
            64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, input_dim)
        output, _ = self.gru(x)  # (batch, window, 128)
        last = output[:, -1, :]  # (batch, 128)
        return self.head(last)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerSignRecognizer(SignRecognizerBase):
    """Small Transformer encoder over the temporal feature sequence."""

    def __init__(self, num_classes: int, input_dim: int = 193, window: int = 32) -> None:
        super().__init__(num_classes, input_dim, window)
        d_model = 64
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=window)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.3,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, input_dim)
        x = self.input_proj(x)  # (batch, window, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (batch, window, d_model)
        x = x.mean(dim=1)  # mean pool over time -> (batch, d_model)
        return self.head(x)


ARCHITECTURES: dict[str, type] = {
    "cnn1d": CNN1DSignRecognizer,
    "bigru": BiGRUSignRecognizer,
    "transformer": TransformerSignRecognizer,
}


def create_model(
    architecture: str, num_classes: int, input_dim: int = 193, window: int = 32
) -> SignRecognizerBase:
    """Factory: create a sign recognizer by architecture name."""
    key = architecture.lower()
    if key not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{architecture}'. Choices: {list(ARCHITECTURES)}"
        )
    return ARCHITECTURES[key](
        num_classes=num_classes, input_dim=input_dim, window=window
    )


def list_architectures() -> list[str]:
    """Return available architecture names."""
    return list(ARCHITECTURES)
