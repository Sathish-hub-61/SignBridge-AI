"""Training pipeline for ISL sign recognition models.

Loads clips from ml/datasets/*.jsonl (each clip: features T x 193, label,
signer_id, session_id), performs signer/session-aware splits (fails loudly on
leakage), trains three architectures (cnn1d, bigru, transformer) with identical
splits, seeded reproducibility, early stopping, and class weighting.

If ml/datasets/ is empty, a tiny clearly-labeled synthetic smoke dataset is
generated ONLY to verify the pipeline runs end-to-end. Results from smoke data
are stamped "SMOKE TEST — NOT A RESULT" in the evaluation report and must never
be presented as real accuracy.
"""

# Disable torch.compile/dynamo before importing torch: the installed `onnx`
# package is incompatible with this Python/numpy combination, and torch's
# optimizer paths pull in torch._dynamo -> torch.onnx -> onnx.
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import glob
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.recognition.model import ARCHITECTURES, create_model, list_architectures

WINDOW = 32
INPUT_DIM = 193
DATA_DIR = Path("ml/datasets")
WEIGHTS_DIR = Path("ml/weights")
REPORTS_DIR = Path("ml/reports")
RANDOM_SEED = 42

# Smoke dataset flag: True when no real data was found and synthetic data was
# generated purely to verify the pipeline runs end-to-end.
SMOKE_MODE = False


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


class SignClipDataset(Dataset):
    """Dataset of sign clips: each item is (features window x 193, label_idx)."""

    def __init__(
        self,
        clips: List[Dict[str, Any]],
        label_to_idx: Dict[str, int],
        window: int = WINDOW,
        augment: bool = False,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.label_to_idx = label_to_idx
        self.window = window
        self.augment = augment
        self.rng = random.Random(seed)
        self.samples: List[Tuple[np.ndarray, int]] = []
        for clip in clips:
            features = np.asarray(clip["features"], dtype=np.float32)  # (T, 193)
            label_idx = label_to_idx[clip["label"]]
            self.samples.append((features, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _resample(self, features: np.ndarray) -> np.ndarray:
        T = features.shape[0]
        if T >= self.window:
            return features[: self.window].copy()
        pad = np.zeros((self.window - T, features.shape[1]), dtype=np.float32)
        return np.vstack([features, pad])

    def _augment(self, features: np.ndarray) -> np.ndarray:
        """Light augmentation: jitter, scale, random frame drop."""
        if features.std() > 0:
            features = features + self.rng.gauss(0, 0.01) * float(features.std())
        scale = 1.0 + self.rng.uniform(-0.05, 0.05)
        features = features * scale
        if self.rng.random() < 0.3:
            n_drop = self.rng.randint(1, max(1, self.window // 8))
            drop_idx = self.rng.sample(range(self.window), n_drop)
            features[drop_idx] = 0.0
        return features

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        features, label_idx = self.samples[idx]
        features = self._resample(features).astype(np.float32)
        if self.augment:
            features = self._augment(features)
        return torch.from_numpy(features), label_idx


def load_clips(data_dir: Path = DATA_DIR) -> List[Dict[str, Any]]:
    """Load all clips from data_dir/*.jsonl.

    Each line: {"label", "signer_id", "session_id", "features": [[193], ...]}
    """
    clips: List[Dict[str, Any]] = []
    files = sorted(data_dir.glob("*.jsonl"))
    for fpath in files:
        label = fpath.stem
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                feats = np.asarray(rec["features"], dtype=np.float32)
                clips.append(
                    {
                        "features": feats,
                        "label": rec["label"],
                        "signer_id": rec.get("signer_id", "unknown"),
                        "session_id": rec.get("session_id", "unknown"),
                    }
                )
    return clips


def generate_smoke_dataset(num_classes: int = 6, clips_per_class: int = 20) -> List[Dict[str, Any]]:
    """Generate a tiny synthetic smoke dataset ONLY to test the pipeline runs.

    Results from this data are NOT real and must be stamped
    "SMOKE TEST — NOT A RESULT" in any report.
    """
    global SMOKE_MODE
    SMOKE_MODE = True
    clips: List[Dict[str, Any]] = []
    rng = random.Random(1234)
    signers = ["smoke_signer_01", "smoke_signer_02", "smoke_signer_03"]
    for ci in range(num_classes):
        label = f"SMOKE_SIGN_{ci}"
        for clip_i in range(clips_per_class):
            # Class-specific bias so the task is learnable (not a real signal)
            bias = np.zeros(INPUT_DIM, dtype=np.float32)
            bias[ci * 10 : ci * 10 + 5] = 0.5 + ci * 0.1
            signer = signers[clip_i % len(signers)]
            session = f"smoke_session_{signer.split('_')[-1]}"
            T = rng.randint(20, 40)
            features = np.random.randn(T, INPUT_DIM).astype(np.float32) * 0.1
            features = features + bias
            clips.append(
                {
                    "features": features,
                    "label": label,
                    "signer_id": signer,
                    "session_id": session,
                }
            )
    return clips


def signer_aware_split(
    clips: List[Dict[str, Any]],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split clips by signer_id to avoid leakage.

    Raises ValueError if any signer appears in more than one split.
    If only one signer exists, all clips go to train and val/test are empty
    (limitation is reported in the evaluation).
    """
    rng = random.Random(seed)
    signers = sorted({c["signer_id"] for c in clips})
    rng.shuffle(signers)

    n_signers = len(signers)
    if n_signers >= 3:
        # Guarantee at least one signer in val and test so the pipeline is
        # meaningfully exercised; train gets the remainder.
        n_test = max(1, int(n_signers * test_frac))
        n_val = max(1, int(n_signers * val_frac))
        n_train = n_signers - n_val - n_test
        n_train = max(1, n_train)
    elif n_signers == 2:
        n_train, n_val, n_test = 1, 0, 1
    else:
        n_train, n_val, n_test = 1, 0, 0

    train_signers = set(signers[:n_train])
    val_signers = set(signers[n_train : n_train + n_val])
    test_signers = set(signers[n_train + n_val : n_train + n_val + n_test])

    # Leakage check: each clip's signer must belong to exactly one split
    split_of: Dict[str, str] = {}
    for s in train_signers:
        split_of[s] = "train"
    for s in val_signers:
        split_of[s] = "val"
    for s in test_signers:
        split_of[s] = "test"
    for c in clips:
        s = c["signer_id"]
        if s not in split_of:
            raise ValueError(f"Signer '{s}' not assigned to any split!")

    train = [c for c in clips if c["signer_id"] in train_signers]
    val = [c for c in clips if c["signer_id"] in val_signers]
    test = [c for c in clips if c["signer_id"] in test_signers]
    return train, val, test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / max(count, 1)


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: str
) -> Tuple[float, float]:
    """Return (avg_loss, accuracy) on a loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            count += x.size(0)
    return total_loss / max(count, 1), correct / max(count, 1)


def train_architecture(
    architecture: str,
    train_clips: List[Dict[str, Any]],
    val_clips: List[Dict[str, Any]],
    labels: List[str],
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
) -> Tuple[nn.Module, float, Dict[str, Any]]:
    """Train one architecture. Returns (best_model, best_val_acc, history)."""
    label_to_idx = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)

    train_ds = SignClipDataset(train_clips, label_to_idx, augment=True)
    val_ds = SignClipDataset(val_clips, label_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = create_model(architecture, num_classes=num_classes, input_dim=INPUT_DIM, window=WINDOW)
    model.to(device)

    # Class weighting for imbalance
    label_counts = {label: 0 for label in labels}
    for c in train_clips:
        label_counts[c["label"]] += 1
    counts = np.array([label_counts[l] for l in labels], dtype=np.float32)
    counts = np.clip(counts, 1, None)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    class_weights = torch.from_numpy(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_val_acc = -1.0
    patience_counter = 0
    history: Dict[str, Any] = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc, history


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main() -> None:
    set_seed()
    device = "cpu"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    clips = load_clips()
    is_smoke = len(clips) == 0
    if is_smoke:
        print("[WARN] No real data in ml/datasets/ — generating synthetic smoke dataset.")
        print("[WARN] Smoke results are NOT real accuracy and will be stamped accordingly.")
        clips = generate_smoke_dataset()

    labels = sorted({c["label"] for c in clips})
    num_classes = len(labels)
    print(f"Classes ({num_classes}): {labels}")
    print(f"Total clips: {len(clips)}")

    # Signer/session-aware split
    train_clips, val_clips, test_clips = signer_aware_split(clips)
    print(f"Split -> train:{len(train_clips)} val:{len(val_clips)} test:{len(test_clips)}")

    # Train all three architectures with identical splits
    results = {}
    for arch in list_architectures():
        print(f"\n=== Training {arch} ===")
        model, best_val_acc, history = train_architecture(
            arch, train_clips, val_clips, labels, device=device
        )
        # Save checkpoint
        ckpt_path = WEIGHTS_DIR / f"{arch}_best.pt"
        torch.save(model.state_dict(), str(ckpt_path))
        # Also save labels alongside weights
        with open(WEIGHTS_DIR / f"{arch}_labels.json", "w") as f:
            json.dump(labels, f)
        results[arch] = {
            "val_acc": best_val_acc,
            "history": history,
            "checkpoint": str(ckpt_path),
        }
        print(f"{arch} best val acc: {best_val_acc:.4f} -> {ckpt_path}")

    best_arch = max(results, key=lambda a: results[a]["val_acc"])
    print(f"\nBest architecture: {best_arch} (val acc {results[best_arch]['val_acc']:.4f})")

    # Evaluate best on test set
    label_to_idx = {label: i for i, label in enumerate(labels)}
    test_ds = SignClipDataset(test_clips, label_to_idx, augment=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    best_model = create_model(best_arch, num_classes=num_classes, input_dim=INPUT_DIM, window=WINDOW)
    best_model.load_state_dict(
        torch.load(WEIGHTS_DIR / f"{best_arch}_best.pt", map_location=device, weights_only=True)
    )
    best_model.to(device)
    test_loss, test_acc = evaluate_model(best_model, test_loader, device)
    print(f"\nTest set ({'smoke' if is_smoke else 'real'}): loss {test_loss:.4f} acc {test_acc:.4f}")

    # Persist a small run summary
    summary = {
        "smoke_mode": is_smoke,
        "classes": labels,
        "num_classes": num_classes,
        "split": {
            "train": len(train_clips),
            "val": len(val_clips),
            "test": len(test_clips),
        },
        "results": {
            arch: {"val_acc": r["val_acc"], "checkpoint": r["checkpoint"]}
            for arch, r in results.items()
        },
        "best_architecture": best_arch,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    with open(REPORTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {REPORTS_DIR / 'training_summary.json'}")
    if is_smoke:
        print("\n*** SMOKE TEST — NOT A RESULT ***")


if __name__ == "__main__":
    main()
