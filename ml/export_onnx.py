"""Export the best trained recognition model to ONNX.

Attempts ONNX export via torch.onnx.export. Falls back to a plain PyTorch
checkpoint if the onnx toolchain is unavailable, and documents the fallback in
ml/reports/recognition_eval.md-adjacent note (printed to stdout).

The exported model expects input (batch, window, input_dim) and emits
(batch, num_classes) logits.
"""

import json
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from ml.train import INPUT_DIM, REPORTS_DIR, WINDOW, WEIGHTS_DIR
from app.recognition.model import create_model

EXPORT_PATH = WEIGHTS_DIR / "best_model.onnx"
SUMMARY_PATH = REPORTS_DIR / "training_summary.json"


def load_summary() -> dict:
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r") as f:
            return json.load(f)
    return {}


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_summary()
    labels = summary.get("classes", [])
    best_arch = summary.get("best_architecture", "cnn1d")

    if not labels:
        print("[export_onnx] No training summary found — nothing to export.")
        return

    num_classes = len(labels)
    device = "cpu"

    model = create_model(
        best_arch, num_classes=num_classes, input_dim=INPUT_DIM, window=WINDOW
    )
    ckpt_path = WEIGHTS_DIR / f"{best_arch}_best.pt"
    if not ckpt_path.exists():
        print(f"[export_onnx] Checkpoint not found: {ckpt_path}")
        return
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    dummy_input = torch.randn(1, WINDOW, INPUT_DIM, device=device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(EXPORT_PATH),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"[export_onnx] Exported {best_arch} -> {EXPORT_PATH}")
        print(f"[export_onnx] Input: (batch, {WINDOW}, {INPUT_DIM})  Output: (batch, {num_classes})")
    except Exception as exc:
        print(f"[export_onnx] ONNX export failed: {exc}")
        print("[export_onnx] Falling back: PyTorch checkpoint remains the inference source.")
        print("[export_onnx] Install a working onnx package to enable ONNX export.")


if __name__ == "__main__":
    main()
