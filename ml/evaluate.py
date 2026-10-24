"""Offline evaluation for ISL sign recognition models.

Produces: top-1/top-3 accuracy, per-class precision/recall/F1, confusion matrix
PNG, per-sign latency (ms/sign) on CPU, model size (MB), and an ablation table
(with/without face features, per architecture).

Writes ml/reports/recognition_eval.md.

If the data source is the synthetic smoke dataset, the report is stamped
"SMOKE TEST — NOT A RESULT" per engineering rule #2 (never fabricate results).
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from ml.train import (
    DATA_DIR,
    INPUT_DIM,
    REPORTS_DIR,
    WINDOW,
    generate_smoke_dataset,
    load_clips,
    signer_aware_split,
)
from app.recognition.model import ARCHITECTURES, create_model, list_architectures

REPORT_PATH = REPORTS_DIR / "recognition_eval.md"
CONFUSION_PNG = REPORTS_DIR / "confusion_matrix.png"
WEIGHTS_DIR = Path("ml/weights")
SUMMARY_PATH = REPORTS_DIR / "training_summary.json"

FACE_START = INPUT_DIM - 42 - 4  # 147
FACE_END = INPUT_DIM - 4  # 189
MASK_START = INPUT_DIM - 4  # 189


def load_summary() -> Dict[str, Any]:
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r") as f:
            return json.load(f)
    return {}


def load_test_data(
    labels: List[str], is_smoke: bool
) -> Tuple[List[Dict[str, Any]], bool]:
    """Load test clips. If no real data, generate smoke data (is_smoke=True)."""
    clips = load_clips()
    actually_smoke = is_smoke or len(clips) == 0
    if actually_smoke:
        clips = generate_smoke_dataset()
    _, _, test_clips = signer_aware_split(clips)
    return test_clips, actually_smoke


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, clips, label_to_idx, window=WINDOW, zero_face=False):
        self.samples = []
        self.window = window
        self.zero_face = zero_face
        for clip in clips:
            feats = np.asarray(clip["features"], dtype=np.float32)
            if zero_face:
                feats = feats.copy()
                feats[:, FACE_START:FACE_END] = 0.0
                # face mask flag -> 0 when face zeroed
                feats[:, MASK_START + 3] = 0.0
            self.samples.append((feats, label_to_idx[clip["label"]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, label = self.samples[idx]
        T = feats.shape[0]
        if T >= self.window:
            feats = feats[: self.window]
        else:
            pad = np.zeros((self.window - T, feats.shape[1]), dtype=np.float32)
            feats = np.vstack([feats, pad])
        return torch.from_numpy(feats), label


def eval_model_on_loader(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    top3_correct = 0
    total = 0
    latencies_ms = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            start = time.perf_counter()
            logits = model(x)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms / x.size(0))
            probs = torch.softmax(logits, dim=-1)
            top1 = logits.argmax(dim=1)
            top3 = torch.topk(probs, k=min(3, num_classes), dim=1).indices
            for i in range(x.size(0)):
                all_preds.append(top1[i].item())
                all_labels.append(y[i].item())
                if y[i].item() in top3[i].tolist():
                    top3_correct += 1
            total += x.size(0)
    avg_latency = sum(latencies_ms) / max(len(latencies_ms), 1)
    return all_preds, all_labels, top3_correct, total, avg_latency


def plot_confusion_matrix(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(labels)), max(5, 0.4 * len(labels))))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def model_size_mb(ckpt_path: Path) -> float:
    if not ckpt_path.exists():
        return 0.0
    return ckpt_path.stat().st_size / (1024 * 1024)


def main() -> None:
    device = "cpu"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_summary()
    is_smoke = summary.get("smoke_mode", False)
    labels = summary.get("classes", [])
    best_arch = summary.get("best_architecture", "cnn1d")

    if not labels:
        # Fallback: infer labels from data files
        clips = load_clips()
        if len(clips) == 0:
            clips = generate_smoke_dataset()
            is_smoke = True
        labels = sorted({c["label"] for c in clips})
        best_arch = "cnn1d"

    num_classes = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    test_clips, actually_smoke = load_test_data(labels, is_smoke)
    print(f"Evaluating on {len(test_clips)} test clips ({'SMOKE' if actually_smoke else 'real'} data)")

    lines: List[str] = []
    lines.append("# ISL Sign Recognition — Evaluation Report")
    lines.append("")
    lines.append(f"- **Data source:** ml/datasets/ (gitignored)")
    lines.append(f"- **Test clips:** {len(test_clips)}")
    lines.append(f"- **Classes ({num_classes}):** {', '.join(labels)}")
    lines.append(f"- **Device:** {device} (CPU)")
    lines.append(f"- **Window:** {WINDOW} frames, {INPUT_DIM}-dim features")
    if actually_smoke:
        lines.append("")
        lines.append("> **SMOKE TEST — NOT A RESULT**")
        lines.append(">")
        lines.append("> No real ISL data exists in ml/datasets/ yet. A tiny synthetic")
        lines.append("> smoke dataset was generated ONLY to verify the pipeline runs.")
        lines.append("> Metrics below are NOT real accuracy and must not be presented as such.")
    lines.append("")

    lines.append("## Per-architecture results (full 193-dim features)")
    lines.append("")
    lines.append("| Architecture | Top-1 Acc | Top-3 Acc | Model Size (MB) | Latency (ms/sign) | Checkpoint |")
    lines.append("|---|---|---|---|---|---|")

    results: Dict[str, Dict[str, Any]] = {}
    for arch in list_architectures():
        ckpt_path = WEIGHTS_DIR / f"{arch}_best.pt"
        labels_path = WEIGHTS_DIR / f"{arch}_labels.json"
        if not ckpt_path.exists():
            lines.append(f"| {arch} | — | — | — | — | not trained |")
            results[arch] = {"trained": False}
            continue
        model = create_model(arch, num_classes=num_classes, input_dim=INPUT_DIM, window=WINDOW)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.to(device)
        ds = EvalDataset(test_clips, label_to_idx, zero_face=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
        preds, trues, top3_correct, total, avg_latency = eval_model_on_loader(
            model, loader, device, num_classes
        )
        top1_acc = sum(p == t for p, t in zip(preds, trues)) / max(total, 1)
        top3_acc = top3_correct / max(total, 1)
        size_mb = model_size_mb(ckpt_path)
        precision, recall, f1, _ = precision_recall_fscore_support(
            trues, preds, labels=list(range(num_classes)), average=None, zero_division=0
        )
        results[arch] = {
            "trained": True,
            "top1": top1_acc,
            "top3": top3_acc,
            "size_mb": size_mb,
            "latency_ms": avg_latency,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "preds": preds,
            "trues": trues,
        }
        lines.append(
            f"| {arch} | {top1_acc:.4f} | {top3_acc:.4f} | {size_mb:.2f} | {avg_latency:.2f} | {ckpt_path} |"
        )

    # Per-class P/R/F1 for best architecture
    if results.get(best_arch, {}).get("trained"):
        br = results[best_arch]
        lines.append("")
        lines.append(f"## Per-class metrics — best architecture ({best_arch})")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 |")
        lines.append("|---|---|---|---|")
        for i, label in enumerate(labels):
            lines.append(
                f"| {label} | {br['precision'][i]:.3f} | {br['recall'][i]:.3f} | {br['f1'][i]:.3f} |"
            )

    # Confusion matrix for best architecture
    if results.get(best_arch, {}).get("trained"):
        cm = confusion_matrix(br["trues"], br["preds"], labels=list(range(num_classes)))
        plot_confusion_matrix(cm, labels, CONFUSION_PNG)
        lines.append("")
        lines.append("## Confusion matrix")
        lines.append("")
        lines.append(f"![Confusion matrix]({CONFUSION_PNG.name})")

    # Ablation: with/without face features
    lines.append("")
    lines.append("## Ablation — face features (full vs face-zeroed)")
    lines.append("")
    lines.append("| Architecture | Top-1 (full) | Top-1 (no face) | Delta |")
    lines.append("|---|---|---|---|")
    for arch in list_architectures():
        ckpt_path = WEIGHTS_DIR / f"{arch}_best.pt"
        if not ckpt_path.exists() or not results.get(arch, {}).get("trained"):
            lines.append(f"| {arch} | — | — | — |")
            continue
        model = create_model(arch, num_classes=num_classes, input_dim=INPUT_DIM, window=WINDOW)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.to(device)
        # Full
        ds_full = EvalDataset(test_clips, label_to_idx, zero_face=False)
        loader_full = torch.utils.data.DataLoader(ds_full, batch_size=16, shuffle=False)
        preds_f, trues_f, _, _, _ = eval_model_on_loader(model, loader_full, device, num_classes)
        top1_full = sum(p == t for p, t in zip(preds_f, trues_f)) / max(len(trues_f), 1)
        # No face
        ds_noface = EvalDataset(test_clips, label_to_idx, zero_face=True)
        loader_noface = torch.utils.data.DataLoader(ds_noface, batch_size=16, shuffle=False)
        preds_nf, trues_nf, _, _, _ = eval_model_on_loader(
            model, loader_noface, device, num_classes
        )
        top1_noface = sum(p == t for p, t in zip(preds_nf, trues_nf)) / max(len(trues_nf), 1)
        delta = top1_full - top1_noface
        lines.append(f"| {arch} | {top1_full:.4f} | {top1_noface:.4f} | {delta:+.4f} |")

    # Velocity note
    lines.append("")
    lines.append("### Velocity features")
    lines.append("")
    lines.append(
        "The current 193-dim feature spec does not include velocity features "
        "(hands/pose/face coordinates + presence masks only). Velocity ablation "
        "is deferred; adding velocity would change the input dimension and require "
        "retraining all architectures."
    )

    # Summary
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if actually_smoke:
        lines.append("> **SMOKE TEST — NOT A RESULT.** No real ISL data has been collected yet.")
        lines.append("> Run the `/collect` page to record real sign clips, then re-run training/evaluation.")
    else:
        best = results.get(best_arch, {})
        lines.append(f"- Best architecture: **{best_arch}**")
        lines.append(f"- Best Top-1 accuracy: **{best.get('top1', 0):.4f}**")
        lines.append(f"- Confusion matrix: `{CONFUSION_PNG}`")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    if actually_smoke:
        print("\n*** SMOKE TEST — NOT A RESULT ***")


if __name__ == "__main__":
    main()
