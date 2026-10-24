"""Offline and live benchmark evaluation runner for SignBridge AI LLM layer.

Evaluates Stage A and Stage B against eval/gloss_to_sentence_testset.jsonl.
Reports: BLEU, ROUGE-L, METEOR, Intent Accuracy, JSON-validity rate, and p50/p95 latency.
Writes output to ml/reports/llm_eval.md.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure backend modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import get_settings  # noqa: E402
from app.llm.pipeline import LLMPipeline  # noqa: E402
from app.schemas import RecognizedToken  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "gloss_to_sentence_testset.jsonl"
OUTPUT_REPORT_PATH = Path(__file__).resolve().parent.parent / "ml" / "reports" / "llm_eval.md"


def tokenize(text: str) -> List[str]:
    """Simple lowercase whitespace and punctuation tokenizer."""
    import re
    return re.findall(r"\w+", text.lower())


def compute_bleu(reference: str, candidate: str) -> float:
    """Compute token-level BLEU-1 precision."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not cand_tokens or not ref_tokens:
        return 0.0
    matches = sum(1 for tok in cand_tokens if tok in ref_tokens)
    return matches / len(cand_tokens)


def compute_rouge_l(reference: str, candidate: str) -> float:
    """Compute Longest Common Subsequence (LCS) F1 score."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 or n == 0:
        return 0.0

    # DP table for LCS length
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == cand_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def compute_meteor(reference: str, candidate: str) -> float:
    """Compute simplified METEOR score (harmonic mean weighted toward recall)."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    if not cand_tokens or not ref_tokens:
        return 0.0
    matches = sum(1 for tok in cand_tokens if tok in ref_tokens)
    if matches == 0:
        return 0.0
    prec = matches / len(cand_tokens)
    rec = matches / len(ref_tokens)
    # Weighted harmonic mean: F_mean with alpha=0.9 favoring recall
    f_mean = (10 * prec * rec) / (9 * prec + rec)
    return round(f_mean, 4)


async def run_evaluation(live: bool = False) -> None:
    """Run full evaluation suite across dataset."""
    settings = get_settings()

    if live and not settings.is_api_key_configured:
        print("ERROR: --live requested but OPENROUTER_API_KEY is not configured in .env.")
        sys.exit(1)

    # If running offline, force empty API key to exercise offline fallback safely
    if not live:
        settings.OPENROUTER_API_KEY = ""

    pipeline = LLMPipeline(settings=settings)

    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    items: List[Dict[str, Any]] = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Loaded {len(items)} evaluation samples from {DATASET_PATH.name}")
    print(f"Running mode: {'LIVE (OpenRouter API)' if live else 'OFFLINE / SMOKE TEST'}")

    latencies_ms: List[float] = []
    intent_correct = 0
    json_valid_count = 0
    bleu_scores: List[float] = []
    rouge_scores: List[float] = []
    meteor_scores: List[float] = []

    for idx, item in enumerate(items, 1):
        tokens = [RecognizedToken(**t) for t in item["glosses"]]
        ref_sentence = item["reference_sentence"]
        expected_intent = item["expected_intent"]

        start_time = time.perf_counter()
        try:
            interp = await pipeline.interpret(tokens, session_id=f"eval-{idx}")
            json_valid_count += 1
        except Exception:
            interp = pipeline._offline_interpret_fallback(tokens)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        latencies_ms.append(elapsed_ms)

        # Check intent match
        if interp.intent.value == expected_intent:
            intent_correct += 1

        # Text quality scores
        bleu_scores.append(compute_bleu(ref_sentence, interp.sentence))
        rouge_scores.append(compute_rouge_l(ref_sentence, interp.sentence))
        meteor_scores.append(compute_meteor(ref_sentence, interp.sentence))

    # Compute aggregates
    latencies_ms.sort()
    p50_lat = latencies_ms[len(latencies_ms) // 2] if latencies_ms else 0.0
    p95_lat = latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0
    intent_acc = (intent_correct / len(items)) * 100 if items else 0.0
    json_valid_rate = (json_valid_count / len(items)) * 100 if items else 0.0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    # Format markdown report
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode_label = "LIVE EVALUATION" if live else "OFFLINE SMOKE TEST (Deterministic Fallback)"
    model_name = settings.OPENROUTER_MODEL if live else "Offline Rule Engine"

    report_content = f"""# LLM Layer Evaluation Report

- **Evaluation Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Mode:** {mode_label}
- **Model:** `{model_name}`
- **Test Samples:** {len(items)} items from `eval/gloss_to_sentence_testset.jsonl`

## Summary Metrics

| Metric | Score / Value | Target / Benchmark |
|---|---|---|
| **Intent Accuracy** | **{intent_acc:.1f}%** | >= 85% |
| **JSON Validity Rate** | **{json_valid_rate:.1f}%** | 100% |
| **BLEU-1 Score** | **{avg_bleu:.3f}** | >= 0.500 |
| **ROUGE-L Score** | **{avg_rouge:.3f}** | >= 0.500 |
| **METEOR Score** | **{avg_meteor:.3f}** | >= 0.400 |
| **Latency p50 (ms)** | **{p50_lat:.1f} ms** | <= 1000 ms |
| **Latency p95 (ms)** | **{p95_lat:.1f} ms** | <= 2500 ms |

> **Note on Evaluation Protocol (Rule #2):**
> Live API evaluation runs only upon explicit user confirmation of model and budget. 
> Results above were generated in `{mode_label}` mode.
"""

    OUTPUT_REPORT_PATH.write_text(report_content, encoding="utf-8")
    print(f"\nReport successfully generated: {OUTPUT_REPORT_PATH}")
    print(f"Intent Accuracy: {intent_acc:.1f}% | BLEU-1: {avg_bleu:.3f} | ROUGE-L: {avg_rouge:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignBridge AI LLM Evaluation Runner")
    parser.add_argument("--live", action="store_true", help="Run live evaluation against OpenRouter")
    args = parser.parse_args()

    asyncio.run(run_evaluation(live=args.live))
