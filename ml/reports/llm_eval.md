# LLM Layer Evaluation Report

- **Evaluation Date:** 2026-10-25 08:04:56 UTC
- **Mode:** OFFLINE SMOKE TEST (Deterministic Fallback)
- **Model:** `Offline Rule Engine`
- **Test Samples:** 65 items from `eval/gloss_to_sentence_testset.jsonl`

## Summary Metrics

| Metric | Score / Value | Target / Benchmark |
|---|---|---|
| **Intent Accuracy** | **46.2%** | >= 85% |
| **Confirmation Match Rate** | **100.0%** | >= 80% |
| **JSON Validity Rate** | **100.0%** | 100% |
| **BLEU-1 Score** | **0.492** | >= 0.500 |
| **ROUGE-L Score** | **0.382** | >= 0.500 |
| **METEOR Score** | **0.484** | >= 0.400 |
| **Latency p50 (ms)** | **0.0 ms** | <= 1000 ms |
| **Latency p95 (ms)** | **0.0 ms** | <= 2500 ms |

> **Note on Evaluation Protocol (Rule #2):**
> Live API evaluation runs only upon explicit user confirmation of model and budget.
> Results above were generated in `OFFLINE SMOKE TEST (Deterministic Fallback)` mode.
> Dataset includes fidelity traps (low-confidence / ambiguous glosses) in category
> `ambiguous_or_low_confidence` which legitimately trigger `needs_confirmation=true`.
> 