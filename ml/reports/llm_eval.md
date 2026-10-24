# LLM Layer Evaluation Report

- **Evaluation Date:** 2026-10-24 17:23:46 UTC
- **Mode:** OFFLINE SMOKE TEST (Deterministic Fallback)
- **Model:** `Offline Rule Engine`
- **Test Samples:** 65 items from `eval/gloss_to_sentence_testset.jsonl`

## Summary Metrics

| Metric | Score / Value | Target / Benchmark |
|---|---|---|
| **Intent Accuracy** | **72.3%** | >= 85% |
| **JSON Validity Rate** | **100.0%** | 100% |
| **BLEU-1 Score** | **0.328** | >= 0.500 |
| **ROUGE-L Score** | **0.315** | >= 0.500 |
| **METEOR Score** | **0.392** | >= 0.400 |
| **Latency p50 (ms)** | **0.0 ms** | <= 1000 ms |
| **Latency p95 (ms)** | **0.0 ms** | <= 2500 ms |

> **Note on Evaluation Protocol (Rule #2):**
> Live API evaluation runs only upon explicit user confirmation of model and budget. 
> Results above were generated in `OFFLINE SMOKE TEST (Deterministic Fallback)` mode.
