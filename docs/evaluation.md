# Evaluation Plan and Real Results

> **Engineering Rule #2:** Never fabricate results, metrics, datasets, citations, or screenshots. If something was not run, say so.

## Status
- **LLM Layer Evaluation (offline smoke test):** Run offline using a local mock fallback (no OpenRouter network calls). Results written to `ml/reports/llm_eval.md`. Live OpenRouter evaluation has **not** been run — requires explicit model and budget confirmation (engineering rule #2).
- **ISL Recognition Models:** Pipeline runs end-to-end on synthetic smoke data. Results stamped "**SMOKE TEST — NOT A RESULT**". Real accuracy requires collected ISL data via the `/collect` page. See `ml/reports/recognition_eval.md`.
- **Latency Benchmarks:** Phase 4 added capture→token and token→first-LLM-token hooks; no end-to-end latency table written yet.

## Evaluation Metrics
1. **Recognition:** Top-1 / Top-3 Accuracy, Per-class Precision/Recall/F1, Confusion Matrix, CPU Latency (ms).
   - Report: `ml/reports/recognition_eval.md` (smoke test results only).
2. **LLM Language Layer:** BLEU, ROUGE-L, METEOR, Intent Classification Accuracy, JSON Validity %, TTFT (Time to First Token).
   - Report: `ml/reports/llm_eval.md` (offline smoke test only).
   - Testset: `eval/gloss_to_sentence_testset.jsonl` (65 items).
   - Runner: `eval/run_llm_eval.py`.

## How to run
- Recognition: `python ml/train.py` then `python ml/evaluate.py` (generates `ml/reports/recognition_eval.md`).
- LLM layer (offline): `python eval/run_llm_eval.py` (uses mock fallback; no network).
- LLM layer (live): set `RUN_LIVE_TESTS=1` and confirm model/budget first.
