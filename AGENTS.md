# AGENTS.md — Non-Negotiable Engineering Rules

This repository hosts **SignBridge AI**, a student capstone project for a Generative AI course. All AI coding agents, contributors, and automated tooling must strictly abide by the following non-negotiable engineering rules:

## 1. Non-Negotiable Engineering Rules (Verbatim)

1. **Planning first:** Produce an implementation plan before code in every phase and wait for approval.
2. **Never fabricate results, metrics, datasets, citations, or screenshots.** If something was not run, say so.
3. **Secrets:** `.env` is gitignored; a startup check refuses to run without the key and prints a helpful message without echoing it. Add a test that scans the repo (excluding `.env`) for key-like strings.
4. **Backend modules have type hints and docstrings; `ruff` clean; `pytest`.** OpenRouter calls are mocked in unit tests; one optional live smoke test runs only if `RUN_LIVE_TESTS=1`.
5. **No silent failures:** Structured logging (no PII, no raw landmark dumps, no API key) and explicit error states in the UI.
6. **Small, reviewable conventional commits.** Don't rewrite unrelated files. Justify every new dependency in the plan.
7. **Ask before any irreversible or costly action** (large downloads such as the 56.8 GB INCLUDE archive, paid model usage, deleting data). Prefer downloading only the INCLUDE category zips we need.
8. **If a requirement conflicts with reality, stop and explain with options and a recommendation instead of silently deviating.**
9. **Scope guard:** Do not add features listed as excluded in Section 3 (no chatbot replies, no assistant persona, no hearing STT, no emergency alerts).

---

## 2. Core Behavioral Invariants

### Invariant A: ONE-WAY ONLY
- The communication is **ONE-WAY: only the signer speaks**.
- The system is **NOT a chatbot** and does **NOT reply to the signer**.
- **Excluded Features:** No hearing-person speech-to-text, no assistant persona, no replies to the signer, no quick-reply suggestions, no emergency/medical alerts. Do not add them even as "nice to have".

### Invariant B: FIDELITY OVER FLUENCY
- The system speaks **AS the signer** in first person.
- The LLM may fix grammar, add articles/tense/politeness, resolve pronouns using context, and choose punctuation/question intonation.
- It must **NEVER add facts, names, numbers, places, requests, or opinions** that the signer did not sign or that are not in the stored context.
- When unsure, it returns up to 2 alternative sentences for the signer to pick from instead of guessing.

### Invariant C: PRIVACY & TRAINING/SERVING PARITY
- **Privacy Invariant:** Raw video **never** leaves the user's browser. Only numeric landmark coordinates reach the local backend, and only plain English text strings reach OpenRouter.
- **Training/Serving Parity:** Because MediaPipe implementations differ between JavaScript and Python, all primary ISL sign training data is collected through the browser `/collect` tool using `@mediapipe/tasks-vision`.
- **Human-in-the-Loop Safety:** The LLM must not silently smooth over wrong sign tokens. The UI must present tokens as editable chips with confidence shading before final synthesis.
- **Confirm Before Speaking:** Because the output is spoken aloud as the signer's own voice, the signer confirms the sentence before it is spoken (confirm step ON by default; auto-speak OFF by default).
- **Limited Scope:** Limited-vocabulary ISL (20–30 signs), focusing on reliability and conversational understanding rather than open-vocabulary translation.
