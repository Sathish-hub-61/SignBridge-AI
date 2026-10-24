# AGENTS.md — Non-Negotiable Engineering Rules

This repository hosts **SignBridge AI**, a student capstone project for a Generative AI course. All AI coding agents, contributors, and automated tooling must strictly abide by the following non-negotiable engineering rules:

## 1. Non-Negotiable Engineering Rules (Verbatim)

1. **Planning first:** Before writing code in any phase, produce an implementation plan and wait for approval.
2. **Never fabricate results, metrics, datasets, citations, or screenshots.** If something was not run, say so.
3. **Secrets:** `.env` is gitignored; a startup check refuses to run if the key is missing and prints a helpful message without echoing the key. Add a test that greps the repo (excluding `.env`) for key-like strings.
4. **Every backend module has type hints and docstrings; `ruff` clean; tests with `pytest`.** OpenRouter calls are mocked in unit tests; one optional live smoke test runs only if `RUN_LIVE_TESTS=1`.
5. **No silent failures:** Structured logging (no PII, no raw landmark dumps, no API key), explicit error states in the UI.
6. **Small, reviewable commits with conventional messages.** Do not rewrite unrelated files. Do not add dependencies without justifying them in the plan.
7. **Ask me before any irreversible or costly action** (large downloads, paid model usage, deleting data).
8. **If a requirement here conflicts with reality** (library limitation, API change), stop and explain the conflict with options and a recommendation instead of silently deviating.

---

## 2. Architecture & Privacy Principles

- **Privacy Invariant:** Raw video **never** leaves the user's browser. Only numeric landmark coordinates are sent to the local backend, and only plain English text strings reach the external LLM (OpenRouter).
- **Training/Serving Parity:** Because MediaPipe implementations differ between JavaScript and Python, all primary ISL sign training data is collected through the browser `/collect` tool using `@mediapipe/tasks-vision`.
- **Human-in-the-Loop Safety:** The LLM must not silently smooth over wrong sign tokens. The UI must present tokens as editable chips with confidence shading before final synthesis.
- **Limited Scope:** Limited-vocabulary ISL (20–30 signs), focusing on reliability and conversational understanding rather than open-vocabulary translation.
