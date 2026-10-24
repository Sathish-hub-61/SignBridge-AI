# SignBridge AI 🤟🤖

> **Bridging Indian Sign Language (ISL) and Context-Aware Generative AI**
> Student Capstone Project for Generative AI

SignBridge AI is a multimodal, privacy-first communication assistant that translates Indian Sign Language (ISL) gestures into conversational English. Unlike traditional systems that stop at sign-to-text conversion, SignBridge AI integrates a Generative AI layer (via OpenRouter) to interpret user intent, resolve ambiguity with conversation history, and generate natural, contextual spoken and text responses.

---

## 🔒 Privacy & Architecture Principles

- **Zero Video Egress:** All camera frames are processed directly inside your browser via `@mediapipe/tasks-vision`. Raw video **never** leaves the client.
- **Minimal Data Ingestion:** Only numerical landmark coordinate vectors are streamed to the backend server.
- **Separation of Concerns:** External LLMs (OpenRouter) receive **only** tokenized text strings and conversational history—never video or biometric data.
- **Human-in-the-Loop Safety:** Recognized sign tokens appear as editable chips before sentence generation, preventing hallucinated over-smoothing.

---

## 🚀 Quickstart (under 10 minutes)

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 1. Configuration
Create your local `.env` file from the provided template:
```bash
cp .env.example .env
# Edit .env and insert your OPENROUTER_API_KEY
```
The app runs fully in **mock mode** without an API key (LLM calls use a deterministic offline fallback).

### 2. Backend
```bash
cd backend
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
pytest          # run the test suite
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```
The web interface will be available at `http://localhost:5173`.

### 4. Demo flow
1. Open the frontend. Backend status shows "connected" (or "offline" if backend isn't running).
2. **Mock mode:** Type ISL glosses (e.g. `HELP I NEED DOCTOR`) → they become editable chips with confidence shading.
3. Tap **Interpret & Send** → the LLM layer returns an interpreted sentence, intent, and a streamed reply with quick replies.
4. **Camera mode (Phase 4):** Start recognition → webcam landmarks stream to the backend → recognized tokens appear as chips → confirm/edit → send.
5. **Data collection:** Switch to the Collect view → pick a label → 3-2-1 countdown → record clips → export JSON to `ml/datasets/`.
6. **Emergency:** An interpreted `emergency` intent shows a prominent card with 112 guidance.

---

## 📁 Repository Structure

```
signbridge-ai/
├── AGENTS.md                # Non-negotiable engineering rules
├── README.md                # This file
├── .env.example             # Configuration template (placeholders only)
├── backend/                 # FastAPI REST and WebSocket server
│   ├── app/
│   │   ├── main.py          # Application entrypoint
│   │   ├── config.py        # Validated configuration & settings
│   │   ├── schemas.py       # Pydantic data schemas
│   │   ├── api/             # API routes (health, chat, stream_ws, collect)
│   │   ├── recognition/     # Feature extraction, segmenter & model inference
│   │   └── llm/             # OpenRouter client, 2-stage prompt pipeline & memory
│   └── tests/               # Pytest suite with secrets scanner & mocked LLM tests
├── frontend/                # Vite + React + TypeScript
├── ml/                      # Machine learning training & evaluation pipelines
│   ├── collect/             # Data collection specification
│   ├── datasets/            # Local collected ISL datasets (gitignored)
│   ├── configs/             # Model hyperparameters & training configs
│   ├── reports/             # Quantitative evaluation reports & confusion matrices
│   ├── train.py             # Training pipeline (3 architectures, signer/session splits)
│   ├── evaluate.py          # Evaluation (top-1/3, P/R/F1, confusion matrix)
│   └── export_onnx.py       # ONNX export of best model
├── eval/                    # LLM testset, offline evaluation runner, human-rating rubric
├── docs/                    # Architecture, design decisions, evaluation, limitations, demo script
└── scripts/                 # Development lifecycle scripts
```

---

## 📋 Claims vs Evidence

Every claim in the project objectives is mapped to the file/test/report that supports it.

| Claim | Evidence |
|-------|----------|
| Zero video egress; only landmarks leave browser | `frontend/src/useSignRecognition.ts` (client-side MediaPipe), `docs/architecture.md` (privacy invariant), `README.md` privacy section |
| OpenRouter used only for language layer, never sign recognition | `backend/app/llm/openrouter_client.py`, `backend/app/llm/pipeline.py`, `docs/design-decisions.md` |
| API key never reaches frontend/logs/tests/git | `backend/tests/test_secrets.py`, `.gitignore` (`.env` excluded), `backend/app/config.py` (no key logging) |
| Two-stage Interpret/Respond pipeline | `backend/app/llm/pipeline.py`, `backend/app/llm/prompts/{interpret_v1,respond_v1}.txt`, `backend/tests/test_llm_pipeline.py` |
| Editable token chips with confidence shading | `frontend/src/App.tsx` (token chips, confidence slider), `docs/architecture.md` (human-in-the-loop safety) |
| Confirm-before-send flow | `frontend/src/App.tsx` (`handleSendTokens` after chip review), `docs/architecture.md` |
| Emergency card with 112 guidance | `frontend/src/App.tsx` (`emergency-card`), intent `emergency` routing in pipeline |
| Sliding-window segmenter | `backend/app/recognition/segmenter.py`, `backend/tests/test_segmenter.py` |
| 3 model architectures (CNN/BiGRU/Transformer) | `backend/app/recognition/model.py`, `backend/tests/test_recognition.py` |
| Signer/session-aware splits (no leakage) | `ml/train.py` (`signer_aware_split` raises on leakage) |
| Evaluation report | `ml/reports/recognition_eval.md`, `ml/reports/llm_eval.md`, `docs/evaluation.md` |
| Live eval NOT run (honesty) | `docs/evaluation.md`, `ml/reports/llm_eval.md` (offline smoke only), `eval/run_llm_eval.py` (`RUN_LIVE_TESTS=1` guard) |
| Input size limits + rate limiting | `backend/app/api/chat.py` (`_enforce_input_limits`, `_enforce_rate_limit`), `backend/tests/test_chat_limits.py` |
| CORS locked to dev origin | `backend/app/main.py` (CORS allow_origins) |
| ruff clean, pytest passing | `backend/` (`ruff check .`, `pytest`), `53 tests passing` |

---

## 📚 Documentation

- `docs/architecture.md` — 9-stage pipeline with Mermaid diagram
- `docs/design-decisions.md` — trade-offs (browser vs server landmarks, 1-call vs 2-call LLM, model choices)
- `docs/evaluation.md` — evaluation status and real results only
- `docs/limitations.md` — honest constraints (vocabulary, signer variance, ISL regional variation, lighting, single-signer risk, LLM hallucination, privacy)
- `docs/demo-script.md` — 5-minute scripted demo with fallback plan
- `eval/human_rating_rubric.md` — human-rating template (unfilled)

---

## ⚠️ Honest Limitations

See `docs/limitations.md`. Key points: limited vocabulary (~20–30 signs), single-signer training risk, ISL regional variation, lighting/camera conditions, recognition errors, LLM hallucination, no medical advice, CPU-only target, smoke-test-only results until real data is collected.
