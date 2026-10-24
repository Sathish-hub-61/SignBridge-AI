# Design Decisions & Trade-Offs

## 1. Browser-Side Landmark Processing vs Server-Side Video
- **Decision:** Run MediaPipe Tasks Vision client-side in Web Workers; stream only landmark arrays.
- **Rationale:**
  - Privacy: Zero video transmission outside the user's personal browser.
  - Bandwidth: Streaming 225 float values at 25-30 FPS requires < 30 KB/s compared to megabytes for high-res video.
  - Server scaling: Landmark classification is lightweight CPU compute (O(1) ms) compared to heavy computer vision frame decoding.

## 2. Training/Serving Parity
- **Decision:** Collect all training samples via the browser `/collect` tool rather than offline Python OpenCV scripts.
- **Rationale:** MediaPipe JavaScript and MediaPipe Python outputs have subtle coordinate differences. Collecting data in browser guarantees perfect parity.

## 3. Two-Stage LLM Pipeline (Interpret vs Respond)
- **Decision:** Stage A outputs structured JSON (`intent`, `sentence`, `needs_confirmation`); Stage B outputs natural streaming conversational text.
- **Rationale:** Prevents hallucinated interpretations from bypassing validation; enables deterministic intent routing for emergency cases.

## 4. One-call vs Two-call LLM
- **Decision:** Two-call (Interpret then Respond) is the default; a config flag can merge A+B into one call.
- **Trade-off:**
  - Two-call: higher latency and cost, but structured intermediate output enables validation, repair retry, and human confirmation before synthesis.
  - One-call: lower latency and cost, but loses the interpretable intermediate step and makes repair/confirmation harder.
- **Documentation:** The merged flag is not exposed by default; enabling it requires updating `backend/app/llm/pipeline.py`.

## 5. Model Choices
- **Backend:** FastAPI + Uvicorn + Pydantic v2 + WebSockets.
- **Frontend:** Vite + React + TypeScript + `@mediapipe/tasks-vision`.
- **ML:** PyTorch for training; ONNX export attempted (fallback to plain PyTorch CPU inference).
- **LLM:** OpenRouter (OpenAI-compatible chat completions) only for the language layer; never for sign recognition from raw video.
- **TTS:** Browser `speechSynthesis` API by default.
