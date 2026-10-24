# SignBridge AI Architecture

## 9-Stage Multimodal Pipeline

```mermaid
flowchart TD
    A[1. Webcam Video in Browser] --> B[2. Client-side Preprocessing]
    B --> C[3. MediaPipe Tasks Vision: Hands, Pose, Face]
    C --> D[4. Backend Sequence Model / Recognizer]
    D --> E[5. Sign Gloss Tokens + Confidences]
    E --> F[6. Multi-turn Intent & Context Resolution]
    F --> G[7. OpenRouter Generative AI Layer]
    G --> H[8. Accessible Text Response & Chips in UI]
    H --> I[9. Client Speech Synthesis TTS]
```

## Privacy Boundary
- **Client (Browser):** Camera capture, MediaPipe inference, audio synthesis.
- **Backend Server (FastAPI):** Feature normalization, sequence classification, session memory.
- **External Cloud (OpenRouter):** Text-only intent interpretation and response generation.
