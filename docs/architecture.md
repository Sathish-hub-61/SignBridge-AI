# SignBridge AI Architecture

## One-Way Compose Pipeline (Part A)

```mermaid
flowchart TD
    A[1. Webcam Video in Browser<br/>Raw video never leaves browser] --> B[2. MediaPipe Tasks Vision<br/>Hands, Pose, Face landmarks]
    B --> C[3. Local Landmark Normalization<br/>193-dim feature vector per frame]
    C --> D[4. Session Segmenter<br/>Sliding window W=32, stride=16, session-isolated buffers]
    D --> E[5. Sign Recognition Model<br/>CNN1D / BiGRU / Transformer → gloss token + confidence]
    E --> F[6. Editable Token Chips UI<br/>Signer reviews, edits, confidence-shades tokens]
    F --> G[7. Single Compose Call<br/>Glosses + scenario + signer profile + history → first-person sentence]
    G --> H[8. Confirm Before Speak<br/>Signer confirms sentence; up to 2 alternatives if uncertain]
    H --> I[9. Client TTS<br/>Sentence spoken aloud as the signer's voice]
    I --> J[10. Show to Listener<br/>Full-screen high-contrast view for hearing person]
```

## Privacy Boundary (Invariant C)
- **Client (Browser):** Camera capture, MediaPipe inference, TTS audio synthesis. Raw video never leaves the browser.
- **Backend Server (FastAPI):** Feature normalization, sequence classification, session memory (signer-only turns), Compose LLM call.
- **External Cloud (OpenRouter):** Text-only — receives plain English prompt (glosses + scenario + profile), returns a JSON sentence. No video, no audio, no PII.

## Data Flow (Invariant A — One-Way Only)
1. Signer produces hand gestures + facial expressions.
2. Browser captures landmarks via MediaPipe Tasks Vision; streams to backend over WebSocket.
3. Backend segments frames into windows, recognizes sign gloss tokens with confidence.
4. Frontend presents tokens as editable chips; signer reviews/edits before composing.
5. One Compose call turns glosses into a first-person sentence spoken AS the signer.
6. Signer confirms the sentence before it is spoken (confirm ON by default; auto-speak OFF).
7. TTS speaks the sentence; it is added to spoken history; optionally shown full-screen to the listener.

## No Assistant Turns
- The system is NOT a chatbot. There are no assistant replies, no quick replies, no emergency/medical alerts, and no hearing-person speech-to-text.
- Memory stores only the signer's composed sentences (one-way memory).