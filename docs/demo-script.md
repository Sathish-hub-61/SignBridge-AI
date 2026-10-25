# SignBridge AI — 5-Minute Capstone Demo Script

## 1. Intro (1 min)
- Introduce problem: Deaf individuals communicating at hospital desks or daily services.
- Highlight innovation: One-way ISL → spoken English. Signs become a fluent first-person sentence spoken AS the signer, not a chatbot reply.
- Emphasize privacy: Raw video never leaves the browser; only numeric landmarks reach the local backend; only plain English text goes to OpenRouter.

## 2. Live Demo (3 min)
- **Flow A (Compose & Speak):** Sign or type ISL glosses (e.g. `HELP PAIN DOCTOR`) → editable confidence-shaded token chips → single Compose call → first-person sentence ("I need a doctor because I am in pain") → Confirm & Speak (confirm ON by default, auto-speak OFF).
- **Flow B (Live Video / Landmarks):** Real-time MediaPipe landmark tracking in browser; tokens stream via WebSocket; signer reviews and edits chips before composing.
- **Flow C (Show to Listener):** Full-screen high-contrast view showing the composed sentence for the hearing person to read.
- **Flow D (Spoken History + TTS):** Previously spoken sentences listed with Replay and Show-to-listener controls; voice/rate/pitch adjustable.

## 3. One-Way Design (Invariant A)
- The signer speaks; the system does NOT reply. No chatbot, no assistant turns, no quick replies, no emergency alerts.
- Confirm-before-speak is ON by default; auto-speak is OFF by default. The signer always confirms before anything is spoken aloud.
- Up to 2 alternative sentences are offered when the LLM is uncertain; the signer picks one.

## 4. Fallback Plan (Contingency)
- If camera fails: toggle Mock Mode to type glosses and exercise the full Compose → TTS layer.
- If OpenRouter API is unavailable: deterministic offline rule-based fallback composes a sentence so the demo never hard-fails.
