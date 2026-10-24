# Limitations

Honest assessment of what SignBridge AI can and cannot do. These are real constraints,
not aspirational roadmap items. Per engineering rule #2, nothing here is fabricated.

## 1. Limited vocabulary (20–30 signs)
SignBridge AI is scoped to a limited ISL vocabulary (~20–30 everyday + emergency signs,
expandable to ~50). It is **not** an open-vocabulary continuous sign-language translation
system. Signs outside the trained vocabulary are not supported.

## 2. Single-signer training risk
Training/session-aware splits require multiple signers to be statistically meaningful.
If only one signer's data exists, the model may not generalize to new signers, and that
limitation must be stated in any evaluation report. Per spec, signer/session splits are
enforced to avoid leakage-inflated accuracy.

## 3. ISL regional variation
Indian Sign Language has regional dialects (e.g., Mumbai, Delhi, Kolkata variants). A
model trained on one signer's regional variant may not transfer to signers from other
regions. This project does not cover regional adaptation.

## 4. Lighting and camera conditions
MediaPipe landmark quality degrades in low light, with motion blur, partial occlusion,
or when hands leave the frame. Quality warnings (hand out of frame, low confidence) are
shown in the UI, but the system cannot fully compensate for poor capture conditions.

## 5. Sign recognition errors
Recognition is never 100% accurate. Confusion between visually similar signs, missed
frames, and normalization drift all produce wrong tokens. The human-in-the-loop confirm
flow mitigates but does not eliminate this. Low-confidence tokens are highlighted and
can be edited or deleted before the LLM is called.

## 6. LLM hallucination
The OpenRouter Generative AI layer can "smooth over" wrong tokens into a confident but
incorrect sentence, or invent details not present in the recognized glosses. The two-stage
pipeline (Interpret + Respond) and editable token chips reduce but do not eliminate this
risk. Prompt-injection hygiene caps input length and delimits untrusted glosses.

## 7. No medical advice
For `emergency` intent, the reply is short, asks one clarifying question, and directs the
user to contact local emergency services (India: 112). SignBridge AI does **not** provide
medical diagnosis, treatment instructions, or medical advice.

## 8. Privacy boundary
Raw video never leaves the browser. Only numeric landmark coordinates go to the local
backend, and only plain English text strings reach OpenRouter. The OpenRouter API key
lives only in backend environment variables and must never reach the frontend, logs,
tests, git history, or screenshots. See `README.md` for the privacy property.

## 9. CPU-only, laptop-realtime target
Models target laptop CPU real-time inference. ONNX export is attempted; if the toolchain
is unavailable, plain PyTorch CPU inference is the fallback. Real-time performance on
lower-end hardware is not guaranteed.

## 10. Evaluation scope
- Live OpenRouter evaluation has **not** been run (requires explicit model/budget
  confirmation per engineering rule #2).
- Recognition evaluation results were generated on a clearly-labeled synthetic smoke
  dataset and are stamped "SMOKE TEST — NOT A RESULT". Real accuracy requires collected
  ISL data via the `/collect` page.
