# SignBridge AI

A multimodal Generative AI system that reads sign-language gestures from a webcam, turns them into text, understands the meaning, replies using an LLM, and speaks the reply aloud.

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Set your API key in `.env` (never commit it):
```
ANTHROPIC_API_KEY=your_key_here
```

## Run

From `src/`:
- `python test_camera.py` — show landmarks on live video
- `python collect_data.py` — record sign sequences
- `python train.py` — train the LSTM model
- `python test_llm.py` — test the LLM layer without the camera
- `python main.py` — full live app (SPACE=send, C=clear, Q=quit)

## Status

- Phase 0 (setup): DONE — `import cv2, mediapipe, tensorflow` prints OK
- Phase 1 (landmarks): TODO
- Phase 2 (data collection): TODO
- Phase 3 (train): TODO
- Phase 4 (LLM): TODO
- Phase 5 (voice): TODO
- Phase 6 (integration): TODO
- Phase 7 (evaluation): TODO

See `SignBridge_AI_Beginner_Handbook.md` for the full plan.