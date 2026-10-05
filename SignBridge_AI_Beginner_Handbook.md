# SignBridge AI: Beginner Project Handbook

**Project:** A multimodal Generative AI system that reads sign-language gestures from a webcam, turns them into text, understands the meaning, replies using a Large Language Model (LLM), and speaks the reply.

**Who this is for:** A beginner developer who follows it step by step.
**Rule #1:** Follow the phases **in order**. Do not skip ahead. Each phase ends with a "Done when" check. Do not start the next phase until you pass it.

---

## 1. The project in plain English

Imagine a Deaf person at a hospital reception desk. The receptionist doesn't know sign language.

1. The person signs in front of a camera, for example **HELP … PAIN … DOCTOR**.
2. Our program watches the camera and finds the hands and body (as dots, called *landmarks*).
3. A small AI model recognises each sign and outputs the word.
4. The word list (`help, pain, doctor`) is sent to an LLM (a chatbot-style AI).
5. The LLM builds a proper sentence ("I am in pain and need a doctor"), works out the intent, and writes a helpful reply ("Please wait, I am calling a staff member now").
6. The reply is shown on screen and **spoken aloud**.

**Why this project is special:** most sign-language projects stop at step 3. Ours adds *context and conversation* (steps 4 to 6) using Generative AI. That is the novelty.

**What this project is NOT:** It is not a full translator for the whole of Indian Sign Language (ISL). Research systems struggle even with that. We build a **reliable demo for about 10 signs** in one setting (hospital reception) and do it well.

### Honesty rules (important)
- Our model will make mistakes. So the system must **show what it recognised** and **ask for confirmation or clarification** when unsure.
- The LLM must **never invent facts** or give medical advice.
- Learn the signs from a trusted ISL source (the ISLRTC dictionary), and ideally have someone who knows ISL check your recordings.
- Never say the system "translates ISL". Say it "recognises a limited vocabulary of ISL signs".

---

## 2. Scope: what you build, in order

| Level | What | Required? |
|---|---|---|
| **MVP (must finish)** | Webcam → landmarks → recognise 10 signs → LLM sentence + reply → on-screen text + voice | **Yes** |
| Level 2 | Confirmation step, conversation memory, confidence handling | Yes |
| Level 3 | Evaluation tables, confusion matrix, test with other people | Yes |
| Stretch A | Web UI (Streamlit) | Optional |
| Stretch B | Hindi/Kannada voice output | Optional |
| Stretch C | More signs (use the public INCLUDE dataset) | Optional |
| Stretch D | Text → sign video lookup (reverse direction) | Optional |

**Scope rule:** If a new idea is not in the MVP list, write it in `BACKLOG.md` and do it only after the MVP works.

---

## 3. Key words you need to know

- **Landmark / keypoint:** An (x, y, z) position of a body or hand point, such as a fingertip. MediaPipe gives about 543 of these per frame. We use hands and upper body.
- **Frame:** One image from the video. A camera gives about 25 to 30 frames per second.
- **Sequence:** A group of frames (we use 30) that together show one sign, because a sign is movement, not a single picture.
- **Feature vector:** The list of numbers (all landmark coordinates) for one frame. Ours has 225 numbers.
- **LSTM:** A neural network layer good at *sequences* (it remembers earlier frames).
- **Epoch:** One full pass through the training data.
- **Confidence:** How sure the model is (0 to 1).
- **LLM:** A large language model (a chatbot-style AI) used through an API.
- **Prompt:** The instructions you send to the LLM.
- **TTS:** Text-to-speech.
- **Signer-independent:** The model works on people it never saw during training. This is the honest test.

---

## 4. Technology stack

| Need | Tool | Why |
|---|---|---|
| Language | Python 3.10 or 3.11 | Required for the library versions below |
| Camera | OpenCV | Standard tool for video |
| Landmarks | MediaPipe Holistic | Free, runs on a normal laptop |
| Model | TensorFlow / Keras (LSTM) | Easiest for beginners |
| Evaluation | scikit-learn, matplotlib | Metrics, confusion matrix |
| LLM | Anthropic API (or Gemini free tier, or Ollama locally) | Context and reply generation |
| Voice | pyttsx3 (offline) | Works without internet |
| Secrets | python-dotenv | Keeps your API key out of code |
| Versioning | Git + GitHub | Save progress, show your work |

---

## 5. Requirements

### 5.1 Functional requirements (what the system must do)
- **FR1:** Capture live video from a webcam.
- **FR2:** Extract hand and body landmarks from each frame.
- **FR3:** Recognise at least 10 signs plus an "idle" (no sign) state.
- **FR4:** Show recognised words and their confidence on screen.
- **FR5:** Send the words to an LLM and get back a sentence, an intent, and a reply.
- **FR6:** If confidence is low, ask the user a clarification question instead of guessing.
- **FR7:** Remember the last few conversation turns.
- **FR8:** Speak the reply aloud.

### 5.2 Non-functional requirements
- Total delay from finishing a sign to hearing the reply is under 5 seconds (target under 3).
- Works on a normal laptop (no GPU needed to *run*).
- Only landmark numbers are sent anywhere, **never video**, and only text goes to the LLM (privacy).
- The program does not crash when the camera or internet fails. It shows a message instead.

### 5.3 What you need
- Laptop with a webcam, 8 GB RAM or more, a good light source.
- Windows, Linux, or macOS Intel. (On Apple Silicon Macs, train in Google Colab.)
- Internet for installing libraries and the LLM API.
- An LLM API key (create an account with Anthropic, or use Google Gemini's free tier, or install Ollama for a free local model).
- GitHub account.
- A second person to record data and test with (very important).

### 5.4 Skills to learn first (Week 0, about 3 to 5 days)
Python basics (functions, lists, dictionaries, files, loops). NumPy arrays basics. What a neural network is (any beginner video). How to use `git add / commit / push`. You do **not** need deep maths.

---

## 6. Architecture

```
Webcam
  │
  ▼
OpenCV reads frames
  │
  ▼
MediaPipe Holistic ──► landmarks (hands + body)
  │
  ▼
Normalise (centre + scale so camera distance doesn't matter)
  │
  ▼
Sliding window of the last 30 frames
  │
  ▼
LSTM model ──► predicted sign + confidence   (ignore "idle" and low confidence)
  │
  ▼
Word buffer:  [("help",0.95), ("pain",0.91), ("doctor",0.88)]
  │   (press SPACE to send)
  ▼
LLM layer (prompt + memory)
  │   returns JSON: interpreted_sentence, intent, confidence,
  │                 needs_clarification, clarification_question, reply
  ▼
Screen text  +  Text-to-speech
```

**Design idea to remember:** The LLM receives the recognised words **with their confidence scores**. When confidence is low it must ask a question rather than guess. This is what makes the system safe and is your main "research" contribution.

---

## 7. Project folder structure

```
signbridge-ai/
├── data/
│   └── raw/                # recorded sequences (one folder per sign)
├── models/                 # saved model + labels
├── src/
│   ├── config.py
│   ├── utils.py            # landmark extraction
│   ├── collect_data.py     # record training data
│   ├── train.py            # train the model
│   ├── llm.py              # LLM layer
│   ├── tts.py              # speech
│   ├── test_llm.py         # test the LLM without the camera
│   └── main.py             # the full live app
├── results/                # confusion matrix, tables, screenshots
├── .env                    # API key (NEVER commit this)
├── .gitignore
├── requirements.txt
├── BACKLOG.md
└── README.md
```

---

## 8. Setup (do this once)

```bash
mkdir signbridge-ai && cd signbridge-ai
git init
python3.10 -m venv venv           # use "python -m venv venv" on Windows
# activate:  Windows: venv\Scripts\activate    Mac/Linux: source venv/bin/activate
mkdir -p data/raw models src results
```

Create `requirements.txt`:

```
numpy<2
mediapipe==0.10.14
opencv-python
tensorflow==2.15.0
scikit-learn
matplotlib
pyttsx3
anthropic
python-dotenv
```

```bash
pip install -r requirements.txt
```

Create `.gitignore`:

```
venv/
.env
data/
__pycache__/
```

Create `.env` (use your own key; **never** put it in code or on GitHub):

```
ANTHROPIC_API_KEY=your_key_here
```

> If `pip` reports version conflicts, check you are using Python 3.10 or 3.11 (`python --version`). Most problems come from the wrong Python version.

**Phase 0 is done when:** `python -c "import cv2, mediapipe, tensorflow; print('OK')"` prints `OK`.

---

## 9. Build phases

Run all scripts from inside the `src/` folder (`cd src`).

### Phase 1: Config and landmark extraction (Day 1 to 3)

**Goal:** See dots on your hands and body on live video.

`src/config.py`
```python
from pathlib import Path

# Demo vocabulary: 10 hospital-reception signs + "idle" (hands up but NOT signing).
# Learn the correct ISL version of each sign from a trusted ISL dictionary (ISLRTC).
SIGNS = ["hello", "thanks", "help", "water", "doctor",
         "pain", "yes", "no", "medicine", "where", "idle"]

SEQ_LEN = 30            # frames per sign
N_SEQUENCES = 40        # recordings per sign per person
FEATURES = 225          # 33 pose*3 + 21 left hand*3 + 21 right hand*3
CONF_THRESHOLD = 0.85

DATA_DIR = Path("../data/raw")
MODEL_PATH = Path("../models/sign_model.keras")
LABELS_PATH = Path("../models/labels.json")
```

`src/utils.py`
```python
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils


def detect(frame_bgr, holistic):
    """Run MediaPipe on one frame and return its results."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return holistic.process(rgb)


def _to_array(landmark_list):
    if landmark_list is None:
        return None
    return np.array([[p.x, p.y, p.z] for p in landmark_list.landmark])


def extract_keypoints(results):
    """Turn MediaPipe results into one list of 225 numbers.
    Coordinates are shifted to the shoulder centre and scaled by shoulder width,
    so it doesn't matter where you stand or how far from the camera you are."""
    pose = _to_array(results.pose_landmarks)
    lh = _to_array(results.left_hand_landmarks)
    rh = _to_array(results.right_hand_landmarks)

    if pose is not None:
        center = (pose[11] + pose[12]) / 2                       # between shoulders
        scale = np.linalg.norm(pose[11][:2] - pose[12][:2]) + 1e-6
    else:
        center, scale = np.zeros(3), 1.0

    def norm(arr, n_points):
        if arr is None:
            return np.zeros(n_points * 3)                        # missing = zeros
        return ((arr - center) / scale).flatten()

    return np.concatenate([norm(pose, 33), norm(lh, 21), norm(rh, 21)])


def draw(frame, results):
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
```

Quick test file `src/test_camera.py`:
```python
import cv2
from utils import mp_holistic, detect, extract_keypoints, draw

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        results = detect(frame, holistic)
        draw(frame, results)
        print(extract_keypoints(results).shape)      # should print (225,)
        cv2.imshow("Test", cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
```

> **Mirroring rule:** We process the *original* frame and only flip the picture for *display*. Always do it the same way in collection and in live use. Mixing them breaks left/right hands and ruins accuracy.

**Done when:** you see dots on your body and hands, and the console prints `(225,)`. Commit to Git.

---

### Phase 2: Collect training data (Week 1 to 2)

**Goal:** Record 40 or more examples of each sign, ideally from 2 or more different people.

`src/collect_data.py`
```python
import time
import cv2
import numpy as np
from config import SIGNS, SEQ_LEN, N_SEQUENCES, DATA_DIR
from utils import mp_holistic, detect, extract_keypoints, draw

person = input("Your ID (e.g. p1): ").strip()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera not found. Try cv2.VideoCapture(1).")


def show(frame, text, color=(0, 255, 0)):
    view = cv2.flip(frame, 1)
    cv2.putText(view, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Collect", view)
    return cv2.waitKey(1) & 0xFF


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for sign in SIGNS:
        folder = DATA_DIR / sign
        folder.mkdir(parents=True, exist_ok=True)
        start = len(list(folder.glob(f"{person}_*.npy")))     # continue where you stopped
        for i in range(start, start + N_SEQUENCES):
            t0 = time.time()
            while time.time() - t0 < 1.5:                      # get-ready pause
                ok, frame = cap.read()
                key = show(frame, f"GET READY: {sign} ({i - start + 1}/{N_SEQUENCES})", (0, 200, 255))
                if key == ord("q"):
                    raise SystemExit
            frames = []
            while len(frames) < SEQ_LEN:                       # record 30 frames
                ok, frame = cap.read()
                results = detect(frame, holistic)
                frames.append(extract_keypoints(results))
                draw(frame, results)
                show(frame, f"RECORDING: {sign}", (0, 0, 255))
            np.save(folder / f"{person}_{i}.npy", np.array(frames))

cap.release()
cv2.destroyAllWindows()
```

**How to record well (this decides your accuracy):**
- Perform the **complete sign** within the 1-second recording. Start the movement as the word "RECORDING" appears.
- Vary slightly: distance from camera, small position changes, lighting (day and evening), clothes, sitting vs standing.
- For **"idle"**: raise your hands as if about to sign, then make random small non-sign movements and rest poses. This teaches the model what "not signing" looks like.
- Get a **second person** to record at least 20 per sign. They will also become your test set later.
- Aim for **60 to 100 per sign** if time allows. More data beats a fancier model.

**Done when:** `data/raw/<sign>/` has at least 40 `.npy` files for every sign, from at least 2 people. Commit the code (not the data).

---

### Phase 3: Train the model (Week 3)

**Goal:** A model that guesses the sign from 30 frames.

`src/train.py`
```python
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from config import SIGNS, SEQ_LEN, FEATURES, DATA_DIR, MODEL_PATH, LABELS_PATH

# 1. Load data
X, y = [], []
for label, sign in enumerate(SIGNS):
    for f in sorted((DATA_DIR / sign).glob("*.npy")):
        X.append(np.load(f))
        y.append(label)
X, y = np.array(X), np.array(y)
print("Data shape:", X.shape)                    # (num_examples, 30, 225)

# 2. Split into train and test
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Build the model
model = Sequential([
    Input(shape=(SEQ_LEN, FEATURES)),
    LSTM(64, return_sequences=True),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(len(SIGNS), activation="softmax"),
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 4. Train (stops early when it stops improving)
model.fit(Xtr, to_categorical(ytr, len(SIGNS)),
          validation_data=(Xte, to_categorical(yte, len(SIGNS))),
          epochs=200, batch_size=16,
          callbacks=[EarlyStopping(patience=20, restore_best_weights=True)])

# 5. Save
model.save(MODEL_PATH)
json.dump(SIGNS, open(LABELS_PATH, "w"))

# 6. Evaluate
pred = np.argmax(model.predict(Xte), axis=1)
print(classification_report(yte, pred, target_names=SIGNS))
ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=SIGNS).plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig("../results/confusion_matrix.png")
```

Run: `python train.py`

**Understanding the result:**
- If both train and test accuracy are high, good.
- If train is high but test is low, the model **memorised** (overfitting). Record more varied data.
- If the confusion matrix shows two signs confused, record more examples of those or check that your signs really look different.
- **Honest test:** this split mixes people. For the real test, **train without person 2's data, then test on person 2 only**. Report both numbers. The second is lower, and that's fine. Reporting it honestly is what makes the project credible.

**Done when:** test accuracy is 85% or higher and the confusion matrix is saved in `results/`. Commit.

---

### Phase 4: The LLM layer (Week 4)

**Goal:** Turn `[("help",0.95),("pain",0.91)]` into a sentence, intent, and safe reply. Test it **without the camera** first.

`src/llm.py`
```python
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()                       # reads ANTHROPIC_API_KEY from .env
MODEL = "claude-haiku-4-5-20251001"        # small and cheap; check provider docs for current names

SYSTEM = """You are the language assistant inside SignBridge AI, used at a {domain}.
You receive signs recognised by a noisy camera model, in signing order, each with a confidence between 0 and 1.
Sign order is not normal grammar, so rebuild the most likely sentence.
Rules:
1. Never invent facts the signer did not express. Use only the given words.
2. If any confidence is below 0.85, or the meaning is ambiguous, set needs_clarification to true
   and ask ONE short yes/no or choice question.
3. Reply in at most 2 short, simple sentences.
4. Never give medical diagnoses or medicine doses. For urgent words (pain, help), say that staff are being called.
Return ONLY JSON with these keys:
interpreted_sentence, intent, confidence (0-1), needs_clarification (true/false),
clarification_question, reply."""


def interpret_and_reply(words, history, domain="hospital reception desk"):
    """words = [("help", 0.95), ("pain", 0.91)]; history = list of past messages."""
    payload = {"recognized_signs": [{"word": w, "confidence": c} for w, c in words]}
    messages = history + [{"role": "user", "content": json.dumps(payload)}]
    resp = client.messages.create(
        model=MODEL, max_tokens=400,
        system=SYSTEM.format(domain=domain), messages=messages,
    )
    text = resp.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:             # the LLM sometimes breaks the format
        return {"interpreted_sentence": " ".join(w for w, _ in words),
                "needs_clarification": True,
                "clarification_question": "Sorry, could you sign that again?",
                "reply": ""}
```

`src/test_llm.py` (test with made-up inputs)
```python
from llm import interpret_and_reply

tests = [
    [("help", 0.95), ("pain", 0.92), ("doctor", 0.90)],   # confident, urgent
    [("water", 0.88), ("where", 0.91)],                    # simple question
    [("medicine", 0.60), ("pain", 0.93)],                  # low confidence -> must ask
    [("yes", 0.97)],                                       # needs context
]
for words in tests:
    print(words)
    print(interpret_and_reply(words, []), "\n")
```

**Free alternatives:** Gemini free tier or Ollama (a local model). Keep the same function name and return format and only change the inside of the function, so the rest of the project doesn't change.

**Done when:** all 4 tests return valid JSON, the low-confidence test sets `needs_clarification: true`, and the replies never contain invented facts or medical advice. If a reply misbehaves, **improve the SYSTEM prompt** and test again. Prompt tuning is a real skill.

---

### Phase 5: Voice (Week 4)

`src/tts.py`
```python
import threading
import pyttsx3

_lock = threading.Lock()


def speak(text):
    """Speak in a background thread so the video does not freeze."""
    if not text:
        return

    def run():
        with _lock:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()
```

**Done when:** `python -c "from tts import speak; speak('Hello, how can I help you?'); import time; time.sleep(4)"` speaks aloud. On Linux you may need `sudo apt install espeak`.

---

### Phase 6: Put it all together (Week 5)

**Goal:** The complete live application.

`src/main.py`
```python
import json
import threading
from collections import deque

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import SEQ_LEN, MODEL_PATH, LABELS_PATH, CONF_THRESHOLD
from utils import mp_holistic, detect, extract_keypoints, draw
from llm import interpret_and_reply
from tts import speak

model = load_model(MODEL_PATH)
labels = json.load(open(LABELS_PATH))

window = deque(maxlen=SEQ_LEN)       # last 30 frames
words = []                           # [(word, confidence)]
history = []                         # conversation memory for the LLM
state = {"interp": "", "reply": "", "busy": False}


def send_to_llm():
    """Send the collected words to the LLM in a background thread."""
    if not words or state["busy"]:
        return
    state["busy"] = True
    snapshot = list(words)
    words.clear()

    def work():
        try:
            out = interpret_and_reply(snapshot, history)
            state["interp"] = out.get("interpreted_sentence", "")
            if out.get("needs_clarification"):
                state["reply"] = out.get("clarification_question", "Could you repeat that?")
            else:
                state["reply"] = out.get("reply", "")
            history.append({"role": "user", "content": state["interp"]})
            history.append({"role": "assistant", "content": state["reply"]})
            del history[:-8]                          # keep only the last 8 messages
            speak(state["reply"])
        except Exception as e:                        # never crash the app
            state["reply"] = f"Error: {e}"
        finally:
            state["busy"] = False

    threading.Thread(target=work, daemon=True).start()


def put(view, text, y, color=(255, 255, 255)):
    cv2.putText(view, text[:70], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


cap = cv2.VideoCapture(0)
frame_count = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        results = detect(frame, holistic)
        draw(frame, results)
        window.append(extract_keypoints(results))
        frame_count += 1

        hands_visible = (results.left_hand_landmarks is not None
                         or results.right_hand_landmarks is not None)

        # predict every 3rd frame, only when hands are visible
        if len(window) == SEQ_LEN and frame_count % 3 == 0 and hands_visible:
            probs = model.predict(np.expand_dims(np.array(window), 0), verbose=0)[0]
            k = int(np.argmax(probs))
            conf = float(probs[k])
            if conf >= CONF_THRESHOLD and labels[k] != "idle":
                words.append((labels[k], round(conf, 2)))
                window.clear()                        # avoid counting the same sign twice

        view = cv2.flip(frame, 1)
        put(view, "Signs: " + " ".join(w for w, _ in words), 30, (0, 255, 255))
        put(view, "I understood: " + state["interp"], 60)
        put(view, "Reply: " + state["reply"], 90, (0, 255, 0))
        put(view, "SPACE=send  C=clear  Q=quit" + ("  [thinking...]" if state["busy"] else ""),
            view.shape[0] - 15, (200, 200, 200))
        cv2.imshow("SignBridge AI", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            words.clear()
        elif key == 32:                               # SPACE
            send_to_llm()

cap.release()
cv2.destroyAllWindows()
```

Run: `python main.py`. Sign a few words, press **SPACE**, and wait for the reply.

**Done when:** you can sign 3 words, press SPACE, see the interpreted sentence, see the reply, and hear it spoken, without crashing. Record a screen video as a backup demo. Commit.

---

### Phase 7: Test and evaluate (Week 6 to 7)

Fill in this table in `results/`. The results matter more than the code.

**A. Recognition**

| Test | Result |
|---|---|
| Accuracy, mixed-people test set | ___ % |
| Accuracy, **new person never seen in training** | ___ % |
| Per-sign accuracy (from the confusion matrix) | table |
| Accuracy in dim light vs good light | ___ % |
| Accuracy at 1 m vs 2 m | ___ % |

**B. LLM layer.** Write 20 test inputs (words plus confidences) and score each reply yourself or with a friend.

| Measure | How | Result |
|---|---|---|
| Sentence is sensible | 1 to 5 rating, average | ___ |
| Reply is appropriate | 1 to 5 rating, average | ___ |
| Invented facts | count out of 20 | ___ |
| Asked a question when confidence was low | count correct out of the low-confidence cases | ___ |

**C. System**

| Measure | Result |
|---|---|
| Time from pressing SPACE to hearing the reply | ___ s |
| Frames per second of the live window | ___ |

**D. Mini experiment (impressive and easy):** Run the 20 tests twice. (1) Normal prompt with confidence scores. (2) Prompt that **ignores confidence and never asks questions**. Compare the "invented/wrong reply" counts. This shows that your uncertainty handling really helps.

**E. User test:** Ask 3 to 5 people (ideally including someone who knows ISL) to try it and write down what confused them.

---

## 10. Project timeline (8 weeks)

| Week | Focus | Deliverable | Done when |
|---|---|---|---|
| 0 | Learn basics, install tools | Working environment | Phase 0 test passes |
| 1 | Landmarks | Camera shows dots, `(225,)` printed | Phase 1 done |
| 1-2 | Data collection | 40+ sequences × 11 classes, 2 people | Phase 2 done |
| 3 | Train model | Model + confusion matrix | ≥85% test accuracy |
| 4 | LLM + voice | `llm.py`, `tts.py` tested | Phase 4 and 5 done |
| 5 | Integration | `main.py` works end to end | Live demo works |
| 6 | Evaluation | Results tables, new-person test | Section 9 tables filled |
| 7 | Improve | Fix the weakest signs, tune prompt, stretch goal | Metrics improved |
| 8 | Finish | README, demo video, slides | Final checklist done |

**Weekly PM routine (every Friday):**
1. What did I finish? 2. What blocked me? 3. Am I on schedule? 4. Update the `README.md` and push to GitHub.

If you fall behind, **cut stretch goals, not testing**.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` conflicts | Check you use Python 3.10 or 3.11; recreate the virtual environment |
| Camera not opening | Change `VideoCapture(0)` to `1`; close Zoom/Teams; allow camera permission |
| Landmarks jump around or disappear | Better light, plain background, keep hands inside the frame |
| Model predicts one sign for everything | Class imbalance or too little data; record more; make sure "idle" has examples |
| High train accuracy, bad live accuracy | Recording conditions differ from live use; record with variation; check mirroring is consistent |
| Same word added repeatedly | Raise `CONF_THRESHOLD` or add `window.clear()` (already included) |
| Random words appear when not signing | Add more "idle" data; keep the `hands_visible` check |
| LLM returns broken JSON | Already handled by the fallback; tighten the prompt ("ONLY JSON") |
| LLM API error | Check `.env`, internet, and your API credit; the app shows "Error" and does not crash |
| No sound | Linux: `sudo apt install espeak`; check volume; try a different TTS (gTTS) |
| Video lags | Predict every 5th frame; close other programs; lower camera resolution |

---

## 12. Risks

| Risk | Likelihood | What to do |
|---|---|---|
| Not enough data for good accuracy | High | Start recording in Week 1; get friends to help; add variation |
| Works for you but not for others | High | Record at least 2 people; always report the new-person test |
| LLM says something wrong or unsafe | Medium | Strict prompt, confirmation step, never medical advice, show the recognised words |
| Library version problems | Medium | Use the pinned versions; ask early |
| Running out of time | Medium | Freeze the 10-sign vocabulary in Week 1; stretch goals last |
| Demo fails on the day | Medium | Record a backup video; test the room's light beforehand; bring a charger and a good webcam |

---

## 13. Final deliverables checklist

- [ ] Working `main.py` with the full pipeline
- [ ] GitHub repository with clean code, `README.md`, `requirements.txt` (no API keys, no raw data)
- [ ] README with: what it is, how to install, how to run, screenshots, known limitations
- [ ] Trained model files (or a link to them)
- [ ] `results/` folder: confusion matrix, evaluation tables, new-person test results
- [ ] 2-minute demo video
- [ ] Slides: problem, gap, architecture, demo, results, limitations, future work
- [ ] Short report (4 to 6 pages): introduction, method, experiments, results, ethics, conclusion

### 3-minute demo script
1. (20 s) Problem: "Deaf people can't easily talk to people who don't know sign language."
2. (20 s) Show the architecture diagram.
3. (60 s) Live demo: sign HELLO → SPACE → hear reply. Sign HELP PAIN DOCTOR → SPACE → hear reply.
4. (30 s) Show a low-confidence case where the system **asks a question** instead of guessing.
5. (30 s) Show the results: accuracy, new-person test, LLM test table.
6. (20 s) Limitations and future work. (Be honest.)

---

## 14. Stretch goals (only after the MVP and evaluation are done)

1. **Web UI with Streamlit:** display the same pipeline in a browser.
2. **Hindi/Kannada voice:** use the `gTTS` library with `lang="hi"` or `lang="kn"`, and ask the LLM to reply in that language.
3. **More signs:** use the public INCLUDE ISL dataset (263 words) to extend the vocabulary.
4. **Better model:** replace the LSTM with a small Transformer, and compare with your baseline.
5. **Reverse direction:** typed text → show a stored sign video for each known word.
6. **Domain switch:** a "bank" or "railway" mode that changes the LLM prompt.

---

## 15. Ethics and limitations (put this in your report)

- The system recognises a **small demo vocabulary**, not the full language.
- It was trained on a few people, so it may not work equally well for everyone (different skin tones, hand sizes, signing speeds, left-handed signers, backgrounds).
- Sign language includes **facial expressions and body movement** that this MVP only partly uses.
- The LLM can make mistakes, so the system is an **assistive aid, not a replacement for human interpreters**.
- Get consent from everyone you record; store recordings privately; don't upload them publicly.
- Say clearly who tested or checked the signs. Involving a Deaf person or ISL interpreter in review is strongly recommended.

---

## 16. If you get stuck

1. Read the error message from the bottom up and copy the **last line**.
2. Search for that exact line.
3. Make the problem smaller (test one function in a tiny file).
4. Ask a mentor or an AI assistant. Give the **exact error, your Python version, and the code** around it.
5. Don't change many things at once. Change one thing, run it, and commit if it works.

**Git habit:** commit at the end of every phase with a clear message, such as `Phase 3: trained LSTM, 91% test accuracy`.
