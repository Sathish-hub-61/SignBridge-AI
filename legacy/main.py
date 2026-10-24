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