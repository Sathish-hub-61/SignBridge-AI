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