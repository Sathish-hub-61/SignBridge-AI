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