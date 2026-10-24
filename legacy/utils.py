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