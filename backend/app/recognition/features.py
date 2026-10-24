"""Feature extraction and coordinate normalization pipeline for SignBridge AI.

Implements the 193-dimensional feature representation documented in docs/feature-spec.md.
Ensures mathematical parity between client-side JavaScript and Python inference/training.
"""

import math
from typing import Any

# MediaPipe Pose keypoint indices (7 landmarks)
POSE_INDICES: list[int] = [0, 11, 12, 13, 14, 15, 16]

# MediaPipe FaceMesh keypoint indices (14 non-manual marker landmarks)
FACE_INDICES: list[int] = [70, 107, 300, 336, 33, 133, 263, 362, 0, 13, 14, 17, 61, 291]

# Feature layout dimensions
HAND_DIM = 21 * 3       # 63
POSE_DIM = len(POSE_INDICES) * 3  # 21
FACE_DIM = len(FACE_INDICES) * 3  # 42
MASKS_DIM = 4           # 4
FEATURE_DIM = HAND_DIM + HAND_DIM + POSE_DIM + FACE_DIM + MASKS_DIM  # 193


def calculate_torso_reference(
    pose_landmarks: list[dict[str, float]] | None,
) -> tuple[tuple[float, float, float], float]:
    """Calculate center midpoint and scale factor (shoulder distance) from pose landmarks.

    Falls back to center (0.5, 0.5, 0.0) and scale 0.25 if pose landmarks are absent.
    """
    if not pose_landmarks or len(pose_landmarks) < 13:
        return (0.5, 0.5, 0.0), 0.25

    ls = pose_landmarks[11]  # Left shoulder
    rs = pose_landmarks[12]  # Right shoulder

    cx = (ls.get("x", 0.0) + rs.get("x", 0.0)) / 2.0
    cy = (ls.get("y", 0.0) + rs.get("y", 0.0)) / 2.0
    cz = (ls.get("z", 0.0) + rs.get("z", 0.0)) / 2.0

    dx = ls.get("x", 0.0) - rs.get("x", 0.0)
    dy = ls.get("y", 0.0) - rs.get("y", 0.0)
    dz = ls.get("z", 0.0) - rs.get("z", 0.0)
    shoulder_dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    if shoulder_dist < 1e-4:
        shoulder_dist = 0.25

    return (cx, cy, cz), shoulder_dist


def normalize_point(
    x: float, y: float, z: float, center: tuple[float, float, float], scale: float
) -> tuple[float, float, float]:
    """Normalize a single 3D point relative to center and scale."""
    return (
        round((x - center[0]) / scale, 6),
        round((y - center[1]) / scale, 6),
        round((z - center[2]) / scale, 6),
    )


def extract_frame_features(frame_data: dict[str, Any]) -> list[float]:
    """Extract a standardized 193-dimensional feature vector from a single video frame.

    Args:
        frame_data: Dictionary containing:
            - 'left_hand': Optional list of 21 landmark dicts ({x, y, z})
            - 'right_hand': Optional list of 21 landmark dicts ({x, y, z})
            - 'pose': Optional list of 33 landmark dicts ({x, y, z})
            - 'face': Optional list of 468/478 landmark dicts ({x, y, z})

    Returns:
        List of 193 normalized Float32 values.
    """
    left_hand = frame_data.get("left_hand")
    right_hand = frame_data.get("right_hand")
    pose = frame_data.get("pose")
    face = frame_data.get("face")

    center, scale = calculate_torso_reference(pose)

    features: list[float] = []

    # 1. Left Hand (63 floats)
    if left_hand and len(left_hand) >= 21:
        for lm in left_hand[:21]:
            nx, ny, nz = normalize_point(lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0), center, scale)
            features.extend([nx, ny, nz])
    else:
        features.extend([0.0] * HAND_DIM)

    # 2. Right Hand (63 floats)
    if right_hand and len(right_hand) >= 21:
        for lm in right_hand[:21]:
            nx, ny, nz = normalize_point(lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0), center, scale)
            features.extend([nx, ny, nz])
    else:
        features.extend([0.0] * HAND_DIM)

    # 3. Upper Pose (21 floats)
    if pose and len(pose) >= 17:
        for idx in POSE_INDICES:
            lm = pose[idx] if idx < len(pose) else {"x": 0.0, "y": 0.0, "z": 0.0}
            nx, ny, nz = normalize_point(lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0), center, scale)
            features.extend([nx, ny, nz])
    else:
        features.extend([0.0] * POSE_DIM)

    # 4. Face Subset (42 floats)
    if face and len(face) >= 363:
        for idx in FACE_INDICES:
            lm = face[idx] if idx < len(face) else {"x": 0.0, "y": 0.0, "z": 0.0}
            nx, ny, nz = normalize_point(lm.get("x", 0.0), lm.get("y", 0.0), lm.get("z", 0.0), center, scale)
            features.extend([nx, ny, nz])
    else:
        features.extend([0.0] * FACE_DIM)

    # 5. Presence Mask Flags (4 floats)
    has_lh = 1.0 if (left_hand and len(left_hand) >= 21) else 0.0
    has_rh = 1.0 if (right_hand and len(right_hand) >= 21) else 0.0
    has_pose = 1.0 if (pose and len(pose) >= 17) else 0.0
    has_face = 1.0 if (face and len(face) >= 363) else 0.0

    features.extend([has_lh, has_rh, has_pose, has_face])

    assert len(features) == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {len(features)}"
    return features
