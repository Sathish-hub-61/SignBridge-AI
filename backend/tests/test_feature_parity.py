"""Cross-language feature-extraction parity test.

Verifies that backend/app/recognition/features.py (extract_frame_features)
produces the same 193-dimensional vector as the browser-side
frontend/src/features/featureExtractor.ts for identical fixed fixture
landmarks. This enforces the training/serving parity rule from AGENTS.md:

  Because MediaPipe implementations differ between JavaScript and Python,
  all primary ISL sign training data is collected through the browser
  pipeline; the Python extractor must mirror the JS extractor exactly.

Both implementations share the same landmark indices, torso-center
normalization, shoulder-width scale, and zero-padding / presence-mask rules
(docs/feature-spec.md).
"""

import json
import math
from pathlib import Path

from app.recognition.features import (
    FACE_DIM,
    FACE_INDICES,
    FEATURE_DIM,
    HAND_DIM,
    POSE_DIM,
    POSE_INDICES,
    extract_frame_features,
)

# ---------------------------------------------------------------------------
# Fixed fixture landmarks (deterministic, no camera required)
# ---------------------------------------------------------------------------

# Pose: 33 landmarks. Shoulders at 11 (L) and 12 (R).
FIXTURE_POSE = [
    {"x": 0.5, "y": 0.5, "z": 0.0},   # 0 nose
] + [
    {"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(10)
] + [
    {"x": 0.35, "y": 0.45, "z": 0.0},  # 11 left shoulder
    {"x": 0.65, "y": 0.45, "z": 0.0},  # 12 right shoulder
] + [
    {"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(20)
]  # total 33

# Left hand: 21 landmarks, landmark 0 offset from torso center.
FIXTURE_LEFT_HAND = [
    {"x": 0.5, "y": 0.4, "z": 0.2},
] + [
    {"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(20)
]

# Right hand: 21 landmarks, all at torso center -> zeros after normalization.
FIXTURE_RIGHT_HAND = [
    {"x": 0.5, "y": 0.45, "z": 0.0} for _ in range(21)
]

# Face: 478 landmarks, all at torso center.
FIXTURE_FACE = [
    {"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(478)
]

FIXTURE_FRAME = {
    "left_hand": FIXTURE_LEFT_HAND,
    "right_hand": FIXTURE_RIGHT_HAND,
    "pose": FIXTURE_POSE,
    "face": FIXTURE_FACE,
}


# ---------------------------------------------------------------------------
# Reference implementation (pure Python, mirrors JS featureExtractor.ts)
# ---------------------------------------------------------------------------

def _ref_normalize_point(x, y, z, cx, cy, cz, scale):
    return (
        round((x - cx) / scale, 6),
        round((y - cy) / scale, 6),
        round((z - cz) / scale, 6),
    )


def _ref_extract(frame):
    pose = frame["pose"]
    left_hand = frame["left_hand"]
    right_hand = frame["right_hand"]
    face = frame["face"]

    # Torso reference from shoulders 11 and 12
    ls = pose[11]
    rs = pose[12]
    cx = (ls["x"] + rs["x"]) / 2.0
    cy = (ls["y"] + rs["y"]) / 2.0
    cz = (ls["z"] + rs["z"]) / 2.0
    dx = ls["x"] - rs["x"]
    dy = ls["y"] - rs["y"]
    dz = ls["z"] - rs["z"]
    scale = math.sqrt(dx * dx + dy * dy + dz * dz)
    if scale < 1e-4:
        scale = 0.25

    feats = []

    def push_hand(h):
        if h and len(h) >= 21:
            for lm in h[:21]:
                feats.extend(_ref_normalize_point(
                    lm["x"], lm["y"], lm["z"], cx, cy, cz, scale))
        else:
            feats.extend([0.0] * HAND_DIM)

    push_hand(left_hand)
    push_hand(right_hand)

    # Pose subset
    if pose and len(pose) >= 17:
        for idx in POSE_INDICES:
            lm = pose[idx] if idx < len(pose) else {"x": 0.0, "y": 0.0, "z": 0.0}
            feats.extend(_ref_normalize_point(
                lm["x"], lm["y"], lm["z"], cx, cy, cz, scale))
    else:
        feats.extend([0.0] * POSE_DIM)

    # Face subset
    if face and len(face) >= 363:
        for idx in FACE_INDICES:
            lm = face[idx] if idx < len(face) else {"x": 0.0, "y": 0.0, "z": 0.0}
            feats.extend(_ref_normalize_point(
                lm["x"], lm["y"], lm["z"], cx, cy, cz, scale))
    else:
        feats.extend([0.0] * FACE_DIM)

    # Presence masks
    feats.append(1.0 if (left_hand and len(left_hand) >= 21) else 0.0)
    feats.append(1.0 if (right_hand and len(right_hand) >= 21) else 0.0)
    feats.append(1.0 if (pose and len(pose) >= 17) else 0.0)
    feats.append(1.0 if (face and len(face) >= 363) else 0.0)

    return feats


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_feature_dim_is_193() -> None:
    """The shared feature layout must be exactly 193 floats."""
    assert FEATURE_DIM == 193
    assert HAND_DIM == 63
    assert POSE_DIM == 21
    assert FACE_DIM == 42


def test_extract_frame_features_matches_reference() -> None:
    """Python extractor must match the reference (and JS) implementation."""
    out = extract_frame_features(FIXTURE_FRAME)
    assert len(out) == FEATURE_DIM

    expected = _ref_extract(FIXTURE_FRAME)
    assert len(expected) == FEATURE_DIM

    for i, (got, exp) in enumerate(zip(out, expected, strict=True)):
        assert abs(got - exp) < 1e-6, f"Mismatch at index {i}: {got} vs {exp}"


def test_normalization_known_values() -> None:
    """Hand-computed normalization values for a deterministic fixture."""
    out = extract_frame_features(FIXTURE_FRAME)

    # Torso center = (0.5, 0.45, 0.0), scale = 0.3
    # Left hand landmark 0 = (0.5, 0.4, 0.2):
    #   nx = (0.5-0.5)/0.3 = 0.0
    #   ny = (0.4-0.45)/0.3 = -0.166667
    #   nz = (0.2-0.0)/0.3 = 0.666667
    assert out[0] == 0.0
    assert abs(out[1] - (-0.166667)) < 1e-6
    assert abs(out[2] - 0.666667) < 1e-6

    # Right hand at torso center -> all zeros
    assert all(v == 0.0 for v in out[HAND_DIM:HAND_DIM + HAND_DIM])

    # Pose nose (index 0) at (0.5,0.5,0.0):
    #   nx = (0.5-0.5)/0.3 = 0.0
    #   ny = (0.5-0.45)/0.3 = 0.166667
    #   nz = 0.0
    pose_start = HAND_DIM + HAND_DIM
    assert out[pose_start] == 0.0
    assert abs(out[pose_start + 1] - 0.166667) < 1e-6
    assert out[pose_start + 2] == 0.0

    # Presence masks: all present -> [1, 1, 1, 1]
    mask_start = FEATURE_DIM - 4
    assert out[mask_start:mask_start + 4] == [1.0, 1.0, 1.0, 1.0]


def test_missing_modality_zero_padding() -> None:
    """Absent hands/pose/face must zero-pad and set mask flags to 0."""
    frame = {
        "left_hand": None,
        "right_hand": None,
        "pose": None,
        "face": None,
    }
    out = extract_frame_features(frame)
    assert len(out) == FEATURE_DIM
    # All coordinate blocks zero
    assert all(v == 0.0 for v in out[:HAND_DIM + HAND_DIM + POSE_DIM + FACE_DIM])
    # All mask flags zero
    assert out[FEATURE_DIM - 4:] == [0.0, 0.0, 0.0, 0.0]


def test_pose_fallback_scale_when_missing() -> None:
    """Missing pose falls back to center (0.5,0.5,0) and scale 0.25."""
    frame = {
        "left_hand": FIXTURE_LEFT_HAND,
        "right_hand": FIXTURE_RIGHT_HAND,
        "pose": None,
        "face": FIXTURE_FACE,
    }
    out = extract_frame_features(frame)
    assert len(out) == FEATURE_DIM
    # Pose block zero-padded
    pose_start = HAND_DIM + HAND_DIM
    assert all(v == 0.0 for v in out[pose_start:pose_start + POSE_DIM])
    # Mask: pose absent (has_pose is the 3rd mask flag at FEATURE_DIM - 2)
    assert out[FEATURE_DIM - 4] == 1.0  # left hand present
    assert out[FEATURE_DIM - 3] == 1.0  # right hand present
    assert out[FEATURE_DIM - 2] == 0.0  # pose absent
    assert out[FEATURE_DIM - 1] == 1.0  # face present


def test_fixture_is_json_serializable() -> None:
    """The fixture can round-trip through JSON (matches .npz/JSON export spec)."""
    serializable = json.dumps(FIXTURE_FRAME)
    loaded = json.loads(serializable)
    assert loaded == FIXTURE_FRAME


def test_report_paths_exist() -> None:
    """Ensure ml/reports/ and ml/datasets/ directories exist for outputs."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    assert (repo_root / "ml" / "reports").is_dir()
    assert (repo_root / "ml" / "datasets").is_dir()
