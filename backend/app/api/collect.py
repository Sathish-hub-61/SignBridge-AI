"""collect.py — /collect API router for ISL training data recording.

Receives per-frame landmark payloads from the browser and appends them
to a JSONL file under data/raw/<sign_label>.jsonl.

Privacy: only numeric landmark coordinates arrive here; raw video never leaves the browser.
"""

import json
import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from app.recognition.features import extract_frame_features

logger = logging.getLogger("signbridge.collect")

router = APIRouter(prefix="/collect", tags=["collect"])

# Data directory — created on first write
DATA_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class Landmark(BaseModel):
    """A single MediaPipe 3-D landmark."""

    x: float
    y: float
    z: float


class CollectFrameRequest(BaseModel):
    """Landmark data for a single captured frame."""

    sign_label: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    session_id: str = Field(..., min_length=1, max_length=64)
    left_hand: list[Landmark] | None = None
    right_hand: list[Landmark] | None = None
    pose: list[Landmark] | None = None
    face: list[Landmark] | None = None


class CollectSampleRequest(BaseModel):
    """An already-averaged 193-dim feature vector for a single sign sample."""

    sign_label: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    session_id: str = Field(..., min_length=1, max_length=64)
    features: list[float] = Field(..., min_length=193, max_length=193)


class CollectResponse(BaseModel):
    ok: bool
    label: str
    total_samples: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_path(label: str) -> Path:
    """Return the JSONL path for a given sign label, ensuring the directory exists."""
    safe = label.replace(" ", "_").lower()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{safe}.jsonl"


def _count_samples(path: Path) -> int:
    """Count lines (samples) in a JSONL file."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/sample", response_model=CollectResponse)
async def save_averaged_sample(
    body: Annotated[CollectSampleRequest, Body()],
) -> CollectResponse:
    """Save a pre-averaged 193-dim feature vector as one training sample.

    Called by the browser after averaging a 30-frame capture window.
    """
    if len(body.features) != 193:
        raise HTTPException(status_code=422, detail=f"Expected 193 features, got {len(body.features)}")

    path = _label_path(body.sign_label)

    record = {
        "label": body.sign_label,
        "session_id": body.session_id,
        "timestamp": time.time(),
        "features": body.features,
    }

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    total = _count_samples(path)
    logger.info("Saved sample for label=%s total=%d session=%s", body.sign_label, total, body.session_id)

    return CollectResponse(ok=True, label=body.sign_label, total_samples=total)


@router.post("/frame", response_model=CollectResponse)
async def save_raw_frame(
    body: Annotated[CollectFrameRequest, Body()],
) -> CollectResponse:
    """Extract features from a single raw frame and save to JSONL.

    Accepts full landmark dictionaries; extracts the 193-dim feature vector
    server-side and saves it. The browser can also send pre-averaged features
    via /collect/sample (preferred path).
    """
    frame_data = {
        "left_hand": [lm.model_dump() for lm in body.left_hand] if body.left_hand else None,
        "right_hand": [lm.model_dump() for lm in body.right_hand] if body.right_hand else None,
        "pose": [lm.model_dump() for lm in body.pose] if body.pose else None,
        "face": [lm.model_dump() for lm in body.face] if body.face else None,
    }

    try:
        features = extract_frame_features(frame_data)
    except Exception as exc:
        logger.error("Feature extraction error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    path = _label_path(body.sign_label)
    record = {
        "label": body.sign_label,
        "session_id": body.session_id,
        "timestamp": time.time(),
        "features": features,
    }

    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    total = _count_samples(path)
    return CollectResponse(ok=True, label=body.sign_label, total_samples=total)


@router.get("/stats")
async def collect_stats() -> dict:
    """Return sample counts per sign label."""
    if not DATA_DIR.exists():
        return {"labels": {}, "total_samples": 0}

    stats: dict[str, int] = {}
    for p in sorted(DATA_DIR.glob("*.jsonl")):
        stats[p.stem] = _count_samples(p)

    return {"labels": stats, "total_samples": sum(stats.values())}
