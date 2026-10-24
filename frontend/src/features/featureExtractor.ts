/**
 * featureExtractor.ts
 *
 * Browser-side 193-dimensional feature vector extraction.
 * Mirrors backend/app/recognition/features.py exactly — same normalization,
 * same landmark indices, same zero-padding for absent modalities.
 *
 * Privacy invariant: only numeric arrays ever leave this module.
 * Raw video frames are never serialized or transmitted.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Landmark3D {
  x: number;
  y: number;
  z: number;
}

export interface RawFrameData {
  leftHand: Landmark3D[] | null;   // 21 landmarks or null
  rightHand: Landmark3D[] | null;  // 21 landmarks or null
  pose: Landmark3D[] | null;       // 33 landmarks or null
  face: Landmark3D[] | null;       // 478 landmarks or null
}

// ---------------------------------------------------------------------------
// Constants — must match features.py
// ---------------------------------------------------------------------------

/** MediaPipe Pose keypoint indices (7 upper-body landmarks) */
const POSE_INDICES = [0, 11, 12, 13, 14, 15, 16] as const;

/** MediaPipe FaceMesh keypoint indices (14 landmarks) */
const FACE_INDICES = [70, 107, 300, 336, 33, 133, 263, 362, 0, 13, 14, 17, 61, 291] as const;

const HAND_DIM  = 21 * 3;                    // 63
const POSE_DIM  = POSE_INDICES.length * 3;   // 21
const FACE_DIM  = FACE_INDICES.length * 3;   // 42
const MASKS_DIM = 4;                          // 4

export const FEATURE_DIM = HAND_DIM + HAND_DIM + POSE_DIM + FACE_DIM + MASKS_DIM; // 193

// ---------------------------------------------------------------------------
// Normalization helpers
// ---------------------------------------------------------------------------

interface TorsoRef {
  cx: number;
  cy: number;
  cz: number;
  scale: number;
}

function calcTorsoRef(pose: Landmark3D[] | null): TorsoRef {
  if (!pose || pose.length < 13) {
    return { cx: 0.5, cy: 0.5, cz: 0.0, scale: 0.25 };
  }
  const ls = pose[11];
  const rs = pose[12];
  const cx = (ls.x + rs.x) / 2;
  const cy = (ls.y + rs.y) / 2;
  const cz = (ls.z + rs.z) / 2;
  const dx = ls.x - rs.x;
  const dy = ls.y - rs.y;
  const dz = ls.z - rs.z;
  let scale = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (scale < 1e-4) scale = 0.25;
  return { cx, cy, cz, scale };
}

function normPoint(lm: Landmark3D, ref: TorsoRef): [number, number, number] {
  return [
    Math.round(((lm.x - ref.cx) / ref.scale) * 1e6) / 1e6,
    Math.round(((lm.y - ref.cy) / ref.scale) * 1e6) / 1e6,
    Math.round(((lm.z - ref.cz) / ref.scale) * 1e6) / 1e6,
  ];
}

// ---------------------------------------------------------------------------
// Main extractor
// ---------------------------------------------------------------------------

export function extractFrameFeatures(frame: RawFrameData): Float32Array {
  const { leftHand, rightHand, pose, face } = frame;
  const ref = calcTorsoRef(pose);
  const out: number[] = [];

  // 1. Left hand (63)
  if (leftHand && leftHand.length >= 21) {
    for (let i = 0; i < 21; i++) out.push(...normPoint(leftHand[i], ref));
  } else {
    for (let i = 0; i < HAND_DIM; i++) out.push(0.0);
  }

  // 2. Right hand (63)
  if (rightHand && rightHand.length >= 21) {
    for (let i = 0; i < 21; i++) out.push(...normPoint(rightHand[i], ref));
  } else {
    for (let i = 0; i < HAND_DIM; i++) out.push(0.0);
  }

  // 3. Pose subset (21)
  if (pose && pose.length >= 17) {
    for (const idx of POSE_INDICES) {
      const lm = idx < pose.length ? pose[idx] : { x: 0, y: 0, z: 0 };
      out.push(...normPoint(lm, ref));
    }
  } else {
    for (let i = 0; i < POSE_DIM; i++) out.push(0.0);
  }

  // 4. Face subset (42)
  if (face && face.length >= 363) {
    for (const idx of FACE_INDICES) {
      const lm = idx < face.length ? face[idx] : { x: 0, y: 0, z: 0 };
      out.push(...normPoint(lm, ref));
    }
  } else {
    for (let i = 0; i < FACE_DIM; i++) out.push(0.0);
  }

  // 5. Presence mask (4)
  out.push(
    leftHand  && leftHand.length  >= 21  ? 1.0 : 0.0,
    rightHand && rightHand.length >= 21  ? 1.0 : 0.0,
    pose      && pose.length      >= 17  ? 1.0 : 0.0,
    face      && face.length      >= 363 ? 1.0 : 0.0,
  );

  if (out.length !== FEATURE_DIM) {
    throw new Error(`Feature dim mismatch: expected ${FEATURE_DIM}, got ${out.length}`);
  }
  return new Float32Array(out);
}

export function averageFeatureSequence(frames: Float32Array[]): Float32Array {
  if (frames.length === 0) throw new Error('Cannot average empty sequence');
  const avg = new Float32Array(FEATURE_DIM);
  for (const f of frames) {
    for (let i = 0; i < FEATURE_DIM; i++) avg[i] += f[i];
  }
  for (let i = 0; i < FEATURE_DIM; i++) avg[i] /= frames.length;
  const maskStart = FEATURE_DIM - MASKS_DIM;
  for (const f of frames) {
    for (let i = maskStart; i < FEATURE_DIM; i++) {
      if (f[i] > avg[i]) avg[i] = f[i];
    }
  }
  return avg;
}
