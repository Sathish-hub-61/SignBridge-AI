# SignBridge Data Collection Specification

## Overview
To guarantee training/serving parity, all primary sign training samples must be collected directly using the browser `/collect` interface with `@mediapipe/tasks-vision`.

## Output Format
Each recorded clip is serialized as a standardized dictionary stored in `.npz` format:
- `landmarks`: array of shape `(num_frames, 225)` (Float32)
- `label`: string ISL sign label (e.g. `HELP`, `PAIN`, `IDLE`)
- `signer_id`: string identifier (e.g. `signer_01`)
- `session_id`: timestamp/session uuid
- `frame_rate`: detected client capture FPS (e.g. 25-30)
- `metadata`: JSON string with lighting/camera tags

## Directory Structure
Collected data is saved to `ml/datasets/` (which is strictly gitignored).
