# SignBridge AI Feature Vector Specification

## 1. Overview
To ensure perfect parity between client-side inference and offline training, all features are extracted from MediaPipe landmark coordinate streams and normalized using a deterministic mathematical pipeline.

## 2. Landmark Subsets & Coordinate Layout
Each frame is transformed into a 1D vector of **193 floating-point numbers** (Float32):

| Feature Block | MediaPipe Source | Count & Indices | Dimension |
|---|---|---|---|
| **Left Hand** | HandLandmarker (`left_hand`) | 21 landmarks $(x, y, z)$ | $21 \times 3 = 63$ |
| **Right Hand** | HandLandmarker (`right_hand`) | 21 landmarks $(x, y, z)$ | $21 \times 3 = 63$ |
| **Upper Pose** | PoseLandmarker (`pose`) | 7 key landmarks:<br>0 (Nose)<br>11 (L Shoulder), 12 (R Shoulder)<br>13 (L Elbow), 14 (R Elbow)<br>15 (L Wrist), 16 (R Wrist) | $7 \times 3 = 21$ |
| **Face Subset** | FaceLandmarker (`face`) | 14 non-manual marker landmarks:<br>70, 107 (L Eyebrow)<br>300, 336 (R Eyebrow)<br>33, 133 (L Eye)<br>263, 362 (R Eye)<br>0, 13, 14, 17 (Lips)<br>61, 291 (Mouth corners) | $14 \times 3 = 42$ |
| **Presence Masks**| Detection status flags | $[is\_lh, is\_rh, is\_pose, is\_face]$ | 4 |
| **Total** | | | **193 floats** |

---

## 3. Normalization Mathematics

Let $P_{ls} = (x_{ls}, y_{ls}, z_{ls})$ be the left shoulder and $P_{rs} = (x_{rs}, y_{rs}, z_{rs})$ be the right shoulder.

1. **Center of Reference (Torso Midpoint):**
   $$C = \frac{P_{ls} + P_{rs}}{2} = \left(\frac{x_{ls} + x_{rs}}{2}, \frac{y_{ls} + y_{rs}}{2}, \frac{z_{ls} + z_{rs}}{2}\right)$$
   *Fallback if pose is missing:* $C = (0.5, 0.5, 0.0)$.

2. **Torso Scale Factor (Shoulder Width):**
   $$S = \|P_{ls} - P_{rs}\|_2 = \sqrt{(x_{ls} - x_{rs})^2 + (y_{ls} - y_{rs})^2 + (z_{ls} - z_{rs})^2}$$
   *Fallback if $S < 10^{-4}$ or pose is missing:* $S = 0.25$.

3. **Coordinate Transformation:**
   For any detected landmark point $P = (x, y, z)$:
   $$\hat{P} = \frac{P - C}{S}$$

4. **Missing Hand / Face Zero-Padding:**
   - If a hand or face is not detected in the current frame, all its corresponding coordinate slots are set to $0.0$.
   - The presence mask flags are set to $1.0$ if present, $0.0$ if absent.
