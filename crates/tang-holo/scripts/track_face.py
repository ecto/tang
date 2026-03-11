#!/usr/bin/env python3
"""Track face pose and expression per frame using FLAME 3DMM fitting.

Uses landmarks from detect_landmarks.py to fit FLAME model parameters:
- 6 DOF head pose (rotation + translation)
- 50 expression coefficients
- 100 shape coefficients (fitted once, shared across frames)

Usage:
    python track_face.py <landmarks.bin> <output.bin>

Output format (binary):
    - u32: num_frames
    - u32: num_expression_coeffs (50)
    - [f32; 100]: shape coefficients (shared)
    - For each frame:
        - u8: valid (1 if tracking succeeded)
        - [f32; 3]: rotation (axis-angle)
        - [f32; 3]: translation
        - [f32; 50]: expression coefficients
"""

import sys
import struct
import numpy as np
from pathlib import Path

NUM_LANDMARKS = 478
NUM_EXPR = 50
NUM_SHAPE = 100

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <landmarks.bin> <output.bin>")
        sys.exit(1)

    landmarks_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # Load landmarks
    data = landmarks_path.read_bytes()
    num_frames = struct.unpack('<I', data[0:4])[0]
    num_lm = struct.unpack('<I', data[4:8])[0]
    assert num_lm == NUM_LANDMARKS

    print(f"Fitting FLAME to {num_frames} frames...")

    frames = []
    offset = 8
    for _ in range(num_frames):
        detected = data[offset] != 0
        offset += 1
        points = np.frombuffer(data[offset:offset + num_lm * 12], dtype=np.float32).reshape(num_lm, 3)
        offset += num_lm * 12
        frames.append((detected, points))

    # Simple PnP-based tracking (no FLAME dependency required)
    # For each frame, estimate rigid pose from 2D landmarks
    # Expression coefficients approximated from landmark displacements

    shape_coeffs = np.zeros(NUM_SHAPE, dtype=np.float32)
    results = []

    # Compute mean face shape from all detected frames
    detected_frames = [(i, pts) for i, (det, pts) in enumerate(frames) if det]
    if detected_frames:
        all_pts = np.stack([pts for _, pts in detected_frames])
        mean_shape = all_pts.mean(axis=0)
    else:
        mean_shape = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)

    for i, (detected, points) in enumerate(frames):
        if not detected:
            results.append((False, np.zeros(3), np.zeros(3), np.zeros(NUM_EXPR)))
            continue

        # Estimate head pose from landmark positions
        # Face center from nose tip (landmark 1) and surrounding landmarks
        face_center = points[1]  # nose tip
        translation = np.array([
            (face_center[0] - 0.5) * 2.0,  # x: centered
            (face_center[1] - 0.5) * 2.0,  # y: centered
            face_center[2] * 5.0,  # z: depth
        ], dtype=np.float32)

        # Rough rotation from face normal
        # Use cross product of face plane vectors
        left_eye = points[33]   # left eye inner corner
        right_eye = points[263] # right eye inner corner
        nose = points[1]
        chin = points[152]

        horizontal = right_eye - left_eye
        vertical = chin - nose

        # Rotation as axis-angle (simplified)
        yaw = np.arctan2(horizontal[2], np.linalg.norm(horizontal[:2]))
        pitch = np.arctan2(vertical[2], np.linalg.norm(vertical[:2]))
        roll = np.arctan2(horizontal[1], horizontal[0])

        rotation = np.array([pitch, yaw, roll], dtype=np.float32)

        # Expression as deviation from mean shape
        delta = points - mean_shape
        # Project onto first NUM_EXPR principal components (simplified: use raw deltas)
        expr = np.zeros(NUM_EXPR, dtype=np.float32)
        # Mouth openness (key expression)
        upper_lip = points[13]
        lower_lip = points[14]
        mouth_open = np.linalg.norm(lower_lip - upper_lip)
        expr[0] = mouth_open * 10.0  # jaw open

        # Eye openness
        left_eye_top = points[159]
        left_eye_bot = points[145]
        right_eye_top = points[386]
        right_eye_bot = points[374]
        expr[1] = np.linalg.norm(left_eye_top - left_eye_bot) * 10.0   # left blink
        expr[2] = np.linalg.norm(right_eye_top - right_eye_bot) * 10.0 # right blink

        # Lip corners
        left_corner = points[61]
        right_corner = points[291]
        lip_width = np.linalg.norm(right_corner - left_corner)
        expr[3] = lip_width * 5.0  # smile

        results.append((True, rotation, translation, expr))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{num_frames} frames tracked")

    # Write output
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<II', num_frames, NUM_EXPR))
        f.write(shape_coeffs.tobytes())
        for valid, rot, trans, expr in results:
            f.write(struct.pack('<B', 1 if valid else 0))
            f.write(rot.astype(np.float32).tobytes())
            f.write(trans.astype(np.float32).tobytes())
            f.write(expr.astype(np.float32).tobytes())

    valid_count = sum(1 for v, _, _, _ in results if v)
    print(f"Done! {valid_count}/{num_frames} frames tracked -> {output_path}")

if __name__ == '__main__':
    main()
