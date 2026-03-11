#!/usr/bin/env python3
"""Detect 478 face landmarks per frame using MediaPipe Face Mesh.

Usage:
    python detect_landmarks.py <frames_dir> <output.bin>

Output format (binary):
    - u32: num_frames
    - u32: num_landmarks (478)
    - For each frame:
        - u8: detected (1 if face found, 0 if not)
        - [f32; 478*3]: (x, y, z) normalized landmarks (if detected)
"""

import sys
import struct
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <frames_dir> <output.bin>")
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    try:
        import mediapipe as mp
        import cv2
    except ImportError:
        print("Install dependencies: pip install mediapipe opencv-python")
        sys.exit(1)

    # Collect frame paths
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        print(f"No frame_*.png files found in {frames_dir}")
        sys.exit(1)

    print(f"Processing {len(frame_paths)} frames...")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,  # 478 landmarks (includes iris)
        min_detection_confidence=0.5,
    )

    num_landmarks = 478
    results_data = bytearray()
    detected_count = 0

    for i, path in enumerate(frame_paths):
        img = cv2.imread(str(path))
        if img is None:
            results_data.append(0)
            results_data.extend(b'\x00' * (num_landmarks * 3 * 4))
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks and len(result.multi_face_landmarks) > 0:
            landmarks = result.multi_face_landmarks[0]
            results_data.append(1)
            detected_count += 1
            for lm in landmarks.landmark:
                results_data.extend(struct.pack('<fff', lm.x, lm.y, lm.z))
        else:
            results_data.append(0)
            results_data.extend(b'\x00' * (num_landmarks * 3 * 4))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(frame_paths)} frames processed ({detected_count} faces found)")

    face_mesh.close()

    # Write output
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<II', len(frame_paths), num_landmarks))
        f.write(results_data)

    print(f"Done! {detected_count}/{len(frame_paths)} frames with faces -> {output_path}")

if __name__ == '__main__':
    main()
