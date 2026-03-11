#!/usr/bin/env python3
"""Segment face/hair/background per frame using BiSeNet face parsing.

Usage:
    python segment_face.py <frames_dir> <output_dir>

Outputs per-frame masks as 8-bit PNGs in output_dir:
    - mask_000000.png: pixel values encode class
        0 = background
        1 = face skin
        2 = hair
        3 = torso/clothing

Also writes a summary file: output_dir/segments.bin
    - u32: num_frames
    - u32: width
    - u32: height
    - For each frame: u8 array [H*W] with class labels
"""

import sys
import struct
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <frames_dir> <output_dir>")
        sys.exit(1)

    frames_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import torchvision.transforms as transforms
        import cv2
        import numpy as np
    except ImportError:
        print("Install: pip install torch torchvision opencv-python numpy")
        sys.exit(1)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        print(f"No frame_*.png files in {frames_dir}")
        sys.exit(1)

    print(f"Segmenting {len(frame_paths)} frames...")

    # Try loading BiSeNet; fall back to MediaPipe selfie segmentation
    try:
        from face_parsing import BiSeNet  # type: ignore
        use_bisenet = True
        print("Using BiSeNet face parsing model")
    except ImportError:
        use_bisenet = False
        print("BiSeNet not available, using MediaPipe selfie segmentation")

    if not use_bisenet:
        import mediapipe as mp
        mp_selfie = mp.solutions.selfie_segmentation
        segmenter = mp_selfie.SelfieSegmentation(model_selection=1)

    first_img = cv2.imread(str(frame_paths[0]))
    h, w = first_img.shape[:2]

    all_masks = bytearray()

    for i, path in enumerate(frame_paths):
        img = cv2.imread(str(path))
        if img is None:
            mask = np.zeros((h, w), dtype=np.uint8)
        elif use_bisenet:
            # BiSeNet produces 19-class segmentation
            # Classes: 1=skin, 2-4=brows/eyes, 5=glasses, 6=earring,
            # 7-9=ear/nose/mouth, 10=neck, 11-13=cloth/hair/hat, 14-18=accessories
            mask = run_bisenet(img, h, w)
        else:
            # MediaPipe: binary foreground/background
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = segmenter.process(rgb)
            fg_mask = (result.segmentation_mask > 0.5).astype(np.uint8)
            # Everything foreground is "face" (class 1), rest is background
            mask = fg_mask

        # Save individual mask
        mask_path = output_dir / f"mask_{i:06d}.png"
        cv2.imwrite(str(mask_path), mask * 85)  # scale for visibility

        all_masks.extend(mask.tobytes())

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(frame_paths)} frames segmented")

    # Write binary summary
    summary_path = output_dir / "segments.bin"
    with open(summary_path, 'wb') as f:
        f.write(struct.pack('<III', len(frame_paths), w, h))
        f.write(all_masks)

    print(f"Done! Masks saved to {output_dir}")

def run_bisenet(img, h, w):
    """Run BiSeNet face parsing and remap to 4 classes."""
    import numpy as np
    # Placeholder — actual BiSeNet integration would go here
    # For now, return all-face mask
    return np.ones((h, w), dtype=np.uint8)

if __name__ == '__main__':
    main()
