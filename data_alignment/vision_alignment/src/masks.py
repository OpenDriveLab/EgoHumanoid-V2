"""Convert SAM class masks into generation and final-compositing masks.

The generation mask includes gloves so the background can be reconstructed.
The removal mask excludes gloves; render.py restores their original RGB pixels.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def make_masks(arm, glove, tracker, radius, tracker_radius=0):
    if arm.shape != glove.shape or arm.shape != tracker.shape or arm.ndim != 2:
        raise ValueError("Mask shapes must match")
    if radius < 0 or tracker_radius < 0:
        raise ValueError("Mask dilation must be non-negative")
    union = np.asarray(arm | glove | tracker, dtype=np.uint8)
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        union = cv2.dilate(union, kernel)
    tracker_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tracker_radius + 1,) * 2)
    glove_border = (
        cv2.dilate(glove.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    )
    near_glove = cv2.dilate(glove.astype(np.uint8), tracker_kernel) > 0
    cuff = arm & ~glove_border & near_glove
    tracker_margin = cv2.dilate((tracker | cuff).astype(np.uint8), tracker_kernel)
    generator = (union > 0) | (tracker_margin > 0)
    return generator, generator & ~glove


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dilation", type=int, default=8)
    parser.add_argument("--tracker-dilation", type=int, default=48)
    args = parser.parse_args()
    report = json.loads((args.masks / "report.json").read_text())
    h, w, n = report["height"], report["width"], report["frames"]
    if args.output.exists():
        raise FileExistsError(args.output)
    for name in ("removal", "gloves"):
        (args.output / name).mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(args.output / "generator.avi"),
        cv2.VideoWriter_fourcc(*"FFV1"),
        report["fps"],
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError("Cannot create lossless mask video")
    try:
        for i in range(n):
            masks = []
            for name in ("arm", "glove", "tracker"):
                mask = cv2.imread(str(args.masks / name / f"{i:06d}.png"), 0)
                if mask is None or mask.shape != (h, w):
                    raise ValueError(f"Missing or invalid {name} mask at frame {i}")
                masks.append(mask > 127)
            generator, removal = make_masks(*masks, args.dilation, args.tracker_dilation)
            writer.write(cv2.cvtColor(generator.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR))
            for name, mask in [("removal", removal), ("gloves", masks[1])]:
                if not cv2.imwrite(
                    str(args.output / name / f"{i:06d}.png"),
                    mask.astype(np.uint8) * 255,
                ):
                    raise OSError(f"Cannot save {name} at frame {i}")
    finally:
        writer.release()
    report["dilation_pixels"] = args.dilation
    report["tracker_dilation_pixels"] = args.tracker_dilation
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
