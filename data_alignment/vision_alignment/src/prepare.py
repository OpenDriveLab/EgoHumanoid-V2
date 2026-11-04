"""Prepare image-space arm alignment while preserving the input joint trajectory.

The shared virtual camera supplies an approximate joint-driven projection.
RGB cuff detections align that projection; arm and glove masks then refine
its image-space anchors. The output stores joints and 2D alignment separately.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .axes import fit_frame_axes, smooth_track
from .geometry import (
    BODY_JOINT_NAMES,
    CameraCalibration,
    align_projected_arms,
    project_g1_arm_landmarks,
)
from .tracking import detect_cuff_trajectory, smoothing_window_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("video", "joints", "masks", "calibration", "robot-model", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError("Cannot open input video")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = (int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    with np.load(args.joints, allow_pickle=False) as data:
        q = data["body_q"].astype(np.float64)
        if q.shape != (n, 29) or not np.isfinite(q).all():
            raise ValueError("body_q must contain 29 finite joint angles per RGB frame")
        if tuple(data["joint_names"].tolist()) != tuple(BODY_JOINT_NAMES):
            raise ValueError("joint_names do not match the G1 MuJoCo joint order")
        if abs(float(data["fps"]) - fps) > 1e-6:
            raise ValueError("Joint and video frame rates differ")
    camera = json.loads(args.calibration.read_text())["camera"]
    calibration = CameraCalibration(
        np.array(camera["position_parent_m"]),
        np.array(camera["rotation_camera_axes_in_parent"]),
        camera["vertical_fov_degrees"],
        camera["width"],
        camera["height"],
    )
    if (calibration.width, calibration.height) != (w, h):
        raise ValueError("Calibration and video dimensions differ")
    # RGB wrist-cuff detections provide targets for the initial projected arms.
    _, _, _, smooth, _ = detect_cuff_trajectory(
        args.video,
        frame_count=n,
        fps=fps,
        stride=5,
        smoothing_window=smoothing_window_frames(fps),
    )
    projection = project_g1_arm_landmarks(args.robot_model, q, calibration)
    aligned = align_projected_arms(projection.pixels, smooth, width=w, height=h)
    # Match the stored geometry precision before fitting image-space boundaries.
    prior_cuffs = smooth.astype(np.float32).astype(np.float64)
    prior_roots = aligned.root_points.astype(np.float32).astype(np.float64)
    # Refine wrist and arm-entry anchors from the removal and glove masks.
    cuffs, roots = [], []
    for i in range(n):
        masks = [
            cv2.imread(str(args.masks / name / f"{i:06d}.png"), 0) for name in ("removal", "gloves")
        ]
        if any(m is None or m.shape != (h, w) for m in masks):
            raise ValueError(f"Missing or incorrectly sized mask at frame {i}")
        cuff, root = fit_frame_axes(masks[0] > 127, masks[1] > 127, prior_cuffs[i], prior_roots[i])
        cuffs.append(cuff)
        roots.append(root)
    # Smooth image-space anchors; extend the wrist overlap beneath the retained glove.
    cuffs = smooth_track(np.asarray(cuffs), 8)
    roots = smooth_track(np.asarray(roots), 8)
    direction = roots - cuffs
    cuffs -= 12 * direction / np.maximum(np.linalg.norm(direction, axis=2, keepdims=True), 1e-6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        body_q=q.astype(np.float32),
        smoothed_cuff_points=smooth.astype(np.float32),
        visual_cuff_points=cuffs,
        aligned_landmarks=aligned.landmarks.astype(np.float32),
        root_points=roots,
        similarity_transforms=aligned.transforms.astype(np.float32),
    )


if __name__ == "__main__":
    main()
