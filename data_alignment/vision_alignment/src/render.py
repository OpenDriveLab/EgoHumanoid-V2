"""Render joint-driven G1 arms, apply image-space alignment, and composite RGB.

Joint angles determine the 3D mesh pose. Geometry prepared from RGB and masks
supplies per-arm 2D transforms. The composite retains the source gloves.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .composite import composite_locked_frame
from .geometry import (
    CameraCalibration,
    G1ArmAppearance,
    G1ArmMeshRenderer,
    G1CaptureSleeve,
    compose_g1_arm_render_transforms,
    compute_g1_arm_root_transforms,
)
from .image_geometry import clip_render_alpha_to_removal_corridor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "video",
        "inpaint",
        "geometry",
        "masks",
        "calibration",
        "robot-model",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-canvas", type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = cv2.VideoCapture(str(args.video))
    bg = cv2.VideoCapture(str(args.inpaint))
    if not raw.isOpened() or not bg.isOpened():
        raise ValueError("Cannot open source or inpaint video")
    n = int(raw.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(raw.get(cv2.CAP_PROP_FPS))
    w, h = (int(raw.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if int(bg.get(cv2.CAP_PROP_FRAME_COUNT)) != n or abs(bg.get(cv2.CAP_PROP_FPS) - fps) > 1e-6:
        raise ValueError("Source and inpaint timelines differ")
    camera = json.loads(args.calibration.read_text())["camera"]
    calibration = CameraCalibration(
        np.asarray(camera["position_parent_m"]),
        np.asarray(camera["rotation_camera_axes_in_parent"]),
        camera["vertical_fov_degrees"],
        camera["width"],
        camera["height"],
    )
    if (calibration.width, calibration.height) != (w, h):
        raise ValueError("Camera calibration resolution differs from the input video")
    with np.load(args.geometry, allow_pickle=False) as f:
        geometry = dict(f)
    if geometry["body_q"].shape != (n, 29):
        raise ValueError("Robot trajectory must contain one 29-DOF pose per video frame")
    cuffs = geometry.get("visual_cuff_points", geometry["smoothed_cuff_points"])
    # Combine the initial projection alignment with mask-refined arm anchors.
    transforms = np.stack(
        [
            compose_g1_arm_render_transforms(
                geometry["similarity_transforms"][i],
                compute_g1_arm_root_transforms(
                    geometry["aligned_landmarks"][i],
                    cuffs[i],
                    geometry["root_points"][i],
                ),
            )
            for i in range(n)
        ]
    )
    appearance = G1ArmAppearance(
        camera_rgb_gain=(2.42, 2.44, 2.37),
        highlight_knee=254.0,
        camera_gamma=1.6,
        headlight_ambient=0.2,
        headlight_diffuse=0.35,
        headlight_specular=0.75,
        silver_pbr_enabled=True,
        silver_pbr_specular=0.82,
        silver_pbr_shininess=0.31,
        silver_pbr_metallic=0.68,
        silver_pbr_roughness=0.27,
        capture_silver_rgb_scale=(0.95, 0.86, 0.77),
        capture_wrist_yaw_scale=0.42,
        capture_dark_rgb_scale=(1.34, 1.2537, 1.1674),
    )
    sleeve = G1CaptureSleeve(
        core_radius_m=0.031,
        rib_radius_m=0.033,
        start_fraction=0.08,
        end_fraction=0.25,
        rib_count=3,
        rib_half_width_fraction=0.025,
        rgb_scale=(1.6758, 1.5674, 1.4589),
        material_profile="dark",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    renderer = None
    samples = []
    sample_indices = set(np.linspace(0, n - 1, 8).round().astype(int))
    started = time.monotonic()
    try:
        renderer = G1ArmMeshRenderer(
            args.robot_model,
            calibration,
            transforms.reshape(-1, 2, 3),
            appearance=appearance,
            capture_sleeve=sleeve,
            maximum_canvas_dimension_px=args.max_canvas,
        )
        for i in range(n):
            ok1, source = raw.read()
            ok2, background = bg.read()
            if not ok1 or not ok2:
                raise RuntimeError(f"Video ended at frame {i}/{n}")
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            background = cv2.cvtColor(cv2.resize(background, (w, h)), cv2.COLOR_BGR2RGB)
            masks = []
            for name in ("removal", "gloves"):
                m = cv2.imread(str(args.masks / name / f"{i:06d}.png"), 0)
                if m is None or m.shape != (h, w):
                    raise ValueError(f"Invalid {name} mask at frame {i}")
                masks.append(m > 127)
            removal, gloves = masks
            # Render the input joint pose, then warp each arm layer in image space.
            layer = renderer.render(geometry["body_q"][i], transforms[i])
            alpha = clip_render_alpha_to_removal_corridor(
                layer.alpha, removal, expansion_px=14, feather_px=3
            )
            old_coverage = np.max(layer.alpha, axis=0)
            coverage = np.max(alpha, axis=0)
            ratio = np.divide(
                coverage,
                old_coverage,
                out=np.zeros_like(coverage),
                where=old_coverage > 0,
            )
            rgb = np.rint(layer.rgb * ratio[..., None]).clip(0, 255).astype(np.uint8)
            frame, _ = composite_locked_frame(source, background, removal, gloves, rgb, alpha)
            allowed = removal | (coverage > 0)
            if not np.array_equal(frame[gloves], source[gloves]):
                raise RuntimeError(f"Source glove changed at frame {i}")
            if not np.array_equal(frame[~allowed], source[~allowed]):
                raise RuntimeError(f"Background outside edit region changed at frame {i}")
            writer.stdin.write(frame.tobytes())
            if i in sample_indices:
                row = cv2.hconcat(
                    [
                        cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), (384, 216))
                        for x in (source, background, frame)
                    ]
                )
                cv2.putText(row, f"Frame {i}", (8, 20), 0, 0.5, (255, 255, 255), 1)
                samples.append(row)
            if i % 100 == 0:
                print(f"render {i + 1}/{n}", flush=True)
    finally:
        if renderer is not None:
            renderer.close()
        raw.release()
        bg.release()
        writer.stdin.close()
        status = writer.wait()
    if status:
        raise RuntimeError(f"Video encoder exited with status {status}")
    cv2.imwrite(str(args.output.with_suffix(".jpg")), cv2.vconcat(samples))
    report = {
        "frames": n,
        "fps": fps,
        "width": w,
        "height": h,
        "source_gloves_preserved_before_encoding": True,
        "background_outside_edit_region_preserved_before_encoding": True,
        "seconds": time.monotonic() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
