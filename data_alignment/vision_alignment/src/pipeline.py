"""Replace human arms in RGB using synchronized G1 joint trajectories.

Stages: segment arms/gloves/devices; fit joint projections to image evidence;
inpaint removed regions; render and composite the image-aligned G1 arms.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .segment import digest

# Insertion order is the full-pipeline execution order.
STAGE_DEPENDENCIES = {
    "segment": (),
    "geometry": ("segment",),
    "inpaint": ("segment",),
    "render": ("geometry", "inpaint"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Input RGB video",
    )
    parser.add_argument("--motion", type=Path, required=True, help="Synchronized joint NPZ")
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for outputs and run state"
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        required=True,
        help="Directory created by setup/vision/install.sh",
    )
    parser.add_argument(
        "--calibration", type=Path, help="Virtual-camera JSON; defaults to the shared approximate projection camera"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-size", type=int, default=960, help="Maximum inpainting image dimension"
    )
    parser.add_argument(
        "--stage",
        choices=["all", *STAGE_DEPENDENCIES],
        default="all",
        help="Run all stages or one stage whose prerequisites are already complete",
    )
    args = parser.parse_args()
    args.video, args.motion, args.output, args.runtime = (
        x.resolve() for x in (args.video, args.motion, args.output, args.runtime)
    )
    package = Path(__file__).resolve().parent
    calibration = (args.calibration or package / "assets/camera.json").resolve()
    video, joints = args.video, args.motion
    checkpoint = args.runtime / "models/sam3.1_multiplex.pt"
    robot = args.runtime / "models/g1/g1_revo2.xml"
    for f in (video, joints, calibration, checkpoint, robot):
        if not f.is_file():
            raise FileNotFoundError(f)
    signature = {
        "video_sha256": digest(video),
        "joints_sha256": digest(joints),
        "camera_sha256": digest(calibration),
        "checkpoint_sha256": digest(checkpoint),
        "robot_xml_sha256": digest(robot),
        "seed": args.seed,
        "max_size": args.max_size,
        "code_sha256": hashlib.sha256(
            b"".join(f.read_bytes() for f in sorted(package.glob("*.py")))
        ).hexdigest(),
    }
    state_path = args.output / "run.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["configuration"] != signature:
            raise ValueError("Inputs or settings changed; use a new output directory")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Output directory is not an existing pipeline run")
        state = {"configuration": signature, "completed": []}
    if args.stage != "all" and args.stage not in state["completed"]:
        missing = [s for s in STAGE_DEPENDENCIES[args.stage] if s not in state["completed"]]
        if missing:
            raise ValueError(
                f"Stage {args.stage} requires completed stages: {', '.join(missing)}. "
                "Run --stage all, or complete those stages in the same output directory."
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    masks = args.output / "masks"
    geometry = args.output / "geometry.npz"
    common = ["--video", str(video)]
    commands = {
        "segment": [
            [
                sys.executable,
                "-m",
                "visual_align.segment",
                *common,
                "--checkpoint",
                str(checkpoint),
                "--seed",
                str(args.seed),
                "--output",
                str(args.output / "sam"),
            ],
            [
                sys.executable,
                "-m",
                "visual_align.masks",
                "--masks",
                str(args.output / "sam"),
                "--output",
                str(masks),
            ],
        ],
        "geometry": [
            [
                sys.executable,
                "-m",
                "visual_align.prepare",
                *common,
                "--joints",
                str(joints),
                "--masks",
                str(masks),
                "--calibration",
                str(calibration),
                "--robot-model",
                str(robot),
                "--output",
                str(geometry),
            ]
        ],
        "inpaint": [
            [
                str(args.runtime / ".venv-inpaint/bin/python"),
                str(package / "inpaint.py"),
                *common,
                "--mask",
                str(masks / "generator.avi"),
                "--output",
                str(args.output / "inpaint"),
                "--model-root",
                str(args.runtime / "diffueraser"),
                "--seed",
                str(args.seed),
                "--max-size",
                str(args.max_size),
            ]
        ],
        "render": [
            [
                sys.executable,
                "-m",
                "visual_align.render",
                *common,
                "--inpaint",
                str(args.output / "inpaint/inpaint.mp4"),
                "--geometry",
                str(geometry),
                "--masks",
                str(masks),
                "--calibration",
                str(calibration),
                "--robot-model",
                str(robot),
                "--output",
                str(args.output / "aligned.mp4"),
            ]
        ],
    }
    for stage in STAGE_DEPENDENCIES:
        stage_commands = commands[stage]
        if args.stage not in ("all", stage) or stage in state["completed"]:
            continue
        print(f"{stage}...", flush=True)
        with (args.output / "logs" / f"{stage}.log").open("w") as log:
            for command in stage_commands:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "MUJOCO_GL": "egl"},
                )
        state["completed"].append(stage)
        if stage == "render":
            state["output_sha256"] = digest(args.output / "aligned.mp4")
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
