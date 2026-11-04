"""Run the official DiffuEraser pipeline with a fixed seed and full frame coverage."""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--max-size", type=int, default=960)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-dilation", type=int, default=1)
    args = parser.parse_args()
    for name in ("video", "mask", "output", "model_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.model_root))
    # Upstream resolves its PCM adapter relative to the repository directory.
    os.chdir(args.model_root)
    import cv2
    import numpy as np
    import torch
    from diffueraser.diffueraser import DiffuEraser
    from propainter.inference import Propainter

    def spec(path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(path)
        result = {
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()
        return result

    source_spec = spec(args.video)
    if spec(args.mask) != source_spec:
        raise ValueError("Mask and source must have identical frame count, FPS and dimensions")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True)
    start = time.monotonic()
    weights = args.model_root / "weights"
    model = DiffuEraser(
        "cuda",
        str(weights / "stable-diffusion-v1-5"),
        str(weights / "sd-vae-ft-mse"),
        str(weights / "diffuEraser"),
        ckpt="2-Step",
    )
    prior_model = Propainter(str(weights / "propainter"), device=torch.device("cuda"))
    duration = (source_spec["frames"] + 0.1) / source_spec["fps"]
    prior = args.output / "prior.mp4"
    prior_model.forward(
        str(args.video),
        str(args.mask),
        str(prior),
        video_length=duration,
        ref_stride=10,
        neighbor_length=10,
        subvideo_length=50,
        mask_dilation=args.mask_dilation,
    )
    shutil.copyfile(prior, args.output / "prior_saved.mp4")
    model.forward(
        str(args.video),
        str(args.mask),
        str(prior),
        str(args.output / "inpaint.mp4"),
        max_img_size=args.max_size,
        video_length=duration,
        mask_dilation_iter=args.mask_dilation,
        seed=args.seed,
    )
    result = spec(args.output / "inpaint.mp4")
    if result["frames"] != source_spec["frames"] or abs(result["fps"] - source_spec["fps"]) > 1e-6:
        raise RuntimeError(f"Output timeline mismatch: {result} versus {source_spec}")
    report = {
        "model": "DiffuEraser",
        "prior_model": "ProPainter",
        "sampler": "PCM 2-Step",
        "seed": args.seed,
        "max_size": args.max_size,
        "mask_dilation": args.mask_dilation,
        "source": source_spec,
        "output": result,
        "input_sha256": digest(args.video),
        "mask_sha256": digest(args.mask),
        "output_sha256": digest(args.output / "inpaint.mp4"),
        "seconds": time.monotonic() - start,
        "torch": torch.__version__,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
