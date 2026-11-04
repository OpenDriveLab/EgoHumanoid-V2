"""Generate full-length arm, glove, and tracker masks with SAM 3.1."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def binary_union(outputs, height, width):
    masks = outputs["out_binary_masks"]
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks, dtype=bool)
    if masks.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    masks = masks.reshape(-1, height, width)
    return masks.any(axis=0).astype(np.uint8) * 255


def propagation_windows(frame_count, fps):
    """Refresh text detection every four seconds, with overlapping context."""
    stride = max(1, round(4 * fps))
    overlap = max(1, round(0.5 * fps))
    return [
        (max(0, start - overlap), min(frame_count - 1, start + stride - 1 + overlap))
        for start in range(0, frame_count, stride)
    ]


def window_resources(video, windows, directory, frame_count):
    """Use lossless frames in temporal order for both propagation directions."""
    frames = directory / "frames"
    frames.mkdir()
    cap = cv2.VideoCapture(str(video))
    try:
        for i in range(frame_count):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {i}")
            if not cv2.imwrite(str(frames / f"{i:06d}.png"), frame):
                raise OSError(f"Cannot save frame {i}")
    finally:
        cap.release()
    resources = []
    for start, end in windows:
        middle = (start + end) // 2
        for indices in (list(range(middle, end + 1)), list(range(middle, start - 1, -1))):
            folder = directory / str(len(resources))
            folder.mkdir()
            for j, i in enumerate(indices):
                (folder / f"{j:06d}.png").symlink_to(frames / f"{i:06d}.png")
            resources.append((indices, folder))
    return resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if n < 1 or fps <= 0:
        raise ValueError("Input video must contain frames and a positive frame rate")

    windows = propagation_windows(n, fps)
    tracker_windows = list(dict.fromkeys([*windows, (0, n - 1)]))
    prompts = {"arm": ["human arm"], "glove": ["blue glove"], "tracker": ["wristband", "watch"]}

    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(args.checkpoint),
        use_fa3=False,
        compile=False,
        async_loading_frames=False,
    )
    args.output.mkdir(parents=True)
    stats = {}
    started = time.monotonic()
    with TemporaryDirectory(dir=args.output, prefix="frames-") as temporary:
        resources = window_resources(args.video, tracker_windows, Path(temporary).resolve(), n)
        for name, texts in prompts.items():
            class_windows = tracker_windows if name == "tracker" else windows
            class_resources = resources[: 2 * len(class_windows)]
            folder = args.output / name
            folder.mkdir()
            seen = set()
            areas = {}
            for prompt in texts:
                for indices, resource in class_resources:
                    session = predictor.handle_request(
                        {
                            "type": "start_session",
                            "resource_path": str(resource),
                            "offload_video_to_cpu": True,
                        }
                    )["session_id"]
                    try:
                        predictor.handle_request(
                            {
                                "type": "add_prompt",
                                "session_id": session,
                                "frame_index": 0,
                                "text": prompt,
                            }
                        )
                        for response in predictor.handle_stream_request(
                            {
                                "type": "propagate_in_video",
                                "session_id": session,
                                "propagation_direction": "forward",
                                "start_frame_index": 0,
                            }
                        ):
                            index = int(response["frame_index"])
                            if not 0 <= index < len(indices):
                                raise RuntimeError(f"Unexpected frame index: {index}")
                            i = indices[index]
                            mask = binary_union(response["outputs"], h, w)
                            if i in seen:
                                mask |= cv2.imread(str(folder / f"{i:06d}.png"), 0)
                            if not cv2.imwrite(str(folder / f"{i:06d}.png"), mask):
                                raise OSError(f"Cannot save mask {i}")
                            areas[i] = int(np.count_nonzero(mask))
                            seen.add(i)
                            if i % 50 == 0:
                                print(f"{name}: frame {i + 1}/{n}; pixels={areas[i]}", flush=True)
                    finally:
                        predictor.handle_request({"type": "close_session", "session_id": session})
            if seen != set(range(n)):
                raise RuntimeError(f"Incomplete {name} masks: {len(seen)}/{n}")
            stats[name] = {
                "text_prompts": texts,
                "prompt_frames": [(start + end) // 2 for start, end in class_windows],
                "frames": n,
                "empty_frames": list(areas.values()).count(0),
                "mean_area_pixels": float(np.mean(list(areas.values()))),
            }

    for name in ("removal", "gloves"):
        (args.output / name).mkdir()
    for i in range(n):
        masks = {
            name: cv2.imread(str(args.output / name / f"{i:06d}.png"), 0) > 127
            for name in ("arm", "glove", "tracker")
        }
        removal = (masks["arm"] | masks["tracker"]) & ~masks["glove"]
        cv2.imwrite(
            str(args.output / "removal" / f"{i:06d}.png"),
            removal.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(args.output / "gloves" / f"{i:06d}.png"),
            masks["glove"].astype(np.uint8) * 255,
        )
    report = {
        "model": "SAM 3.1",
        "checkpoint_sha256": digest(args.checkpoint),
        "input_sha256": digest(args.video),
        "frames": n,
        "width": w,
        "height": h,
        "fps": fps,
        "seed": args.seed,
        "prompts": stats,
        "propagation": "independent_bidirectional_windows",
        "windows": windows,
        "tracker_windows": tracker_windows,
        "seconds": time.monotonic() - started,
        "torch": torch.__version__,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
