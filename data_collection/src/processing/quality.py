"""Structural and signal-quality diagnostics for raw or synchronized Pico datasets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from human.processing.resample import probe_video_stream


BODY_POSE = "observation.pico.body_pose"
BODY_AVAILABLE = "observation.pico.body_available"
LEFT_HAND_POSE = "observation.pico.left_hand_pose"
RIGHT_HAND_POSE = "observation.pico.right_hand_pose"
LEFT_HAND_ACTIVE = "observation.pico.left_hand_active"
RIGHT_HAND_ACTIVE = "observation.pico.right_hand_active"


@dataclass(frozen=True)
class QualityThresholds:
    """Task-independent capture integrity limits, expressed in physical units."""

    min_frames: int = 100
    max_body_identical_transition_fraction: float = 0.25
    max_body_identical_run_frames: int = 50
    # OpenXR may relocalize an individual distal joint for one frame.  Keep the
    # diagnostic limit above those observed single-joint corrections while
    # recording larger whole-capture discontinuities.
    max_body_position_step_m: float = 0.75
    max_body_rotation_step_deg: float = 120.0
    max_timestamp_step_error_ms: float = 5.0
    min_body_available_fraction: float = 0.995
    quaternion_norm_tolerance: float = 0.05
    # Palm-center recovery may bridge long optical gaps from the tracked body
    # hand pose, but it still needs a small set of direct OpenXR samples for
    # the per-episode body-hand transform and anatomical orientation fit.
    min_hand_direct_valid_frames: int = 10


def _as_array(
    arrays: Mapping[str, Any],
    name: str,
    *,
    dtype: Any,
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"missing required column: {name}")
    return np.asarray(arrays[name], dtype=dtype)


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _quaternion_validity(
    poses: np.ndarray,
    active: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    if poses.ndim != 3 or poses.shape[2] != 7:
        return False
    selected = poses[np.asarray(active, dtype=bool), :, 3:7]
    if selected.size == 0:
        return True
    norms = np.linalg.norm(selected, axis=2)
    return bool(
        np.isfinite(norms).all()
        and np.all(norms > 1e-8)
        and np.max(np.abs(norms - 1.0)) <= tolerance
    )


def _max_rotation_step_deg(poses: np.ndarray, active: np.ndarray) -> float:
    if len(poses) < 2:
        return 0.0
    q0 = np.asarray(poses[:-1, :, 3:7], dtype=np.float64)
    q1 = np.asarray(poses[1:, :, 3:7], dtype=np.float64)
    valid = np.asarray(active[:-1] & active[1:], dtype=bool)[:, None]
    n0 = np.linalg.norm(q0, axis=2)
    n1 = np.linalg.norm(q1, axis=2)
    valid = valid & (n0 > 1e-8) & (n1 > 1e-8)
    if not np.any(valid):
        return 0.0
    dots = np.abs(np.sum(q0 * q1, axis=2) / np.maximum(n0 * n1, 1e-12))
    angles = np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))
    return float(np.max(angles[valid]))


def analyze_episode_arrays(
    arrays: Mapping[str, Any],
    *,
    expected_fps: float,
    thresholds: QualityThresholds | None = None,
    hand_tracking_policy: str = "both",
) -> dict[str, Any]:
    """Evaluate one episode without task-specific motion assumptions."""

    limits = thresholds or QualityThresholds()
    if hand_tracking_policy not in {"both", "either", "ignore"}:
        raise ValueError("hand_tracking_policy must be 'both', 'either', or 'ignore'")
    timestamp = _as_array(arrays, "timestamp", dtype=np.float64).reshape(-1)
    frame_index = _as_array(arrays, "frame_index", dtype=np.int64).reshape(-1)
    body_available = _as_array(arrays, BODY_AVAILABLE, dtype=bool).reshape(-1)
    body = _as_array(arrays, BODY_POSE, dtype=np.float64)
    left_hand = _as_array(arrays, LEFT_HAND_POSE, dtype=np.float64)
    right_hand = _as_array(arrays, RIGHT_HAND_POSE, dtype=np.float64)
    left_active = _as_array(arrays, LEFT_HAND_ACTIVE, dtype=bool).reshape(-1)
    right_active = _as_array(arrays, RIGHT_HAND_ACTIVE, dtype=bool).reshape(-1)
    frames = len(timestamp)

    issues: list[str] = []
    expected_shapes = {
        BODY_POSE: (frames, 24, 7),
        LEFT_HAND_POSE: (frames, 26, 7),
        RIGHT_HAND_POSE: (frames, 26, 7),
    }
    observed = {
        BODY_POSE: body.shape,
        LEFT_HAND_POSE: left_hand.shape,
        RIGHT_HAND_POSE: right_hand.shape,
    }
    if any(observed[name] != shape for name, shape in expected_shapes.items()):
        issues.append("invalid_pose_shape")
        return {
            "issues": issues,
            "metrics": {
                "frames": frames,
                "pose_shapes": {name: list(shape) for name, shape in observed.items()},
            },
        }
    vector_lengths = {
        len(frame_index),
        len(body_available),
        len(left_active),
        len(right_active),
        frames,
    }
    if len(vector_lengths) != 1:
        issues.append("column_length_mismatch")

    if frames < limits.min_frames:
        issues.append("too_few_frames")
    if not np.array_equal(frame_index, np.arange(frames, dtype=np.int64)):
        issues.append("noncontiguous_frame_index")

    timestamp_steps = np.diff(timestamp)
    expected_step = 1.0 / float(expected_fps)
    timestamp_step_max_error_ms = (
        float(np.max(np.abs(timestamp_steps - expected_step)) * 1000.0)
        if len(timestamp_steps)
        else 0.0
    )
    if not np.isfinite(timestamp).all():
        issues.append("nonfinite_timestamp")
    if np.any(timestamp_steps <= 0.0):
        issues.append("nonmonotonic_timestamp")
    if timestamp_step_max_error_ms > limits.max_timestamp_step_error_ms:
        issues.append("timestamp_cadence")
    if "observation.pico.local_timestamp_ns" in arrays:
        clock = np.asarray(arrays["observation.pico.local_timestamp_ns"], dtype=np.float64).reshape(-1)
        if len(clock) != frames or not np.isfinite(clock).all() or np.any(np.diff(clock) <= 0):
            issues.append("invalid_local_clock")
    if "observation.pico.body_joint_timestamp_ns" in arrays:
        clocks = np.asarray(arrays["observation.pico.body_joint_timestamp_ns"], dtype=np.float64)
        if clocks.shape != (frames, 24) or not np.isfinite(clocks).all() or np.any(np.diff(clocks, axis=0) < 0):
            issues.append("invalid_body_clock")
        elif _longest_true_run(np.all(np.diff(clocks, axis=0) == 0, axis=1)) >= limits.max_body_identical_run_frames:
            issues.append("body_timestamp_stalled")

    finite_poses = bool(
        np.isfinite(body).all()
        and (
            hand_tracking_policy == "ignore"
            or (np.isfinite(left_hand).all() and np.isfinite(right_hand).all())
        )
    )
    if not finite_poses:
        issues.append("nonfinite_pose")
    body_available_fraction = float(np.mean(body_available)) if frames else 0.0
    if body_available_fraction < limits.min_body_available_fraction:
        issues.append("body_tracking_unavailable")

    if not _quaternion_validity(
        body,
        body_available,
        tolerance=limits.quaternion_norm_tolerance,
    ):
        issues.append("invalid_body_quaternion")
    if hand_tracking_policy != "ignore" and not (
        _quaternion_validity(
            left_hand,
            left_active,
            tolerance=limits.quaternion_norm_tolerance,
        )
        and _quaternion_validity(
            right_hand,
            right_active,
            tolerance=limits.quaternion_norm_tolerance,
        )
    ):
        issues.append("invalid_hand_quaternion")

    hand_tracking_metrics: list[dict[str, Any]] = []
    hand_direct_sufficient: list[bool] = []
    for side, poses, active in (
        ("left", left_hand, left_active),
        ("right", right_hand, right_active),
    ):
        palm = poses[:, 0]
        direct_valid = (
            active
            & np.all(np.isfinite(palm), axis=1)
            & (np.linalg.norm(palm[:, 3:7], axis=1) > 1e-8)
        )
        direct_frames = int(np.count_nonzero(direct_valid))
        unavailable_run = _longest_true_run(~direct_valid)
        hand_tracking_metrics.append(
            {
                "direct_valid_frames": direct_frames,
                "direct_valid_fraction": (
                    direct_frames / frames if frames else 0.0
                ),
                "longest_unavailable_run_frames": int(unavailable_run),
            }
        )
        hand_direct_sufficient.append(
            direct_frames >= limits.min_hand_direct_valid_frames
        )
        if hand_tracking_policy == "both" and not hand_direct_sufficient[-1]:
            issues.append(f"{side}_hand_insufficient_direct_tracking")
    if hand_tracking_policy == "either" and not any(hand_direct_sufficient):
        issues.append("hands_insufficient_direct_tracking")

    body_delta = np.diff(body, axis=0)
    identical = (
        np.max(np.abs(body_delta), axis=(1, 2)) <= 1e-12
        if len(body_delta)
        else np.zeros(0, dtype=bool)
    )
    identical_fraction = float(np.mean(identical)) if len(identical) else 0.0
    longest_identical_run_frames = _longest_true_run(identical) + (
        1 if np.any(identical) else 0
    )
    warnings = []
    if identical_fraction > limits.max_body_identical_transition_fraction:
        warnings.append("body_repeated_frames")
    if longest_identical_run_frames > limits.max_body_identical_run_frames:
        warnings.append("body_frozen_run")

    body_position_steps = (
        np.linalg.norm(np.diff(body[:, :, :3], axis=0), axis=2)
        if frames > 1
        else np.zeros((0, 24), dtype=np.float64)
    )
    body_position_step_max_m = (
        float(np.nanmax(body_position_steps)) if body_position_steps.size else 0.0
    )
    body_rotation_step_max_deg = (
        _max_rotation_step_deg(body, body_available)
        if finite_poses
        else float("nan")
    )
    if body_position_step_max_m > limits.max_body_position_step_m:
        issues.append("body_position_jump")
    if (
        np.isfinite(body_rotation_step_max_deg)
        and body_rotation_step_max_deg > limits.max_body_rotation_step_deg
    ):
        issues.append("body_rotation_jump")

    return {
        "issues": list(dict.fromkeys(issues)),
        "warnings": warnings,
        "metrics": {
            "frames": int(frames),
            "duration_s": float(timestamp[-1] - timestamp[0]) if frames else 0.0,
            "timestamp_step_max_error_ms": timestamp_step_max_error_ms,
            "body_available_fraction": body_available_fraction,
            "hand_active_fraction": [
                float(np.mean(left_active)) if frames else 0.0,
                float(np.mean(right_active)) if frames else 0.0,
            ],
            "hand_tracking": hand_tracking_metrics,
            "body_identical_transition_fraction": identical_fraction,
            "body_longest_identical_run_frames": int(longest_identical_run_frames),
            "body_position_step_max_m": body_position_step_max_m,
            "body_rotation_step_max_deg": body_rotation_step_max_deg,
        },
    }


def _episode_path(root: Path, info: Mapping[str, Any], episode: int) -> Path:
    return root / str(info["data_path"]).format(
        episode_chunk=episode // int(info["chunks_size"]),
        episode_index=episode,
    )


def _video_paths(root: Path, info: Mapping[str, Any], episode: int) -> list[Path]:
    paths: list[Path] = []
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") != "video":
            continue
        paths.append(
            root
            / str(info["video_path"]).format(
                episode_chunk=episode // int(info["chunks_size"]),
                episode_index=episode,
                video_key=key,
            )
        )
    return paths


def _audit_episode(dataset_root: Path, info: Mapping[str, Any], **kwargs) -> dict[str, Any]:
    try:
        return _inspect_episode(dataset_root, info, **kwargs)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, pa.ArrowException, subprocess.SubprocessError) as exc:
        return {"episode_index": kwargs["episode"],
                "issues": ["unreadable_episode"], "error": str(exc), "metrics": {"frames": 0}}


def _inspect_episode(
    dataset_root: Path,
    info: Mapping[str, Any],
    *,
    episode: int,
    fps: float,
    require_exact_video_frames: bool,
    ffprobe: str,
    thresholds: QualityThresholds | None,
    hand_tracking_policy: str,
) -> dict[str, Any]:
    parquet_path = _episode_path(dataset_root, info, episode)
    if not parquet_path.exists():
        result = {
            "issues": ["missing_parquet"],
            "metrics": {"frames": 0},
        }
    else:
        if not parquet_path.resolve().is_relative_to(dataset_root):
            raise ValueError("parquet path escapes dataset")
        table = pq.read_table(parquet_path)
        arrays = {name: table[name].to_pylist() for name in table.column_names}
        result = analyze_episode_arrays(
            arrays,
            expected_fps=fps,
            thresholds=thresholds,
            hand_tracking_policy=hand_tracking_policy,
        )
        rows = info.get("_episodes", {})
        if episode in rows and rows[episode]["length"] != len(table):
            result["issues"].append("episode_length_mismatch")
        if "episode_index" not in arrays or not np.all(np.asarray(arrays["episode_index"]) == episode):
            result["issues"].append("episode_index_mismatch")
        tasks = info.get("_tasks", {})
        ids = set(np.asarray(arrays.get("task_index", []), dtype=int).reshape(-1).tolist())
        if not ids or not ids.issubset(tasks):
            result["issues"].append("unknown_task_index")
        elif episode in rows and set(rows[episode]["tasks"]) != {tasks[i]["task"] for i in ids}:
            result["issues"].append("episode_tasks_mismatch")
        if episode in rows:
            start = sum(rows[i]["length"] for i in sorted(rows) if i < episode)
            if not np.array_equal(np.asarray(arrays.get("index", [])).reshape(-1), np.arange(start, start + len(table))):
                result["issues"].append("global_index_mismatch")
        declared = {key for key, feature in info["features"].items() if feature.get("dtype") != "video"}
        if set(arrays) != declared:
            result["issues"].append("feature_columns_mismatch")
    result = {"episode_index": episode, **result}
    if result["metrics"]["frames"]:
        video_metrics: list[dict[str, Any]] = []
        features = [value for value in info["features"].values() if value.get("dtype") == "video"]
        for video_path, feature in zip(_video_paths(dataset_root, info, episode), features):
            if not video_path.resolve().is_relative_to(dataset_root):
                raise ValueError("video path escapes dataset")
            if not video_path.exists():
                result["issues"].append("missing_video")
                video_metrics.append({"path": str(video_path), "exists": False})
                continue
            count, video_fps = probe_video_stream(video_path, ffprobe=ffprobe)
            declared_fps = feature.get("video_info", feature.get("info", {})).get("video.fps")
            if declared_fps is not None and (not np.isfinite(float(declared_fps)) or abs(video_fps - float(declared_fps)) > .02):
                result["issues"].append("video_metadata_fps_mismatch")
            matches = (
                count == result["metrics"]["frames"]
                and abs(video_fps - fps) <= 1e-6
            )
            video_metrics.append(
                {
                    "path": str(video_path),
                    "exists": True,
                    "frames": count,
                    "fps": video_fps,
                    "exact_match": matches,
                }
            )
            if require_exact_video_frames and not matches:
                result["issues"].append("video_frame_mismatch")
        result["metrics"]["videos"] = video_metrics
    return result


def _metadata_issues(root: Path, info: dict) -> list[str]:
    issues = []
    total = int(info["total_episodes"])
    if total <= 0 or int(info["chunks_size"]) <= 0 or not np.isfinite(float(info["fps"])) or float(info["fps"]) <= 0:
        raise ValueError("invalid total_episodes/chunks_size/fps")
    for filename, key, count, target in (
        ("episodes.jsonl", "episode_index", total, "_episodes"),
        ("tasks.jsonl", "task_index", int(info.get("total_tasks", 0)), "_tasks"),
    ):
        try:
            rows = [json.loads(line) for line in (root / "meta" / filename).read_text().splitlines() if line.strip()]
            if sorted(row[key] for row in rows) != list(range(count)):
                issues.append(f"invalid_{filename}")
            info[target] = {row[key]: row for row in rows}
        except (OSError, ValueError, KeyError, TypeError):
            info[target] = {}
            issues.append(f"invalid_{filename}")
    expected_data = {_episode_path(root, info, ep).resolve() for ep in range(total)}
    expected_video = {path.resolve() for ep in range(total) for path in _video_paths(root, info, ep)}
    if any(not path.is_relative_to(root) for path in expected_data | expected_video):
        raise ValueError("metadata path escapes dataset")
    if expected_data != {p.resolve() for p in (root / "data").rglob("*.parquet")}:
        issues.append("episode_file_set_mismatch")
    if expected_video != {p.resolve() for p in (root / "videos").rglob("*.mp4")}:
        issues.append("video_file_set_mismatch")
    if info.get("total_videos", len(expected_video)) != len(expected_video):
        issues.append("total_videos_mismatch")
    return issues


def audit_dataset(
    root: str | Path,
    *,
    expected_fps: float | None = None,
    require_exact_video_frames: bool = False,
    ffprobe: str = "ffprobe",
    thresholds: QualityThresholds | None = None,
    hand_tracking_policy: str = "both",
    workers: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Record structural and signal diagnostics for every declared episode."""

    if hand_tracking_policy not in {"both", "either", "ignore"}:
        raise ValueError("hand_tracking_policy must be 'both', 'either', or 'ignore'")
    dataset_root = Path(root).expanduser().resolve()
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    issues = _metadata_issues(dataset_root, info)
    fps = float(expected_fps if expected_fps is not None else info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("expected_fps must be positive and finite")
    total_episodes = int(info["total_episodes"])
    episode_reports: list[dict[str, Any] | None] = [None] * total_episodes
    episodes_with_issues: list[int] = []
    completed = 0

    def record_progress() -> None:
        nonlocal completed
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, total_episodes)

    audit_kwargs = {
        "fps": fps,
        "require_exact_video_frames": require_exact_video_frames,
        "ffprobe": ffprobe,
        "thresholds": thresholds,
        "hand_tracking_policy": hand_tracking_policy,
    }
    worker_count = min(max(1, int(workers)), max(1, total_episodes))
    if worker_count == 1:
        for episode in range(total_episodes):
            episode_reports[episode] = _audit_episode(
                dataset_root,
                info,
                episode=episode,
                **audit_kwargs,
            )
            record_progress()
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(
                    _audit_episode,
                    dataset_root,
                    info,
                    episode=episode,
                    **audit_kwargs,
                ): episode
                for episode in range(total_episodes)
            }
            for future in as_completed(futures):
                episode = futures[future]
                episode_reports[episode] = future.result()
                record_progress()

    ordered_reports: list[dict[str, Any]] = []
    for episode, result in enumerate(episode_reports):
        if result is None:
            raise RuntimeError(f"episode audit produced no result: {episode}")
        if result["issues"]:
            episodes_with_issues.append(episode)
        ordered_reports.append(result)

    actual_frames = sum(item["metrics"].get("frames", 0) for item in ordered_reports)
    if info.get("total_frames") != actual_frames:
        issues.append("total_frames_mismatch")
    report = {
        "schema_version": 1,
        "source_root": str(dataset_root),
        "expected_fps": fps,
        "thresholds": asdict(thresholds or QualityThresholds()),
        "hand_tracking_policy": hand_tracking_policy,
        "total_episodes": total_episodes,
        "total_frames": int(
            sum(item["metrics"].get("frames", 0) for item in ordered_reports)
        ),
        "issues": issues,
        "episodes_with_issues": episodes_with_issues,
        "episodes": ordered_reports,
    }
    return report


def write_report(path: Path, report: dict) -> None:
    """Keep nonfinite numerical diagnostics machine-readable as JSON nulls."""
    content = json.loads(json.dumps(report), parse_constant=lambda _: None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-fps", type=float)
    parser.add_argument("--require-exact-video-frames", action="store_true")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--hand-tracking-policy",
        choices=("both", "either", "ignore"),
        default="both",
        help=(
            "Require direct calibration samples from both hands per episode, or "
            "from at least one hand when a later validated cohort calibration "
            "will reconstruct the other hand."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Audit independent episodes concurrently while preserving report order.",
    )
    args = parser.parse_args(argv)
    started = time.monotonic()

    def print_progress(done: int, total: int) -> None:
        interval = max(1, total // 20)
        if done != 1 and done != total and done % interval:
            return
        elapsed = time.monotonic() - started
        eta = elapsed * (total - done) / done if done else float("nan")
        print(
            f"[quality] {done}/{total} ({100.0 * done / total:5.1f}%) "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )

    report = audit_dataset(
        args.root,
        expected_fps=args.expected_fps,
        require_exact_video_frames=args.require_exact_video_frames,
        ffprobe=args.ffprobe,
        hand_tracking_policy=args.hand_tracking_policy,
        workers=args.workers,
        progress_callback=print_progress,
    )
    report_path = args.report or args.root / "meta/data_quality_report.json"
    write_report(report_path, report)
    print(
        f"recorded {report['total_episodes']} episodes; "
        f"episodes_with_issues={report['episodes_with_issues']} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
