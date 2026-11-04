"""Materialize a frame-synchronized copy of a Pico/GoPro dataset."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from human.devices.gopro import GOPRO_VIDEO_KEY
from human.storage.lerobot import _episode_stats
from human.storage.lerobot import discover_episode_indices


def _alignment_time_model(alignment: dict[str, Any]) -> tuple[float, float]:
    model = str(alignment.get("alignment_model", "constant"))
    if model == "affine":
        offset_s = float(alignment["affine_offset_s"])
        time_scale = 1.0 + float(alignment.get("drift_s_per_s", 0.0))
    else:
        offset_s = float(alignment["offset_s"])
        time_scale = 1.0
    if not math.isfinite(offset_s) or not math.isfinite(time_scale):
        raise ValueError("Alignment timing values must be finite")
    if time_scale <= 0.0:
        raise ValueError(f"Alignment time scale must be positive, got {time_scale}")
    return offset_s, time_scale


def compute_overlap_frame_range(
    alignment: dict[str, Any],
    *,
    action_frames: int,
    action_fps: float,
    source_video_frames: int,
    source_video_fps: float,
) -> dict[str, int | float]:
    """Find action rows whose timestamps have real source-video coverage."""

    action_frames = int(action_frames)
    source_video_frames = int(source_video_frames)
    action_fps = float(action_fps)
    source_video_fps = float(source_video_fps)
    if action_frames <= 0 or source_video_frames <= 0:
        raise ValueError("Action and source video frame counts must be positive")
    if action_fps <= 0.0 or source_video_fps <= 0.0:
        raise ValueError("Action and source video FPS must be positive")
    offset_s, time_scale = _alignment_time_model(alignment)
    video_start_s = offset_s
    video_stop_s = offset_s + time_scale * (
        (source_video_frames - 1) / source_video_fps
    )
    epsilon = 1e-9
    action_start_frame = max(
        0,
        int(math.ceil(video_start_s * action_fps - epsilon)),
    )
    action_stop_frame = min(
        action_frames,
        int(math.floor(video_stop_s * action_fps + epsilon)) + 1,
    )
    if action_stop_frame <= action_start_frame:
        raise ValueError("Pico actions and aligned GoPro video do not overlap")
    action_start_s = action_start_frame / action_fps
    camera_start_s = (action_start_s - offset_s) / time_scale
    if camera_start_s < -epsilon:
        raise ValueError("Computed overlap begins before the source video")
    camera_start_s = max(0.0, camera_start_s)
    return {
        "action_start_frame": action_start_frame,
        "action_stop_frame": action_stop_frame,
        "overlap_frames": action_stop_frame - action_start_frame,
        "action_start_s": action_start_s,
        "camera_start_s": camera_start_s,
        "video_start_on_action_axis_s": video_start_s,
        "video_stop_on_action_axis_s": video_stop_s,
    }


def _episode_time_axis_s(
    parquet_path: Path,
    *,
    nominal_fps: float,
) -> tuple[np.ndarray, str, int | None]:
    """Return the real Pico sample clock, preferring host capture timestamps."""

    parquet = pq.ParquetFile(parquet_path)
    names = set(parquet.schema_arrow.names)
    local_name = "observation.pico.local_timestamp_ns"
    if local_name in names:
        values = np.asarray(
            pq.read_table(parquet_path, columns=[local_name])[local_name],
            dtype=np.int64,
        )
        if len(values) < 1:
            raise ValueError(f"Pico local timestamp column is empty: {parquet_path}")
        if np.any(np.diff(values) <= 0):
            raise ValueError(
                f"Pico local timestamps must be strictly increasing: {parquet_path}"
            )
        origin_ns = int(values[0])
        return (values - origin_ns).astype(np.float64) * 1e-9, local_name, origin_ns

    frame_count = int(parquet.metadata.num_rows)
    return (
        np.arange(frame_count, dtype=np.float64) / float(nominal_fps),
        "nominal_fps",
        None,
    )


def _compute_overlap_time_grid(
    alignment: dict[str, Any],
    *,
    action_timestamps_s: np.ndarray,
    output_fps: float,
    source_video_frames: int,
    source_video_fps: float,
) -> dict[str, Any]:
    """Build the fixed-rate Pico grid covered by both real sample and video time."""

    timestamps = np.asarray(action_timestamps_s, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 1:
        raise ValueError("Action timestamps must be a non-empty 1-D array")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Action timestamps must be finite and strictly increasing")
    if output_fps <= 0.0 or source_video_fps <= 0.0 or source_video_frames <= 0:
        raise ValueError("Output and source-video timing must be positive")

    offset_s, time_scale = _alignment_time_model(alignment)
    video_start_s = offset_s
    video_stop_s = offset_s + time_scale * (
        (int(source_video_frames) - 1) / float(source_video_fps)
    )
    overlap_start_s = max(float(timestamps[0]), video_start_s)
    overlap_stop_s = min(float(timestamps[-1]), video_stop_s)
    epsilon = 1e-9
    start_tick = int(math.ceil(overlap_start_s * output_fps - epsilon))
    stop_tick = int(math.floor(overlap_stop_s * output_fps + epsilon)) + 1
    if stop_tick <= start_tick:
        raise ValueError("Pico actions and aligned GoPro video do not overlap")
    target_times_s = np.arange(start_tick, stop_tick, dtype=np.float64) / output_fps
    action_start_s = float(target_times_s[0])
    camera_start_s = (action_start_s - offset_s) / time_scale
    if camera_start_s < -epsilon:
        raise ValueError("Computed overlap begins before the source video")
    source_start_index = int(np.searchsorted(timestamps, target_times_s[0], side="left"))
    source_stop_index = int(np.searchsorted(timestamps, target_times_s[-1], side="right"))
    return {
        "action_start_frame": start_tick,
        "action_stop_frame": stop_tick,
        "overlap_frames": len(target_times_s),
        "action_start_s": action_start_s,
        "camera_start_s": max(0.0, camera_start_s),
        "video_start_on_action_axis_s": video_start_s,
        "video_stop_on_action_axis_s": video_stop_s,
        "source_action_start_index": source_start_index,
        "source_action_stop_index": source_stop_index,
        "source_action_duration_s": float(timestamps[-1] - timestamps[0]),
        "target_times_s": target_times_s,
    }


def build_video_filter(
    alignment: dict[str, Any],
    *,
    duration_s: float,
    output_fps: float,
    frame_count: int | None = None,
    action_start_s: float = 0.0,
) -> str:
    """Build an ffmpeg filter that maps native camera PTS onto Pico time."""

    offset_s, time_scale = _alignment_time_model(alignment)
    duration_s = float(duration_s)
    output_fps = float(output_fps)
    action_start_s = float(action_start_s)
    if not all(
        math.isfinite(value) for value in (duration_s, output_fps, action_start_s)
    ):
        raise ValueError("Alignment timing values must be finite")
    if duration_s <= 0.0 or output_fps <= 0.0:
        raise ValueError("Output duration and FPS must be positive")
    if frame_count is not None and int(frame_count) <= 0:
        raise ValueError("frame_count must be positive")

    camera_start_s = (action_start_s - offset_s) / time_scale
    if camera_start_s < -1e-9:
        raise ValueError("action_start_s has no source-video coverage")
    camera_start_s = max(0.0, camera_start_s)
    filters = [
        f"trim=start={camera_start_s:.6f}",
        f"setpts=(PTS-STARTPTS)*{time_scale:.9f}",
        f"fps={output_fps:.6f}:eof_action=pass",
        (
            f"trim=end_frame={int(frame_count)}"
            if frame_count is not None
            else f"trim=duration={duration_s:.6f}"
        ),
        "setpts=PTS-STARTPTS",
    ]
    return ",".join(filters)


def probe_video_stream(
    video_path: str | Path,
    *,
    ffprobe: str = "ffprobe",
) -> tuple[int, float]:
    """Return the decoded video frame count and average frame rate."""

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    try:
        stream = payload["streams"][0]
        raw_count = stream.get("nb_read_frames") or stream["nb_frames"]
        frame_count = int(raw_count)
        fps = float(Fraction(stream["avg_frame_rate"]))
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Could not probe video timing: {video_path}") from exc
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid video timing: {video_path}")
    return frame_count, fps


def apply_dataset_sync(
    root: str | Path,
    *,
    output_root: str | Path,
    output_fps: float | None = None,
    overwrite: bool = False,
    workers: int = 1,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    """Create a synchronized dataset copy while leaving ``root`` untouched.

    Actions and video are both cropped to their real common time interval.
    Episode parquets, indices, metadata, and statistics are regenerated, and
    every video is newly rendered on the same fixed-rate action clock.
    """

    root = Path(root).resolve()
    output_root = Path(output_root).resolve()
    if root == output_root:
        raise ValueError("output_root must differ from the raw dataset root")
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output dataset already exists: {output_root}")
    worker_count = int(workers)
    if worker_count <= 0:
        raise ValueError("workers must be > 0")

    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}") from exc
    video_template = info.get("video_path")
    if not isinstance(video_template, str):
        raise ValueError("Only single-camera string video_path datasets are supported")
    data_template = info.get("data_path")
    if not isinstance(data_template, str):
        raise ValueError("Dataset data_path must be a string")
    chunks_size = int(info.get("chunks_size", 1000))
    target_fps = float(output_fps if output_fps is not None else info.get("fps", 50.0))

    targets = []
    for episode_index in discover_episode_indices(root):
        relative_video = Path(
            video_template.format(
                episode_chunk=episode_index // chunks_size,
                episode_index=episode_index,
                video_key=GOPRO_VIDEO_KEY,
            )
        )
        source_video = root / relative_video
        relative_parquet = Path(
            data_template.format(
                episode_chunk=episode_index // chunks_size,
                episode_index=episode_index,
            )
        )
        source_parquet = root / relative_parquet
        alignment_path = root / "sync" / f"episode_{episode_index:06d}.gopro_alignment.json"
        if not source_video.exists():
            raise FileNotFoundError(f"Source video not found: {source_video}")
        try:
            alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Alignment result not found: {alignment_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Alignment result is invalid JSON: {alignment_path}") from exc
        if not source_parquet.exists():
            raise FileNotFoundError(f"Source parquet not found: {source_parquet}")
        frame_count = int(pq.ParquetFile(source_parquet).metadata.num_rows)
        if frame_count <= 0:
            raise ValueError(f"Source parquet has no action frames: {source_parquet}")
        action_timestamps_s, action_time_source, local_time_origin_ns = (
            _episode_time_axis_s(source_parquet, nominal_fps=target_fps)
        )
        if alignment.get("alignment_evidence") == "exact_frame_identity_only":
            # The prealigned fallback has just decoded and verified this exact
            # stream before writing the manifest. Reuse that auditable result
            # instead of decoding every long video a second time.
            source_video_frames = int(alignment["source_video_frames"])
            source_video_fps = float(alignment["source_video_fps"])
            if source_video_frames != frame_count:
                raise ValueError(
                    f"Episode {episode_index} prealigned manifest frame count "
                    f"does not match Parquet: {source_video_frames}/{frame_count}"
                )
        else:
            source_video_frames, source_video_fps = probe_video_stream(
                source_video,
                ffprobe=ffprobe,
            )
        overlap = _compute_overlap_time_grid(
            alignment,
            action_timestamps_s=action_timestamps_s,
            output_fps=target_fps,
            source_video_frames=source_video_frames,
            source_video_fps=source_video_fps,
        )
        overlap_frames = int(overlap["overlap_frames"])
        duration_s = overlap_frames / target_fps
        targets.append(
            (
                episode_index,
                relative_video,
                source_video,
                relative_parquet,
                alignment,
                duration_s,
                frame_count,
                source_video_frames,
                source_video_fps,
                action_timestamps_s,
                action_time_source,
                local_time_origin_ns,
                overlap,
            )
        )

    temporary_root = output_root.with_name(f".{output_root.name}.syncing")
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    report: dict[str, Any] = {
        "source_root": str(root),
        "output_root": str(output_root),
        "output_fps": target_fps,
        "workers": worker_count,
        "applied": [],
    }
    try:
        _copy_non_video_dataset(root, temporary_root)
        source_episodes = {
            int(row["episode_index"]): row
            for row in _load_jsonl(root / "meta" / "episodes.jsonl")
        }
        output_episodes = []
        output_stats = []
        total_output_frames = 0
        materialize_started = time.monotonic()
        completed_exports = 0
        pending_videos: list[dict[str, Any]] = []

        def finish_video(job: dict[str, Any]) -> None:
            nonlocal completed_exports
            process = job.get("process")
            if process is not None:
                return_code = process.wait()
                if return_code:
                    raise subprocess.CalledProcessError(
                        return_code, job["command"]
                    )
            video_frames, video_fps = probe_video_stream(
                job["output_video"],
                ffprobe=ffprobe,
            )
            frame_count = int(job["frame_count"])
            episode_index = int(job["episode_index"])
            if video_frames != frame_count:
                raise RuntimeError(
                    f"Episode {episode_index} expected {frame_count} frames, "
                    f"got {video_frames}"
                )
            if not math.isclose(
                video_fps, target_fps, rel_tol=0.0, abs_tol=1e-6
            ):
                raise RuntimeError(
                    f"Episode {episode_index} expected {target_fps:.6f} FPS, "
                    f"got {video_fps:.6f}"
                )
            item = dict(job["report_item"])
            item.update(
                {
                    "video_frames": video_frames,
                    "video_fps": video_fps,
                    "frame_count_match": video_frames == frame_count,
                }
            )
            report["applied"].append(item)
            completed_exports += 1
            elapsed = max(1e-6, time.monotonic() - materialize_started)
            rate = completed_exports / elapsed
            eta = (
                (len(targets) - completed_exports) / rate
                if rate > 0.0
                else 0.0
            )
            print(
                f"[sync-export x{worker_count}] {completed_exports}/{len(targets)} "
                f"({100.0 * completed_exports / len(targets):5.1f}%) "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

        for (
            episode_index,
            relative_video,
            source_video,
            relative_parquet,
            alignment,
            duration_s,
            source_action_frames,
            source_video_frames,
            source_video_fps,
            action_timestamps_s,
            action_time_source,
            local_time_origin_ns,
            overlap,
        ) in targets:
            action_start_frame = int(overlap["action_start_frame"])
            action_stop_frame = int(overlap["action_stop_frame"])
            frame_count = int(overlap["overlap_frames"])
            output_parquet = temporary_root / relative_parquet
            cropped_table = _resample_episode_parquet(
                output_parquet,
                source_times_s=action_timestamps_s,
                target_times_s=np.asarray(overlap["target_times_s"], dtype=np.float64),
                episode_index=episode_index,
                global_start=total_output_frames,
                fps=target_fps,
                local_time_origin_ns=local_time_origin_ns,
            )
            episode_row = dict(source_episodes.get(episode_index, {}))
            episode_row["episode_index"] = episode_index
            episode_row["length"] = frame_count
            output_episodes.append(episode_row)
            output_stats.append(
                {
                    "episode_index": episode_index,
                    "stats": _episode_stats(
                        cropped_table.to_pylist(),
                        info.get("features", {}),
                    ),
                }
            )
            total_output_frames += frame_count

            output_video = temporary_root / relative_video
            output_video.parent.mkdir(parents=True, exist_ok=True)
            video_filter = build_video_filter(
                alignment,
                duration_s=duration_s,
                output_fps=target_fps,
                frame_count=frame_count,
                action_start_s=float(overlap["action_start_s"]),
            )
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_video),
                "-vf",
                video_filter,
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_video),
            ]
            report_item = {
                    "episode_index": episode_index,
                    "offset_s": float(alignment["offset_s"]),
                    "alignment_model": str(alignment.get("alignment_model", "constant")),
                    "output_video": str(output_root / relative_video),
                    "source_action_frames": source_action_frames,
                    "source_video_frames": source_video_frames,
                    "source_video_fps": source_video_fps,
                    "action_start_frame": action_start_frame,
                    "action_stop_frame": action_stop_frame,
                    "dropped_action_start_frames": int(
                        overlap["source_action_start_index"]
                    ),
                    "dropped_action_end_frames": (
                        source_action_frames
                        - int(overlap["source_action_stop_index"])
                    ),
                    "camera_start_s": float(overlap["camera_start_s"]),
                    "action_frames": frame_count,
                    "action_time_source": action_time_source,
                    "source_action_duration_s": float(
                        overlap["source_action_duration_s"]
                    ),
                }
            job: dict[str, Any] = {
                "command": command,
                "episode_index": episode_index,
                "frame_count": frame_count,
                "output_video": output_video,
                "report_item": report_item,
            }
            if worker_count == 1:
                subprocess.run(command, check=True)
                finish_video(job)
            else:
                job["process"] = subprocess.Popen(command)
                pending_videos.append(job)
                if len(pending_videos) >= worker_count:
                    finish_video(pending_videos.pop(0))

        while pending_videos:
            finish_video(pending_videos.pop(0))

        _write_jsonl(temporary_root / "meta" / "episodes.jsonl", output_episodes)
        _write_jsonl(
            temporary_root / "meta" / "episodes_stats.jsonl",
            output_stats,
        )
        _update_dataset_info(
            temporary_root / "meta" / "info.json",
            output_fps=target_fps,
            total_frames=total_output_frames,
            total_episodes=len(output_episodes),
        )
        manifest_path = temporary_root / "sync" / "sync_application.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        temporary_root.replace(output_root)
    except BaseException:
        for job in locals().get("pending_videos", []):
            process = job.get("process")
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise
    return report


def _copy_non_video_dataset(root: Path, temporary_root: Path) -> None:
    data_root = root / "data"

    def ignore(source: str, names: list[str]) -> set[str]:
        source_path = Path(source)
        ignored = {"videos"} if source_path == root and "videos" in names else set()
        if source_path == root / "sync" and "gopro_originals" in names:
            ignored.add("gopro_originals")
        return ignored

    def copy_file(source: str, destination: str) -> str:
        source_path = Path(source)
        try:
            source_path.relative_to(data_root)
        except ValueError:
            return shutil.copy2(source, destination)
        try:
            os.link(source, destination)
            return destination
        except OSError:
            return shutil.copy2(source, destination)

    shutil.copytree(
        root,
        temporary_root,
        copy_function=copy_file,
        ignore=ignore,
    )


def _nearest_sample_indices(
    source_times_s: np.ndarray,
    target_times_s: np.ndarray,
) -> np.ndarray:
    right = np.searchsorted(source_times_s, target_times_s, side="left")
    right = np.clip(right, 0, len(source_times_s) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(target_times_s - source_times_s[left]) <= np.abs(
        source_times_s[right] - target_times_s
    )
    return np.where(choose_left, left, right)


def _interpolate_float_values(
    values: np.ndarray,
    *,
    source_times_s: np.ndarray,
    target_times_s: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    flat = array.reshape(len(source_times_s), -1)
    sampled = np.empty((len(target_times_s), flat.shape[1]), dtype=np.float64)
    for index in range(flat.shape[1]):
        sampled[:, index] = np.interp(
            target_times_s,
            source_times_s,
            flat[:, index],
        )
    return sampled.reshape((len(target_times_s),) + array.shape[1:])


def _slerp_quaternion_values(
    values: np.ndarray,
    *,
    source_times_s: np.ndarray,
    target_times_s: np.ndarray,
) -> np.ndarray:
    quaternions = np.asarray(values, dtype=np.float64)
    if quaternions.shape[-1] != 4:
        raise ValueError(f"Quaternion values need a trailing width of 4: {quaternions.shape}")
    if len(source_times_s) == 1:
        return np.repeat(quaternions, len(target_times_s), axis=0)

    right = np.searchsorted(source_times_s, target_times_s, side="right")
    right = np.clip(right, 1, len(source_times_s) - 1)
    left = right - 1
    denominator = source_times_s[right] - source_times_s[left]
    alpha = ((target_times_s - source_times_s[left]) / denominator).reshape(-1, 1, 1)
    q0 = quaternions[left].reshape(len(target_times_s), -1, 4)
    q1 = quaternions[right].reshape(len(target_times_s), -1, 4)
    q0_norm = np.linalg.norm(q0, axis=-1, keepdims=True)
    q1_norm = np.linalg.norm(q1, axis=-1, keepdims=True)
    valid_pair = (q0_norm >= 1e-12) & (q1_norm >= 1e-12)
    nearest = np.where(alpha <= 0.5, q0, q1)
    q0 = q0 / np.where(q0_norm < 1e-12, 1.0, q0_norm)
    q1 = q1 / np.where(q1_norm < 1e-12, 1.0, q1_norm)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-7
    safe_sin = np.where(near, 1.0, sin_theta)
    weight0 = np.sin((1.0 - alpha) * theta) / safe_sin
    weight1 = np.sin(alpha * theta) / safe_sin
    weight0 = np.where(near, 1.0 - alpha, weight0)
    weight1 = np.where(near, alpha, weight1)
    sampled = weight0 * q0 + weight1 * q1
    sampled_norm = np.linalg.norm(sampled, axis=-1, keepdims=True)
    sampled /= np.where(sampled_norm < 1e-12, 1.0, sampled_norm)
    sampled = np.where(valid_pair, sampled, nearest)
    return sampled.reshape((len(target_times_s),) + quaternions.shape[1:])


def _resample_arrow_column(
    name: str,
    column: pa.ChunkedArray,
    *,
    source_times_s: np.ndarray,
    target_times_s: np.ndarray,
) -> list[Any]:
    values = column.to_pylist()
    nearest = _nearest_sample_indices(source_times_s, target_times_s)

    leaf_type = column.type
    while pa.types.is_list(leaf_type) or pa.types.is_large_list(leaf_type):
        leaf_type = leaf_type.value_type
    if not pa.types.is_floating(leaf_type):
        return [values[index] for index in nearest]

    array = np.asarray(values, dtype=np.float64)
    if "pose" in name and array.ndim >= 2 and array.shape[-1] == 7:
        position = _interpolate_float_values(
            array[..., :3],
            source_times_s=source_times_s,
            target_times_s=target_times_s,
        )
        quaternion = _slerp_quaternion_values(
            array[..., 3:7],
            source_times_s=source_times_s,
            target_times_s=target_times_s,
        )
        return np.concatenate((position, quaternion), axis=-1).tolist()
    return _interpolate_float_values(
        array,
        source_times_s=source_times_s,
        target_times_s=target_times_s,
    ).tolist()


def _resample_episode_parquet(
    path: Path,
    *,
    source_times_s: np.ndarray,
    target_times_s: np.ndarray,
    episode_index: int,
    global_start: int,
    fps: float,
    local_time_origin_ns: int | None,
) -> pa.Table:
    table = pq.read_table(path)
    source_times_s = np.asarray(source_times_s, dtype=np.float64)
    target_times_s = np.asarray(target_times_s, dtype=np.float64)
    frame_count = len(target_times_s)
    output_arrays: list[pa.Array] = []
    for field, column in zip(table.schema, table.columns, strict=True):
        if field.name == "timestamp":
            values: Any = np.arange(frame_count, dtype=np.float64) / float(fps)
        elif field.name == "frame_index":
            values = range(frame_count)
        elif field.name == "episode_index":
            values = [episode_index] * frame_count
        elif field.name == "index":
            values = range(global_start, global_start + frame_count)
        elif (
            field.name == "observation.pico.local_timestamp_ns"
            and local_time_origin_ns is not None
        ):
            values = (
                int(local_time_origin_ns)
                + np.rint(target_times_s * 1e9).astype(np.int64)
            )
        else:
            values = _resample_arrow_column(
                field.name,
                column,
                source_times_s=source_times_s,
                target_times_s=target_times_s,
            )
        output_arrays.append(pa.array(values, type=field.type))

    output = pa.Table.from_arrays(output_arrays, schema=table.schema)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(output, temporary)
    temporary.replace(path)
    return output


def _crop_episode_parquet(
    path: Path,
    *,
    start_frame: int,
    stop_frame: int,
    episode_index: int,
    global_start: int,
    fps: float,
) -> pa.Table:
    table = pq.read_table(path).slice(start_frame, stop_frame - start_frame)
    frame_count = table.num_rows

    def replace(name: str, values: Any) -> None:
        nonlocal table
        if name not in table.column_names:
            return
        column_index = table.column_names.index(name)
        field = table.schema.field(column_index)
        table = table.set_column(
            column_index,
            field,
            pa.array(values, type=field.type),
        )

    replace("timestamp", np.arange(frame_count, dtype=np.float64) / float(fps))
    replace("frame_index", range(frame_count))
    replace("episode_index", [episode_index] * frame_count)
    replace("index", range(global_start, global_start + frame_count))
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)
    return table


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _update_dataset_info(
    info_path: Path,
    *,
    output_fps: float,
    total_frames: int,
    total_episodes: int,
) -> None:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["fps"] = float(output_fps)
    info["total_frames"] = int(total_frames)
    info["total_episodes"] = int(total_episodes)
    info["total_videos"] = int(total_episodes)
    info["splits"] = {"train": f"0:{total_episodes}"}
    features = info.get("features", {})
    for feature in features.values():
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            continue
        for metadata_key in ("video_info", "info"):
            metadata = feature.get(metadata_key)
            if isinstance(metadata, dict):
                metadata["video.fps"] = float(output_fps)
                metadata["video.codec"] = "h264"
                metadata["has_audio"] = False
    info_path.write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a new LeRobot dataset whose GoPro frames are on the Pico clock."
    )
    parser.add_argument("--root", required=True, type=Path, help="Raw LeRobot dataset root")
    parser.add_argument("--output-root", required=True, type=Path, help="New synchronized dataset root")
    parser.add_argument("--output-fps", type=float, default=None, help="Default: dataset/Pico FPS")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)

    report = apply_dataset_sync(
        args.root,
        output_root=args.output_root,
        output_fps=args.output_fps,
        overwrite=args.overwrite,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    print(
        f"wrote {report['output_root']}: {len(report['applied'])} synchronized episodes "
        f"at {report['output_fps']:.3f} FPS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
