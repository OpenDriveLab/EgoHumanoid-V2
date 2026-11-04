"""Track and smooth image-space cuff anchors across a recording."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np

from .geometry import robust_smooth_keypoints
from .image_geometry import (
    DEFAULT_GEOMETRY_CUFF_FOREARM_FRACTION,
    cuff_anchor_points,
    detect_arm_anchors,
)
from .video import iter_video_rgb


def interpolate_sampled_keypoints(
    sample_indices: np.ndarray, samples: np.ndarray, *, frame_count: int
) -> np.ndarray:
    """Linearly densify sparse per-frame keypoints without extrapolation drift."""
    indices = np.asarray(sample_indices, dtype=np.int64)
    values = np.asarray(samples, dtype=np.float64)
    if indices.ndim != 1 or len(indices) < 2:
        raise ValueError("sample_indices must contain at least two entries")
    if values.shape[0] != len(indices) or values.ndim < 2:
        raise ValueError("samples must have the same leading size as sample_indices")
    if frame_count < 2 or indices[0] != 0 or indices[-1] != frame_count - 1:
        raise ValueError("samples must include the first and final video frames")
    if np.any(np.diff(indices) <= 0) or not np.isfinite(values).all():
        raise ValueError("sample indices must increase and sample values must be finite")
    dense_indices = np.arange(frame_count, dtype=np.float64)
    flattened = values.reshape(len(indices), -1)
    dense = np.empty((frame_count, flattened.shape[1]), dtype=np.float64)
    for column in range(flattened.shape[1]):
        dense[:, column] = np.interp(
            dense_indices, indices.astype(np.float64), flattened[:, column]
        )
    return dense.reshape((frame_count,) + values.shape[1:])


def repair_isolated_keypoint_outliers(
    sample_indices: np.ndarray, samples: np.ndarray, *, maximum_residual_px: float
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Replace isolated per-side detections far from their two neighbors."""
    indices = np.asarray(sample_indices, dtype=np.int64)
    values = np.asarray(samples, dtype=np.float64)
    if indices.ndim != 1 or values.shape != (len(indices), 2, 2):
        raise ValueError("samples must have shape (sample_count, 2, 2)")
    if len(indices) < 3 or np.any(np.diff(indices) <= 0):
        raise ValueError("at least three increasing sample indices are required")
    if maximum_residual_px <= 0.0 or not np.isfinite(values).all():
        raise ValueError("outlier threshold and samples must be finite and positive")
    repaired = values.copy()
    repairs: list[dict[str, int]] = []
    for index in range(1, len(indices) - 1):
        interval = float(indices[index + 1] - indices[index - 1])
        alpha = float(indices[index] - indices[index - 1]) / interval
        expected = values[index - 1] + alpha * (values[index + 1] - values[index - 1])
        residuals = np.linalg.norm(values[index] - expected, axis=1)
        previous_jumps = np.linalg.norm(values[index] - values[index - 1], axis=1)
        next_jumps = np.linalg.norm(values[index] - values[index + 1], axis=1)
        neighbor_spans = np.linalg.norm(values[index + 1] - values[index - 1], axis=1)
        isolated = (
            (residuals > maximum_residual_px)
            & (previous_jumps > maximum_residual_px)
            & (next_jumps > maximum_residual_px)
            & (neighbor_spans <= maximum_residual_px)
        )
        for side in np.flatnonzero(isolated):
            repaired[index, side] = expected[side]
            repairs.append(
                {
                    "sample_index": index,
                    "frame_index": int(indices[index]),
                    "side": int(side),
                }
            )
    return (repaired, repairs)


def limit_keypoint_speed(
    sample_indices: np.ndarray,
    samples: np.ndarray,
    *,
    fps: float,
    frame_diagonal_px: float,
    maximum_speed_fraction_per_second: float = 0.5,
    maximum_repair_seconds: float = 1.0,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Interpolate target-switch blocks bounded by implausible visual jumps."""
    indices = np.asarray(sample_indices, dtype=np.int64)
    values = np.asarray(samples, dtype=np.float64)
    if indices.ndim != 1 or values.shape != (len(indices), 2, 2):
        raise ValueError("samples must have shape (sample_count, 2, 2)")
    if len(indices) < 3 or np.any(np.diff(indices) <= 0):
        raise ValueError("at least three increasing sample indices are required")
    if not np.isfinite(values).all() or not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("samples and fps must be finite and fps positive")
    if (
        frame_diagonal_px <= 0.0
        or maximum_speed_fraction_per_second <= 0.0
        or maximum_repair_seconds <= 0.0
    ):
        raise ValueError("speed scale and frame diagonal must be positive")
    repaired = values.copy()
    repairs: list[dict[str, int]] = []
    for side in range(2):
        elapsed = np.diff(indices) / float(fps)
        limits = maximum_speed_fraction_per_second * frame_diagonal_px * elapsed
        jumps = np.linalg.norm(np.diff(values[:, side], axis=0), axis=1) > limits
        jump_edges = np.flatnonzero(jumps)
        for left_edge, right_edge in pairwise(jump_edges):
            if right_edge <= left_edge:
                continue
            start = left_edge
            stop = right_edge + 1
            endpoint_elapsed = (indices[stop] - indices[start]) / float(fps)
            if endpoint_elapsed > maximum_repair_seconds:
                continue
            endpoint_limit = (
                maximum_speed_fraction_per_second * frame_diagonal_px * endpoint_elapsed
            )
            if np.linalg.norm(values[stop, side] - values[start, side]) > endpoint_limit:
                continue
            for sample_index in range(start + 1, stop):
                alpha = (indices[sample_index] - indices[start]) / (indices[stop] - indices[start])
                repaired[sample_index, side] = values[start, side] + alpha * (
                    values[stop, side] - values[start, side]
                )
                repairs.append(
                    {
                        "sample_index": sample_index,
                        "frame_index": int(indices[sample_index]),
                        "side": side,
                    }
                )
    return (repaired, repairs)


def smoothing_window_frames(fps: float, *, seconds: float = 1.0) -> int:
    """Convert a task-independent temporal duration to an odd frame window."""
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("smoothing seconds must be finite and positive")
    frames = max(3, round(fps * seconds))
    return frames if frames % 2 else frames + 1


def detect_cuff_trajectory(
    video_path: Path,
    *,
    frame_count: int,
    fps: float,
    stride: int,
    smoothing_window: int,
    forearm_fraction: float = DEFAULT_GEOMETRY_CUFF_FOREARM_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, int]]]:
    if stride < 1:
        raise ValueError("cuff stride must be positive")
    requested = set(range(0, frame_count, stride))
    requested.add(frame_count - 1)
    indices: list[int] = []
    samples: list[np.ndarray] = []
    frame_diagonal: float | None = None
    for frame_index, rgb in enumerate(iter_video_rgb(video_path)):
        if frame_diagonal is None:
            frame_diagonal = float(np.hypot(*rgb.shape[:2]))
        if frame_index in requested:
            anchors = detect_arm_anchors(rgb, right_minimum_y_fraction=0.45)
            indices.append(frame_index)
            samples.append(
                cuff_anchor_points(anchors, forearm_fraction=forearm_fraction).astype(np.float64)
            )
    if not indices or indices[-1] != frame_count - 1:
        raise RuntimeError("video decoding ended before its declared final frame")
    sample_indices = np.asarray(indices, dtype=np.int64)
    sample_values = np.stack(samples)
    sample_values, outlier_repairs = repair_isolated_keypoint_outliers(
        sample_indices, sample_values, maximum_residual_px=0.1 * float(frame_diagonal)
    )
    sample_values, speed_repairs = limit_keypoint_speed(
        sample_indices, sample_values, fps=fps, frame_diagonal_px=float(frame_diagonal)
    )
    outlier_repairs.extend(speed_repairs)
    dense = interpolate_sampled_keypoints(sample_indices, sample_values, frame_count=frame_count)
    smoothed = robust_smooth_keypoints(
        dense,
        window_length=smoothing_window,
        polynomial_order=2,
        outlier_sigma=3.5,
        minimum_outlier_distance=2.0,
    )
    return (sample_indices, sample_values, dense, smoothed, outlier_repairs)
