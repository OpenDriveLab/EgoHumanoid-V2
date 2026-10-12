from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

from human.devices.gopro import GOPRO_VIDEO_PATH
from human.devices.pico import CHEST_JOINT_INDEX, HEAD_JOINT_INDEX


RESAMPLE_HZ = 120.0
DEFAULT_WINDOW_S = 3.0
DEFAULT_STEP_S = 1.5
MOTION_SMOOTH_SIGMA_S = 0.035
MOTION_FLOOR_QUANTILE = 0.20
MOTION_POWER = 0.50
MAX_CONSENSUS_WINDOWS = 3
CONSENSUS_CLUSTER_WIDTH_S = 0.100
CONSENSUS_MIN_CONFIDENCE = 0.65
CONSENSUS_MIN_PEAK_MARGIN = 0.010


def select_consensus_segments(
    segments: list[dict],
    *,
    reference_offset_s: float,
    prior_offset_s: float | None = None,
    limit: int = MAX_CONSENSUS_WINDOWS,
    cluster_width_s: float = CONSENSUS_CLUSTER_WIDTH_S,
) -> list[dict]:
    """Choose repeated offset evidence instead of isolated high-energy peaks.

    The session prior only breaks ties between equally supported/evidenced
    clusters. It cannot make a singleton peak into consensus.
    """

    eligible = [
        segment
        for segment in segments
        if not bool(segment.get("search_boundary_hit", False))
        and np.isfinite(float(segment.get("offset_s", float("nan"))))
        and float(segment.get("confidence", 0.0)) >= CONSENSUS_MIN_CONFIDENCE
        and float(segment.get("peak_margin", 0.0)) >= CONSENSUS_MIN_PEAK_MARGIN
    ]
    best: tuple[tuple[float, ...], tuple[dict, ...]] | None = None
    maximum_size = min(max(0, int(limit)), len(eligible))
    for size in range(2, maximum_size + 1):
        for candidate in combinations(eligible, size):
            offsets = np.asarray(
                [float(segment["offset_s"]) for segment in candidate],
                dtype=np.float64,
            )
            if float(np.max(offsets) - np.min(offsets)) > float(cluster_width_s):
                continue
            median_offset_s = float(np.median(offsets))
            evidence = sum(
                float(segment.get("confidence", 0.0))
                * float(segment.get("peak_margin", 0.0))
                * np.sqrt(max(0.0, float(segment.get("motion_energy", 0.0))))
                for segment in candidate
            )
            prior_distance = (
                abs(median_offset_s - float(prior_offset_s))
                if prior_offset_s is not None
                else 0.0
            )
            score = (
                float(size),
                float(evidence),
                -float(prior_distance),
                -abs(median_offset_s - float(reference_offset_s)),
            )
            if best is None or score > best[0]:
                best = (score, candidate)
    if best is not None:
        return list(best[1])

    fallback = [
        segment
        for segment in segments
        if not bool(segment.get("search_boundary_hit", False))
    ]
    if not fallback:
        fallback = list(segments)
    return sorted(
        fallback,
        key=lambda segment: float(segment.get("motion_energy", 0.0)),
        reverse=True,
    )[: max(0, int(limit))]


def load_lerobot_joint_angvel(
    root: str | Path,
    *,
    episode_index: int,
    joint_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one Pico joint's angular-velocity magnitude from an episode."""

    root = Path(root)
    parquet_path = root / _episode_data_path(episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(f"LeRobot episode not found: {parquet_path}")

    table = pq.read_table(
        parquet_path,
        columns=[
            "observation.pico.body_velocity",
            "observation.pico.body_joint_timestamp_ns",
            "observation.pico.local_timestamp_ns",
        ],
    )
    data = table.to_pydict()
    velocity = np.asarray(data["observation.pico.body_velocity"], dtype=np.float64)
    if velocity.ndim != 3 or velocity.shape[1:] != (24, 6):
        raise ValueError(
            "observation.pico.body_velocity must have shape (frames, 24, 6); "
            "re-record with the current collector"
        )

    joint_timestamps = np.asarray(
        data["observation.pico.body_joint_timestamp_ns"],
        dtype=np.float64,
    )
    local_timestamps = np.asarray(data["observation.pico.local_timestamp_ns"], dtype=np.float64)
    if not 0 <= joint_index < velocity.shape[1]:
        raise ValueError(f"Pico joint index is out of range: {joint_index}")
    timestamps_ns = _select_pico_timestamps_ns(
        joint_timestamps,
        local_timestamps,
        joint_index=joint_index,
    )
    angular_velocity = velocity[:, joint_index, 3:6]
    magnitude = np.linalg.norm(angular_velocity, axis=1)
    timestamps_s = timestamps_ns * 1e-9
    return timestamps_s - timestamps_s[0], magnitude


def load_lerobot_joint_pose_angvel(
    root: str | Path,
    *,
    episode_index: int,
    joint_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive angular-speed magnitude from consecutive Pico joint poses.

    Pico SDK velocity has proved noisy for cross-device synchronization. The
    quaternion delta is the same physical quantity measured by the GoPro gyro,
    and its sample belongs at the midpoint of the two source poses.
    """

    root = Path(root)
    parquet_path = root / _episode_data_path(episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(f"LeRobot episode not found: {parquet_path}")

    table = pq.read_table(
        parquet_path,
        columns=[
            "observation.pico.body_pose",
            "observation.pico.body_joint_timestamp_ns",
            "observation.pico.local_timestamp_ns",
        ],
    )
    data = table.to_pydict()
    body_pose = np.asarray(data["observation.pico.body_pose"], dtype=np.float64)
    if body_pose.ndim != 3 or body_pose.shape[1:] != (24, 7):
        raise ValueError(
            "observation.pico.body_pose must have shape (frames, 24, 7); "
            "re-record with the current collector"
        )
    if not 0 <= joint_index < body_pose.shape[1]:
        raise ValueError(f"Pico joint index is out of range: {joint_index}")

    joint_timestamps = np.asarray(
        data["observation.pico.body_joint_timestamp_ns"],
        dtype=np.float64,
    )
    local_timestamps = np.asarray(
        data["observation.pico.local_timestamp_ns"],
        dtype=np.float64,
    )
    timestamps_ns = _select_pico_timestamps_ns(
        joint_timestamps,
        local_timestamps,
        joint_index=joint_index,
    )
    timestamps_s = timestamps_ns * 1e-9
    timestamps_s -= timestamps_s[0]

    quaternions = body_pose[:, joint_index, 3:7]
    norms = np.linalg.norm(quaternions, axis=1)
    valid_quaternions = np.all(np.isfinite(quaternions), axis=1) & (norms > 1e-8)
    normalized = np.zeros_like(quaternions)
    normalized[valid_quaternions] = (
        quaternions[valid_quaternions] / norms[valid_quaternions, np.newaxis]
    )
    delta_s = np.diff(timestamps_s)
    valid_pairs = (
        valid_quaternions[:-1]
        & valid_quaternions[1:]
        & np.isfinite(delta_s)
        & (delta_s > 0.0)
    )
    if np.count_nonzero(valid_pairs) < 2:
        raise ValueError("Pico joint pose has fewer than two valid quaternion deltas")

    dot = np.sum(normalized[:-1] * normalized[1:], axis=1)
    # q and -q encode the same rotation, so use the shortest rotation arc.
    delta_angle_rad = 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    midpoint_s = 0.5 * (timestamps_s[:-1] + timestamps_s[1:])
    angular_speed = np.zeros_like(delta_angle_rad)
    np.divide(delta_angle_rad, delta_s, out=angular_speed, where=valid_pairs)
    return midpoint_s[valid_pairs], angular_speed[valid_pairs]


def load_lerobot_head_angvel(
    root: str | Path,
    *,
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Pico head angular-velocity magnitude from a LeRobot episode."""

    return load_lerobot_joint_angvel(
        root,
        episode_index=episode_index,
        joint_index=HEAD_JOINT_INDEX,
    )


def load_lerobot_chest_angvel(
    root: str | Path,
    *,
    episode_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Pico upper-chest angular-velocity magnitude from an episode."""

    return load_lerobot_joint_angvel(
        root,
        episode_index=episode_index,
        joint_index=CHEST_JOINT_INDEX,
    )


def load_gopro_gyro(video: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load GoPro gyro magnitude from GPMF telemetry."""

    try:
        import telemetry_parser
    except ImportError as exc:
        raise RuntimeError(
            "GoPro alignment requires telemetry-parser. Install project dependencies again."
        ) from exc

    try:
        parser = telemetry_parser.Parser(str(video))
        imu = parser.normalized_imu()
    except OSError as exc:
        raise ValueError(f"GoPro telemetry cannot be read from {video}") from exc
    if not imu:
        raise ValueError(f"GoPro video has no IMU telemetry: {video}")

    timestamps_ms: list[float] = []
    gyro_samples: list[np.ndarray] = []
    for sample in imu:
        try:
            timestamp_ms = float(sample["timestamp_ms"])
            gyro = np.asarray(sample.get("gyro"), dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not np.isfinite(timestamp_ms)
            or gyro.shape != (3,)
            or not np.all(np.isfinite(gyro))
        ):
            continue
        timestamps_ms.append(timestamp_ms)
        gyro_samples.append(gyro)
    if len(timestamps_ms) < 2:
        raise ValueError(f"GoPro video has fewer than two valid gyro samples: {video}")

    timestamps_s = np.asarray(timestamps_ms, dtype=np.float64) * 1e-3
    gyro = np.asarray(gyro_samples, dtype=np.float64)
    order = np.argsort(timestamps_s, kind="stable")
    timestamps_s = timestamps_s[order]
    gyro = gyro[order]
    unique = np.concatenate(([True], np.diff(timestamps_s) > 0.0))
    timestamps_s = timestamps_s[unique]
    gyro = gyro[unique]
    if len(timestamps_s) < 2:
        raise ValueError(f"GoPro video has fewer than two distinct gyro timestamps: {video}")
    magnitude = np.linalg.norm(gyro, axis=1)
    return timestamps_s - timestamps_s[0], magnitude


def xcorr_offset(
    ts_a: np.ndarray,
    a: np.ndarray,
    ts_b: np.ndarray,
    b: np.ndarray,
    *,
    max_offset_s: float = 5.0,
    hz: float = RESAMPLE_HZ,
) -> tuple[float, float]:
    """Return the offset that should be added to b's timestamps to align with a."""

    diagnostics = xcorr_offset_diagnostics(
        ts_a,
        a,
        ts_b,
        b,
        max_offset_s=max_offset_s,
        hz=hz,
    )
    return diagnostics["offset_s"], diagnostics["confidence"]


def motion_correlation_curve(
    ts_a: np.ndarray,
    a: np.ndarray,
    ts_b: np.ndarray,
    b: np.ndarray,
    *,
    max_offset_s: float = 5.0,
    hz: float = RESAMPLE_HZ,
    center_offset_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the robust lag curve used by the automatic aligner."""

    ts_a = np.asarray(ts_a, dtype=np.float64)
    ts_b = np.asarray(ts_b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    _validate_timeseries(ts_a, a, "a")
    _validate_timeseries(ts_b, b, "b")
    _, prepared_a, prepared_b = _prepare_motion_pair(ts_a, a, ts_b, b, hz=hz)
    return _overlap_pearson_curve(
        prepared_a,
        prepared_b,
        max_offset_s=max_offset_s,
        hz=hz,
        center_offset_s=center_offset_s,
    )


def xcorr_offset_diagnostics(
    ts_a: np.ndarray,
    a: np.ndarray,
    ts_b: np.ndarray,
    b: np.ndarray,
    *,
    max_offset_s: float = 5.0,
    hz: float = RESAMPLE_HZ,
    center_offset_s: float = 0.0,
) -> dict:
    """Return cross-correlation alignment and quality diagnostics.

    The offset is the value to add to b's timestamps to align them to a.
    """

    ts_a = np.asarray(ts_a, dtype=np.float64)
    ts_b = np.asarray(ts_b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    _validate_timeseries(ts_a, a, "a")
    _validate_timeseries(ts_b, b, "b")

    grid, resampled_a, resampled_b = _prepare_motion_pair(ts_a, a, ts_b, b, hz=hz)
    lags_window, corr_norm = _overlap_pearson_curve(
        resampled_a,
        resampled_b,
        max_offset_s=max_offset_s,
        hz=hz,
        center_offset_s=center_offset_s,
    )
    diagnostics = _correlation_diagnostics(
        lags_window,
        corr_norm,
        hz=hz,
        center_offset_s=center_offset_s,
        max_offset_s=max_offset_s,
    )
    peak = int(diagnostics.pop("peak_index"))
    peak_value = float(corr_norm[peak])
    second_value = _second_peak_value(
        corr_norm,
        peak,
        exclusion=max(1, int(round(0.10 * hz))),
    )
    peak_ratio = float(abs(peak_value) / (abs(second_value) + 1e-12))
    peak_margin = float(peak_value - second_value)
    half_width_s = _peak_half_width_s(corr_norm, peak, hz)
    confidence = float(diagnostics["confidence"])
    estimated_error_s = max(
        0.5 / float(hz),
        min(0.5, half_width_s * (1.0 - min(max(confidence, 0.0), 0.99))),
    )
    return {
        **diagnostics,
        "offset_frames": float(diagnostics["offset_s"] * hz),
        "peak_ratio": peak_ratio,
        "peak_margin": peak_margin,
        "estimated_offset_error_s": float(estimated_error_s),
        "peak_half_width_s": float(half_width_s),
        "peak_fwhm_s": float(half_width_s),
        "resample_hz": float(hz),
        "overlap_s": float(grid[-1] - grid[0]),
        "correlation_normalization": "overlap_pearson",
        "motion_preprocessing": {
            "gaussian_sigma_s": MOTION_SMOOTH_SIGMA_S,
            "floor_quantile": MOTION_FLOOR_QUANTILE,
            "power": MOTION_POWER,
            "zero_phase": True,
        },
    }


def estimate_alignment_quality(
    ts_pico: np.ndarray,
    mag_pico: np.ndarray,
    ts_gopro: np.ndarray,
    mag_gopro: np.ndarray,
    *,
    max_offset_s: float = 1.0,
    hz: float = RESAMPLE_HZ,
    window_s: float = DEFAULT_WINDOW_S,
    step_s: float | None = None,
    center_offset_s: float = 0.0,
) -> dict:
    """Estimate offset, drift, and practical timestamp error for an episode."""

    global_diag = xcorr_offset_diagnostics(
        ts_pico,
        mag_pico,
        ts_gopro,
        mag_gopro,
        max_offset_s=max_offset_s,
        hz=hz,
        center_offset_s=center_offset_s,
    )
    pico_period_s = _median_period_s(ts_pico)
    gopro_imu_period_s = _median_period_s(ts_gopro)
    segments = _segment_offset_diagnostics(
        ts_pico,
        mag_pico,
        ts_gopro,
        mag_gopro,
        max_offset_s=max_offset_s,
        hz=hz,
        window_s=window_s,
        step_s=step_s,
        center_offset_s=center_offset_s,
    )
    selected_segments = select_consensus_segments(
        segments,
        reference_offset_s=float(global_diag["offset_s"]),
    )
    for segment in selected_segments:
        segment["selected_for_consensus"] = True
    selected_offsets = np.asarray(
        [segment["offset_s"] for segment in selected_segments],
        dtype=np.float64,
    )
    if len(selected_offsets):
        window_median_s = float(np.median(selected_offsets))
        window_mad_s = float(np.median(np.abs(selected_offsets - window_median_s)))
        global_window_delta_s = abs(float(global_diag["offset_s"]) - window_median_s)
        consensus_50ms = int(
            np.sum(np.abs(selected_offsets - float(global_diag["offset_s"])) <= 0.050)
        )
        consensus_100ms = int(
            np.sum(np.abs(selected_offsets - float(global_diag["offset_s"])) <= 0.100)
        )
    else:
        window_median_s = float(global_diag["offset_s"])
        window_mad_s = float(max_offset_s)
        global_window_delta_s = float(max_offset_s)
        consensus_50ms = 0
        consensus_100ms = 0
    result = {
        **global_diag,
        "alignment_model": "constant",
        "affine_offset_s": global_diag["offset_s"],
        "drift_s_per_s": 0.0,
        "segment_offsets": segments,
        "pico_sample_period_s": pico_period_s,
        "gopro_imu_sample_period_s": gopro_imu_period_s,
        "window_count": int(len(selected_segments)),
        "window_offset_median_s": window_median_s,
        "window_offset_mad_s": window_mad_s,
        "global_window_delta_s": global_window_delta_s,
        "window_consensus_50ms": consensus_50ms,
        "window_consensus_100ms": consensus_100ms,
        "consensus_window_limit": MAX_CONSENSUS_WINDOWS,
    }
    fit = _fit_segment_drift(segments)
    if fit is not None:
        result.update(fit)
        # Natural-motion windows can choose different broad peaks. Keep the
        # fitted slope for diagnosis, but never retime video with it by default.
        result["drift_diagnostic_only"] = True

    sampling_floor_s = max(
        0.5 / float(hz),
        0.5 * pico_period_s,
        0.5 * gopro_imu_period_s,
    )
    if len(selected_offsets) >= 2:
        stability_error_s = max(
            sampling_floor_s,
            global_window_delta_s,
            1.4826 * window_mad_s,
        )
        error_basis = "high_motion_window_stability"
    else:
        stability_error_s = max(
            sampling_floor_s,
            float(global_diag["estimated_offset_error_s"]),
        )
        error_basis = "correlation_peak_width_fallback"
    result["sampling_floor_s"] = float(sampling_floor_s)
    result["estimated_sync_error_s"] = float(stability_error_s)
    result["estimated_sync_error_basis"] = error_basis
    result["estimated_sync_error_is_ground_truth"] = False
    return result


def apply_session_consensus_prior(payload: dict, *, prior_offset_s: float) -> dict:
    """Re-evaluate window consensus with a leave-one-out session tie-breaker."""

    if not isinstance(payload.get("segment_offsets"), list):
        return dict(payload)
    result = dict(payload)
    segments = [dict(segment) for segment in payload["segment_offsets"]]
    for segment in segments:
        segment.pop("selected_for_consensus", None)
    selected_segments = select_consensus_segments(
        segments,
        reference_offset_s=float(payload["offset_s"]),
        prior_offset_s=float(prior_offset_s),
    )
    selected_ids = {id(segment) for segment in selected_segments}
    for segment in segments:
        if id(segment) in selected_ids:
            segment["selected_for_consensus"] = True
    result["segment_offsets"] = segments
    selected_offsets = np.asarray(
        [float(segment["offset_s"]) for segment in selected_segments],
        dtype=np.float64,
    )
    reference_offset_s = float(payload["offset_s"])
    if len(selected_offsets):
        window_median_s = float(np.median(selected_offsets))
        window_mad_s = float(np.median(np.abs(selected_offsets - window_median_s)))
        global_window_delta_s = abs(reference_offset_s - window_median_s)
        consensus_50ms = int(
            np.sum(np.abs(selected_offsets - reference_offset_s) <= 0.050)
        )
        consensus_100ms = int(
            np.sum(np.abs(selected_offsets - reference_offset_s) <= 0.100)
        )
    else:
        window_median_s = reference_offset_s
        window_mad_s = 1.0
        global_window_delta_s = 1.0
        consensus_50ms = 0
        consensus_100ms = 0
    result.update(
        {
            "window_count": int(len(selected_segments)),
            "window_offset_median_s": window_median_s,
            "window_offset_mad_s": window_mad_s,
            "global_window_delta_s": global_window_delta_s,
            "window_consensus_50ms": consensus_50ms,
            "window_consensus_100ms": consensus_100ms,
        }
    )
    sampling_floor_s = float(payload.get("sampling_floor_s", 0.0))
    if len(selected_offsets) >= 2:
        result["estimated_sync_error_s"] = float(
            max(
                sampling_floor_s,
                global_window_delta_s,
                1.4826 * window_mad_s,
            )
        )
        result["estimated_sync_error_basis"] = "session_prior_window_stability"
    else:
        result["estimated_sync_error_s"] = float(
            max(
                sampling_floor_s,
                float(payload.get("estimated_offset_error_s", 1.0)),
            )
        )
        result["estimated_sync_error_basis"] = "correlation_peak_width_fallback"
    return result


def align_lerobot_episode(
    *,
    root: str | Path,
    episode_index: int,
    video: str | Path | None = None,
    max_offset_s: float = 1.0,
    resample_hz: float = RESAMPLE_HZ,
    window_s: float = DEFAULT_WINDOW_S,
    step_s: float | None = None,
    use_capture_prior: bool = True,
    drift: bool = False,
    pico_joint: str = "chest",
    pico_motion_source: str = "pose_delta",
    session_offset_prior_s: float | None = None,
    session_prior_tolerance_s: float = 0.050,
) -> dict:
    root = Path(root)
    video_path = (
        Path(video)
        if video is not None
        else _resolve_alignment_video_path(root, episode_index=episode_index)
    )
    prior = load_capture_timing_prior(root, episode_index=episode_index) if use_capture_prior else None
    ts_gopro, mag_gopro = load_gopro_gyro(video_path)
    capture_prior_offset_s = float(prior["capture_prior_offset_s"]) if prior is not None else 0.0
    ts_gopro_on_pico_axis = ts_gopro + capture_prior_offset_s
    if pico_joint not in {"auto", "chest", "head"}:
        raise ValueError("pico_joint must be auto, chest, or head")
    if pico_motion_source not in {"pose_delta", "sdk_velocity"}:
        raise ValueError("pico_motion_source must be pose_delta or sdk_velocity")
    candidate_names = ("chest", "head") if pico_joint == "auto" else (pico_joint,)
    candidates: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    failures: dict[str, str] = {}
    for candidate_name in candidate_names:
        try:
            joint_index = (
                CHEST_JOINT_INDEX if candidate_name == "chest" else HEAD_JOINT_INDEX
            )
            if pico_motion_source == "pose_delta":
                ts_pico_candidate, mag_pico_candidate = load_lerobot_joint_pose_angvel(
                    root,
                    episode_index=episode_index,
                    joint_index=joint_index,
                )
            else:
                loader = (
                    load_lerobot_chest_angvel
                    if candidate_name == "chest"
                    else load_lerobot_head_angvel
                )
                ts_pico_candidate, mag_pico_candidate = loader(
                    root,
                    episode_index=episode_index,
                )
            candidate_quality = estimate_alignment_quality(
                ts_pico_candidate,
                mag_pico_candidate,
                ts_gopro_on_pico_axis,
                mag_gopro,
                max_offset_s=max_offset_s,
                hz=resample_hz,
                window_s=window_s,
                step_s=step_s,
                center_offset_s=0.0,
            )
            candidates[candidate_name] = (
                ts_pico_candidate,
                mag_pico_candidate,
                _restore_camera_time_offsets(
                    candidate_quality,
                    capture_prior_offset_s,
                ),
            )
        except ValueError as exc:
            failures[candidate_name] = str(exc)
    if not candidates:
        detail = "; ".join(f"{name}: {error}" for name, error in failures.items())
        raise ValueError(f"No usable Pico motion signal for alignment: {detail}")
    selected_name = max(
        candidates,
        key=lambda name: _alignment_candidate_score(
            candidates[name][2],
            preferred_offset_s=session_offset_prior_s,
            preferred_tolerance_s=session_prior_tolerance_s,
        ),
    )
    ts_pico, mag_pico, quality = candidates[selected_name]
    offset = quality["offset_s"]
    confidence = quality["confidence"]
    result = {
        "episode_index": int(episode_index),
        "video_path": str(video_path),
        "offset_s": offset,
        "confidence": confidence,
        "pico_duration_s": float(ts_pico[-1]),
        "gopro_duration_s": float(ts_gopro[-1]),
        "pico_samples": int(len(ts_pico)),
        "gopro_imu_samples": int(len(ts_gopro)),
        "pico_motion_joint": selected_name,
        "pico_motion_joint_index": (
            CHEST_JOINT_INDEX if selected_name == "chest" else HEAD_JOINT_INDEX
        ),
        "pico_motion_source": pico_motion_source,
        "pico_motion_candidates": {
            name: {
                "confidence": float(candidate[2]["confidence"]),
                "peak_ratio": float(candidate[2].get("peak_ratio", 0.0)),
                "estimated_sync_error_s": float(
                    candidate[2]["estimated_sync_error_s"]
                ),
                "offset_s": float(candidate[2]["offset_s"]),
            }
            for name, candidate in candidates.items()
        },
    }
    if failures:
        result["pico_motion_candidate_failures"] = failures
    result.update(quality)
    if prior is not None:
        result.update(prior)
        result["capture_prior_delta_s"] = float(quality["gyro_residual_offset_s"])
    frame_error = _frame_pairing_error_s(
        root,
        quality,
        episode_index=episode_index,
        video_path=video_path,
    )
    if frame_error is not None:
        result.update(frame_error)
    if drift:
        legacy = _estimate_legacy_drift(
            ts_pico,
            mag_pico,
            ts_gopro_on_pico_axis,
            mag_gopro,
            max_offset_s,
        )
        result.update(_restore_camera_time_offsets(legacy, capture_prior_offset_s))
    return result


def _alignment_candidate_score(
    quality: dict,
    *,
    preferred_offset_s: float | None = None,
    preferred_tolerance_s: float = 0.050,
) -> tuple[bool, bool, int, float, float, float]:
    prior_consistent = (
        True
        if preferred_offset_s is None
        else abs(float(quality.get("offset_s", float("inf"))) - preferred_offset_s)
        <= float(preferred_tolerance_s)
    )
    return (
        not bool(quality.get("search_boundary_hit", False)),
        prior_consistent,
        _alignment_candidate_evidence_tier(quality),
        float(quality.get("confidence", 0.0)),
        float(quality.get("peak_ratio", 0.0)),
        -float(quality.get("estimated_sync_error_s", float("inf"))),
    )


def _alignment_candidate_evidence_tier(quality: dict) -> int:
    """Rank pass/warn/fail window evidence before scalar confidence."""

    try:
        confidence = float(quality["confidence"])
        peak_margin = float(quality["peak_margin"])
        error_s = float(quality["estimated_sync_error_s"])
        window_count = int(quality["window_count"])
        window_mad_s = float(quality["window_offset_mad_s"])
        global_window_delta_s = float(quality["global_window_delta_s"])
        consensus_50ms = int(quality["window_consensus_50ms"])
        consensus_100ms = int(quality["window_consensus_100ms"])
    except (KeyError, TypeError, ValueError):
        return 0
    if (
        window_count >= 2
        and confidence >= 0.75
        and peak_margin >= 0.015
        and error_s <= 0.10
        and window_mad_s <= 0.05
        and global_window_delta_s <= 0.05
        and consensus_50ms >= 2
    ):
        return 2
    if (
        window_count >= 2
        and confidence >= 0.65
        and peak_margin >= 0.010
        and error_s <= 0.20
        and window_mad_s <= 0.10
        and global_window_delta_s <= 0.10
        and consensus_100ms >= 2
    ):
        return 1
    return 0


def _restore_camera_time_offsets(diagnostics: dict, capture_prior_offset_s: float) -> dict:
    """Express residual correlation offsets on the camera's native time axis."""

    result = dict(diagnostics)
    shift = float(capture_prior_offset_s)
    if "offset_s" in result:
        residual = float(result["offset_s"])
        result["gyro_residual_offset_s"] = residual
        result["offset_s"] = residual + shift
        if "resample_hz" in result:
            result["gyro_residual_offset_frames"] = residual * float(result["resample_hz"])
            result["offset_frames"] = float(result["offset_s"]) * float(
                result["resample_hz"]
            )
    if "peak_lag_s" in result:
        result["gyro_residual_peak_lag_s"] = float(result["peak_lag_s"])
        result["peak_lag_s"] = float(result["peak_lag_s"]) + shift
    if "affine_offset_s" in result:
        result["gyro_residual_affine_offset_s"] = float(result["affine_offset_s"])
        result["affine_offset_s"] = float(result["affine_offset_s"]) + shift
    if "window_offset_median_s" in result:
        residual = float(result["window_offset_median_s"])
        result["gyro_residual_window_offset_median_s"] = residual
        result["window_offset_median_s"] = residual + shift
    if "segment_offsets" in result:
        transformed_segments = []
        for segment in result["segment_offsets"]:
            transformed = dict(segment)
            if "offset_s" in transformed:
                residual = float(transformed["offset_s"])
                transformed["gyro_residual_offset_s"] = residual
                transformed["offset_s"] = residual + shift
            transformed_segments.append(transformed)
        result["segment_offsets"] = transformed_segments
    for key in ("legacy_offset_start_s", "legacy_offset_end_s"):
        if key in result:
            result[f"gyro_residual_{key}"] = float(result[key])
            result[key] = float(result[key]) + shift
    result["capture_time_axis_shift_s"] = shift
    return result


def default_alignment_path(root: str | Path, *, episode_index: int) -> Path:
    return Path(root) / "sync" / f"episode_{episode_index:06d}.gopro_alignment.json"


def capture_timing_path(root: str | Path, *, episode_index: int) -> Path:
    return Path(root) / "sync" / f"episode_{episode_index:06d}.gopro_capture.json"


def _resolve_alignment_video_path(root: Path, *, episode_index: int) -> Path:
    """Prefer the preserved GoPro original because transcoding removes GPMF."""

    timing_path = capture_timing_path(root, episode_index=episode_index)
    timing: dict = {}
    if timing_path.exists():
        try:
            payload = json.loads(timing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            timing = payload

    configured = timing.get("original_video_path")
    candidates: list[Path] = []
    if isinstance(configured, str) and configured.strip():
        configured_path = Path(configured).expanduser()
        if configured_path.is_absolute():
            candidates.append(configured_path)
        else:
            candidates.append(root / configured_path)
            if "sync" in configured_path.parts:
                sync_index = configured_path.parts.index("sync")
                candidates.append(root.joinpath(*configured_path.parts[sync_index:]))
        candidates.append(root / "sync" / "gopro_originals" / configured_path.name)

    originals_dir = root / "sync" / "gopro_originals"
    media = timing.get("media")
    if isinstance(media, dict) and isinstance(media.get("name"), str):
        candidates.extend(
            sorted(
                originals_dir.glob(
                    f"episode_{episode_index:06d}__*__{Path(media['name']).name}"
                )
            )
        )
    candidates.extend(sorted(originals_dir.glob(f"episode_{episode_index:06d}__*")))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return root / _episode_video_path(episode_index)


def load_capture_timing_prior(root: str | Path, *, episode_index: int) -> dict | None:
    root = Path(root)
    timing_path = capture_timing_path(root, episode_index=episode_index)
    if not timing_path.exists():
        return None
    try:
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    first_pico_local_ns = _first_pico_local_timestamp_ns(root, episode_index=episode_index)
    if first_pico_local_ns is None:
        return None
    try:
        start_request_ns = int(timing["start_request_host_ns"])
        start_ack_ns = int(timing["start_ack_host_ns"])
    except (KeyError, TypeError, ValueError):
        return None
    start_mid_ns = 0.5 * (start_request_ns + start_ack_ns)
    command_half_rtt_s = abs(start_ack_ns - start_request_ns) * 0.5e-9
    prior_offset_s = -float(first_pico_local_ns - start_mid_ns) * 1e-9
    return {
        "capture_prior_offset_s": prior_offset_s,
        "capture_prior_uncertainty_s": command_half_rtt_s,
        "capture_timing_path": str(timing_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Align GoPro video telemetry to Pico LeRobot data.")
    parser.add_argument("--root", required=True, type=Path, help="LeRobot dataset root")
    parser.add_argument("--episode-index", required=True, type=int, help="Episode index to align")
    parser.add_argument("--video", type=Path, default=None, help="GoPro MP4 path")
    parser.add_argument(
        "--max-offset",
        type=float,
        default=1.0,
        help="Residual gyro search radius in seconds after applying the capture-time prior",
    )
    parser.add_argument("--resample-hz", type=float, default=RESAMPLE_HZ, help="Correlation resampling rate")
    parser.add_argument(
        "--window-sec",
        type=float,
        default=DEFAULT_WINDOW_S,
        help="High-motion consensus window length",
    )
    parser.add_argument(
        "--step-sec",
        type=float,
        default=DEFAULT_STEP_S,
        help="High-motion consensus window step",
    )
    parser.add_argument(
        "--no-capture-prior",
        action="store_true",
        help="Ignore sync/episode_*.gopro_capture.json host timestamp prior",
    )
    parser.add_argument("--drift", action="store_true", help="Also estimate clock drift from start/end offsets")
    parser.add_argument(
        "--pico-joint",
        choices=("auto", "chest", "head"),
        default="chest",
        help="Pico motion joint; chest matches the chest-mounted GoPro",
    )
    parser.add_argument(
        "--pico-motion-source",
        choices=("pose_delta", "sdk_velocity"),
        default="pose_delta",
        help="Derive angular speed from pose deltas (recommended) or use SDK velocity",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    result = align_lerobot_episode(
        root=args.root,
        episode_index=args.episode_index,
        video=args.video,
        max_offset_s=args.max_offset,
        resample_hz=args.resample_hz,
        window_s=args.window_sec,
        step_s=args.step_sec,
        use_capture_prior=not args.no_capture_prior,
        drift=args.drift,
        pico_joint=args.pico_joint,
        pico_motion_source=args.pico_motion_source,
    )
    output_path = args.out or default_alignment_path(args.root, episode_index=args.episode_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "GoPro offset: "
        f"{result['offset_s'] * 1000.0:.1f} ms, "
        f"confidence {result['confidence']:.3f}, "
        f"estimated sync error {result['estimated_sync_error_s'] * 1000.0:.1f} ms"
    )
    print(
        f"Pico motion signal: {result['pico_motion_joint']} "
        f"(joint {result['pico_motion_joint_index']}, {result['pico_motion_source']})"
    )
    if result.get("alignment_model") == "affine":
        print(f"drift: {result['drift_s_per_s'] * 1000.0:.3f} ms/s")
    print(f"wrote {output_path}")
    return 0


def _episode_data_path(episode_index: int) -> str:
    return f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"


def _episode_video_path(episode_index: int) -> str:
    return GOPRO_VIDEO_PATH.format(
        episode_chunk=episode_index // 1000,
        episode_index=episode_index,
    )


def _select_pico_timestamps_ns(
    joint_timestamps: np.ndarray,
    local_timestamps: np.ndarray,
    *,
    joint_index: int,
) -> np.ndarray:
    if joint_timestamps.ndim == 2 and joint_timestamps.shape[1] > joint_index:
        selected_timestamps = joint_timestamps[:, joint_index]
        if np.all(selected_timestamps > 0) and np.all(np.diff(selected_timestamps) >= 0):
            return selected_timestamps
    return local_timestamps


def _validate_timeseries(ts: np.ndarray, values: np.ndarray, name: str) -> None:
    if ts.ndim != 1 or values.ndim != 1 or len(ts) != len(values):
        raise ValueError(f"{name} timestamps and values must be 1-D arrays with the same length")
    if len(ts) < 2:
        raise ValueError(f"{name} timeseries must have at least two samples")
    if not np.all(np.isfinite(ts)) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} timeseries contains NaN or Inf")
    if np.any(np.diff(ts) < 0):
        raise ValueError(f"{name} timestamps must be monotonic")


def _resample(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    t0: float,
    t1: float,
    hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(t0, t1, 1.0 / float(hz))
    return grid, np.interp(grid, timestamps_s, values, left=0.0, right=0.0)


def _prepare_motion_pair(
    ts_a: np.ndarray,
    a: np.ndarray,
    ts_b: np.ndarray,
    b: np.ndarray,
    *,
    hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(hz) or hz <= 0.0:
        raise ValueError("hz must be positive")
    t0 = max(float(ts_a[0]), float(ts_b[0]))
    t1 = min(float(ts_a[-1]), float(ts_b[-1]))
    if t1 - t0 < 1.0:
        raise ValueError("overlapping time range is shorter than 1 second")
    grid, resampled_a = _resample(ts_a, a, t0, t1, hz)
    _, resampled_b = _resample(ts_b, b, t0, t1, hz)
    if len(grid) < 3:
        raise ValueError("common resampling grid has fewer than three samples")
    return (
        grid,
        _preprocess_motion(resampled_a, hz=hz),
        _preprocess_motion(resampled_b, hz=hz),
    )


def _preprocess_motion(values: np.ndarray, *, hz: float) -> np.ndarray:
    """Apply the same zero-phase robust transform to both motion sensors."""

    smoothed = _gaussian_smooth_reflect(
        np.asarray(values, dtype=np.float64),
        sigma_samples=MOTION_SMOOTH_SIGMA_S * float(hz),
    )
    floor = float(np.quantile(smoothed, MOTION_FLOOR_QUANTILE))
    transformed = np.maximum(smoothed - floor, 0.0) ** MOTION_POWER
    transformed -= transformed.mean()
    scale = float(transformed.std())
    if scale <= 1e-12:
        return np.zeros_like(transformed)
    return transformed / scale


def _gaussian_smooth_reflect(values: np.ndarray, *, sigma_samples: float) -> np.ndarray:
    if len(values) < 2 or sigma_samples <= 0.0:
        return values.copy()
    radius = max(1, int(np.ceil(4.0 * sigma_samples)))
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _overlap_pearson_curve(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_offset_s: float,
    hz: float,
    center_offset_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    max_offset_s = float(max_offset_s)
    if not np.isfinite(max_offset_s) or max_offset_s < 0.0:
        raise ValueError("max_offset_s must be non-negative")
    low = int(np.ceil((float(center_offset_s) - max_offset_s) * hz))
    high = int(np.floor((float(center_offset_s) + max_offset_s) * hz))
    available = min(len(a), len(b)) - 2
    low = max(low, -available)
    high = min(high, available)
    if low > high:
        raise ValueError("max_offset_s leaves no correlation lags to search")

    lag_samples = np.arange(low, high + 1, dtype=np.int64)
    scores = np.empty(len(lag_samples), dtype=np.float64)
    for index, lag in enumerate(lag_samples):
        if lag < 0:
            overlap_a, overlap_b = a[:lag], b[-lag:]
        elif lag > 0:
            overlap_a, overlap_b = a[lag:], b[:-lag]
        else:
            overlap_a, overlap_b = a, b
        centered_a = overlap_a - overlap_a.mean()
        centered_b = overlap_b - overlap_b.mean()
        denominator = np.linalg.norm(centered_a) * np.linalg.norm(centered_b)
        scores[index] = (
            float(np.dot(centered_a, centered_b) / denominator)
            if denominator > 1e-12
            else 0.0
        )
    return lag_samples.astype(np.float64) / float(hz), scores


def _correlation_diagnostics(
    lags_s: np.ndarray,
    correlation: np.ndarray,
    *,
    hz: float,
    center_offset_s: float,
    max_offset_s: float,
) -> dict:
    if len(correlation) == 0 or not np.any(np.isfinite(correlation)):
        raise ValueError("correlation search produced no finite scores")
    peak = int(np.nanargmax(correlation))
    sub_sample_delta = _parabolic_peak_delta(correlation, peak)
    offset_s = float(lags_s[peak] + sub_sample_delta / float(hz))
    return {
        "peak_index": peak,
        "offset_s": offset_s,
        "confidence": float(correlation[peak]),
        "peak_lag_s": float(lags_s[peak]),
        "sub_sample_delta": float(sub_sample_delta),
        "search_boundary_hit": peak == 0 or peak == len(correlation) - 1,
        "search_center_offset_s": float(center_offset_s),
        "search_radius_s": float(max_offset_s),
    }


def _parabolic_peak_delta(values: np.ndarray, peak: int) -> float:
    if peak <= 0 or peak >= len(values) - 1:
        return 0.0
    y0 = values[peak - 1]
    y1 = values[peak]
    y2 = values[peak + 1]
    denominator = y0 - 2 * y1 + y2
    if denominator == 0:
        return 0.0
    return float(0.5 * (y0 - y2) / denominator)


def _second_peak_value(values: np.ndarray, peak: int, *, exclusion: int) -> float:
    if len(values) <= 1:
        return 0.0
    masked = np.asarray(values, dtype=np.float64).copy()
    low = max(0, peak - exclusion)
    high = min(len(masked), peak + exclusion + 1)
    masked[low:high] = -np.inf
    if not np.any(np.isfinite(masked)):
        return 0.0
    return float(np.max(masked))


def _peak_half_width_s(values: np.ndarray, peak: int, hz: float) -> float:
    peak_value = float(values[peak])
    if peak_value <= 0:
        return 1.0 / float(hz)
    threshold = 0.5 * peak_value
    left = peak
    while left > 0 and values[left] >= threshold:
        left -= 1
    right = peak
    while right < len(values) - 1 and values[right] >= threshold:
        right += 1
    return float(max(1, right - left) / float(hz))


def _median_period_s(timestamps_s: np.ndarray) -> float:
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    diffs = np.diff(timestamps_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 0.0
    return float(np.median(diffs))


def _segment_offset_diagnostics(
    ts_pico: np.ndarray,
    mag_pico: np.ndarray,
    ts_gopro: np.ndarray,
    mag_gopro: np.ndarray,
    *,
    max_offset_s: float,
    hz: float,
    window_s: float,
    step_s: float | None,
    center_offset_s: float,
) -> list[dict]:
    window_s = float(window_s)
    if not np.isfinite(window_s) or window_s < 1.0:
        raise ValueError("window_s must be at least one second")
    grid, pico_prepared, gopro_prepared = _prepare_motion_pair(
        ts_pico,
        mag_pico,
        ts_gopro,
        mag_gopro,
        hz=hz,
    )
    window_samples = int(round(window_s * hz))
    if len(grid) < window_samples:
        return []
    step_s = float(step_s if step_s is not None else window_s / 2.0)
    if not np.isfinite(step_s) or step_s <= 0.0:
        raise ValueError("step_s must be positive")
    step_samples = max(1, int(round(step_s * hz)))
    segments: list[dict] = []
    for start in range(0, len(grid) - window_samples + 1, step_samples):
        stop = start + window_samples
        pico_window = pico_prepared[start:stop]
        gopro_window = gopro_prepared[start:stop]
        try:
            lags_s, correlation = _overlap_pearson_curve(
                pico_window,
                gopro_window,
                max_offset_s=max_offset_s,
                hz=hz,
                center_offset_s=center_offset_s,
            )
            diag = _correlation_diagnostics(
                lags_s,
                correlation,
                hz=hz,
                center_offset_s=center_offset_s,
                max_offset_s=max_offset_s,
            )
        except ValueError:
            continue
        peak = int(diag.pop("peak_index"))
        second = _second_peak_value(
            correlation,
            peak,
            exclusion=max(1, int(round(0.10 * hz))),
        )
        peak_value = float(correlation[peak])
        half_width_s = _peak_half_width_s(correlation, peak, hz)
        segments.append(
            {
                "center_s": float(grid[start] + 0.5 * window_s),
                "offset_s": float(diag["offset_s"]),
                "confidence": float(diag["confidence"]),
                "peak_ratio": float(abs(peak_value) / (abs(second) + 1e-12)),
                "peak_margin": float(peak_value - second),
                "peak_fwhm_s": float(half_width_s),
                "motion_energy": float(
                    np.std(pico_window) * np.std(gopro_window)
                ),
                "search_boundary_hit": bool(diag["search_boundary_hit"]),
            }
        )
    return segments


def _fit_segment_drift(segments: list[dict]) -> dict | None:
    good = [
        segment
        for segment in segments
        if segment["confidence"] > 0.25 and np.isfinite(segment["offset_s"])
    ]
    if len(good) < 2:
        return None
    centers = np.asarray([segment["center_s"] for segment in good], dtype=np.float64)
    offsets = np.asarray([segment["offset_s"] for segment in good], dtype=np.float64)
    weights = np.asarray([max(0.05, segment["confidence"]) for segment in good], dtype=np.float64)
    coeff = np.polyfit(centers, offsets, deg=1, w=weights)
    drift_s_per_s = float(coeff[0])
    affine_offset_s = float(coeff[1])
    fitted = drift_s_per_s * centers + affine_offset_s
    residuals = offsets - fitted
    residual_rms = float(np.sqrt(np.average(residuals**2, weights=weights)))
    return {
        "affine_offset_s": affine_offset_s,
        "drift_s_per_s": drift_s_per_s,
        "segment_residual_rms_s": residual_rms,
        "segment_residual_max_s": float(np.max(np.abs(residuals))),
        "segment_count": int(len(good)),
    }


def _frame_pairing_error_s(
    root: Path,
    quality: dict,
    *,
    episode_index: int,
    video_path: Path | None = None,
) -> dict | None:
    fps = probe_video_fps(video_path) if video_path is not None else None
    fps_source = "video_probe"
    if fps is None:
        fps = _gopro_fps(root)
        fps_source = "dataset_metadata"
    if fps is None or fps <= 0:
        return None
    camera_half_frame_s = 0.5 / float(fps)
    pico_half_sample_s = 0.5 * float(quality.get("pico_sample_period_s", 0.0))
    continuous_error_s = float(quality["estimated_sync_error_s"])
    nearest_error_s = float(
        np.sqrt(continuous_error_s**2 + camera_half_frame_s**2 + pico_half_sample_s**2)
    )
    return {
        "video_fps": float(fps),
        "video_fps_source": fps_source,
        "video_frame_offset": float(quality["offset_s"]) * float(fps),
        "video_frame_offset_rounded": int(round(float(quality["offset_s"]) * float(fps))),
        "camera_half_frame_s": camera_half_frame_s,
        "estimated_nearest_frame_pairing_error_s": nearest_error_s,
    }


def probe_video_fps(video: str | Path) -> float | None:
    """Read the real video-stream rate instead of trusting dataset metadata."""

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(completed.stdout).get("streams", [])
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return None
    if not streams:
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = streams[0].get(key)
        try:
            fps = float(Fraction(str(raw)))
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(fps) and fps > 0.0:
            return fps
    return None


def _gopro_fps(root: Path) -> float | None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        video_info = info["features"]["observation.images.gopro"]["video_info"]
        return float(video_info["video.fps"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _first_pico_local_timestamp_ns(root: Path, *, episode_index: int) -> int | None:
    parquet_path = root / _episode_data_path(episode_index)
    if not parquet_path.exists():
        return None
    try:
        table = pq.read_table(
            parquet_path,
            columns=["observation.pico.local_timestamp_ns"],
        )
    except (FileNotFoundError, KeyError):
        return None
    values = table.to_pydict().get("observation.pico.local_timestamp_ns", [])
    if not values:
        return None
    try:
        return int(values[0])
    except (TypeError, ValueError):
        return None


def _estimate_legacy_drift(
    ts_pico: np.ndarray,
    mag_pico: np.ndarray,
    ts_gopro: np.ndarray,
    mag_gopro: np.ndarray,
    max_offset_s: float,
) -> dict:
    overlap_start_s = max(float(ts_pico[0]), float(ts_gopro[0]))
    overlap_end_s = min(float(ts_pico[-1]), float(ts_gopro[-1]))
    third = (overlap_end_s - overlap_start_s) / 3.0
    if third <= 0:
        raise ValueError("Pico and GoPro have no overlapping range for drift estimation")
    pico_start = _segment(ts_pico, mag_pico, overlap_start_s, overlap_start_s + third)
    gopro_start = _segment(ts_gopro, mag_gopro, overlap_start_s, overlap_start_s + third)
    pico_end = _segment(ts_pico, mag_pico, overlap_end_s - third, overlap_end_s)
    gopro_end = _segment(ts_gopro, mag_gopro, overlap_end_s - third, overlap_end_s)
    start_offset, start_conf = xcorr_offset(*pico_start, *gopro_start, max_offset_s=max_offset_s)
    end_offset, end_conf = xcorr_offset(*pico_end, *gopro_end, max_offset_s=max_offset_s)
    midpoint_start = overlap_start_s + third / 2.0
    midpoint_end = overlap_end_s - third / 2.0
    drift_s_per_s = (end_offset - start_offset) / (midpoint_end - midpoint_start)
    return {
        "legacy_offset_start_s": start_offset,
        "legacy_offset_end_s": end_offset,
        "legacy_drift_s_per_s": drift_s_per_s,
        "legacy_confidence_start": start_conf,
        "legacy_confidence_end": end_conf,
    }


def _segment(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    low_s: float,
    high_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (timestamps_s >= low_s) & (timestamps_s <= high_s)
    return timestamps_s[mask], values[mask]


if __name__ == "__main__":
    raise SystemExit(main())
