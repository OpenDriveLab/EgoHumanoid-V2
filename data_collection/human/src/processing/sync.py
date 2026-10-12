"""Audit or batch-align Pico and GoPro episodes without rewriting raw media."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable

from human.storage.lerobot import discover_episode_indices
from human.processing.resample import apply_dataset_sync
from human.processing.quality import audit_dataset, write_report

from human.processing.offset import (
    RESAMPLE_HZ,
    apply_session_consensus_prior,
    align_lerobot_episode,
    default_alignment_path,
)


def _alignment_score(payload: dict) -> int:
    """Rank candidate offsets for session-prior selection, never filter exports."""

    if payload.get("alignment_evidence") == "exact_frame_identity_only":
        # Processed exports may retain exact Pico/video frame identity after
        # GoPro telemetry has been stripped. They are usable for deterministic
        # resampling, but must never be presented as an independently measured
        # temporal offset.
        return (
            1
            if payload.get("exact_frame_count_match") is True
            and payload.get("temporal_offset_verified") is False
            and float(payload.get("offset_s", float("nan"))) == 0.0
            else -1
        )
    if payload.get("search_boundary_hit") is True:
        return 0
    if payload.get("pico_motion_source") == "pose_delta":
        try:
            confidence = float(payload["confidence"])
            peak_margin = float(payload["peak_margin"])
            error_s = float(payload["estimated_sync_error_s"])
            window_count = int(payload["window_count"])
            window_mad_s = float(payload["window_offset_mad_s"])
            global_window_delta_s = float(payload["global_window_delta_s"])
            consensus_50ms = int(payload["window_consensus_50ms"])
            consensus_100ms = int(payload["window_consensus_100ms"])
        except (KeyError, TypeError, ValueError):
            return -1
        session_prior_outlier = _is_session_prior_outlier(payload)
        if (
            window_count >= 2
            and confidence >= 0.75
            and peak_margin >= 0.015
            and error_s <= 0.10
            and window_mad_s <= 0.05
            and global_window_delta_s <= 0.05
            and consensus_50ms >= 2
        ):
            return 1 if session_prior_outlier else 2
        if (
            window_count >= 2
            and confidence >= 0.65
            and peak_margin >= 0.010
            and error_s <= 0.20
            and window_mad_s <= 0.10
            and global_window_delta_s <= 0.10
            and consensus_100ms >= 2
        ):
            return 0 if session_prior_outlier else 1
        if (
            _has_stable_session_support(payload)
            and window_count >= 2
            and confidence >= 0.50
            and peak_margin >= 0.015
            and error_s <= 0.10
            and window_mad_s <= 0.05
            and global_window_delta_s <= 0.075
            and consensus_100ms >= 2
        ):
            return 1
        return 0
    try:
        confidence = float(payload["confidence"])
        peak_ratio = float(payload["peak_ratio"])
        error_s = float(payload["estimated_sync_error_s"])
    except (KeyError, TypeError, ValueError):
        return -1
    if confidence >= 0.5 and peak_ratio >= 1.05 and error_s <= 0.1:
        return 2
    if confidence >= 0.25 and peak_ratio >= 1.01 and error_s <= 0.3:
        return 1
    return 0


def _is_session_prior_outlier(payload: dict) -> bool:
    try:
        support = int(payload["session_prior_support"])
        prior_mad_s = float(payload["session_prior_mad_s"])
        prior_delta_s = float(payload["session_prior_delta_s"])
    except (KeyError, TypeError, ValueError):
        return False
    if support < 4 or not all(math.isfinite(value) for value in (prior_mad_s, prior_delta_s)):
        return False
    return prior_mad_s <= 0.050 and prior_delta_s > max(0.100, 4.0 * prior_mad_s)


def _has_stable_session_support(payload: dict) -> bool:
    try:
        support = int(payload["session_prior_support"])
        prior_mad_s = float(payload["session_prior_mad_s"])
        prior_delta_s = float(payload["session_prior_delta_s"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        # Three high-confidence offsets are the smallest cohort whose
        # median remains defined by two agreeing observations when one is an
        # outlier.  Requiring four prevented otherwise coherent low-motion
        # sessions from using any session prior at all.
        support >= 3
        and all(math.isfinite(value) for value in (prior_mad_s, prior_delta_s))
        and prior_mad_s <= 0.050
        and prior_delta_s <= max(0.100, 4.0 * prior_mad_s)
    )


def estimate_session_offset_prior(
    alignments: dict[int, dict],
    *,
    episode_index: int,
    neighbor_count: int = 6,
    minimum_support: int = 3,
) -> dict:
    """Build a local prior from at least three high-confidence neighbors."""

    target = alignments.get(int(episode_index))
    if not isinstance(target, dict):
        return {}
    try:
        target_offset_s = float(target["offset_s"])
    except (KeyError, TypeError, ValueError):
        return {}
    candidates: list[tuple[int, int, float]] = []
    for candidate_index, payload in alignments.items():
        candidate_index = int(candidate_index)
        if candidate_index == int(episode_index) or not isinstance(payload, dict):
            continue
        if _alignment_score(payload) != 2:
            continue
        try:
            offset_s = float(payload["offset_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(offset_s):
            continue
        candidates.append(
            (abs(candidate_index - int(episode_index)), candidate_index, offset_s)
        )
    selected = sorted(candidates)[: max(0, int(neighbor_count))]
    if len(selected) < int(minimum_support):
        return {}
    offsets = [offset_s for _, _, offset_s in selected]
    prior_offset_s = float(statistics.median(offsets))
    prior_mad_s = float(
        statistics.median(abs(offset_s - prior_offset_s) for offset_s in offsets)
    )
    return {
        "session_prior_offset_s": prior_offset_s,
        "session_prior_mad_s": prior_mad_s,
        "session_prior_delta_s": abs(target_offset_s - prior_offset_s),
        "session_prior_support": len(selected),
        "session_prior_episode_indices": [index for _, index, _ in selected],
        "session_prior_method": "nearest_high_confidence_leave_one_out_median",
    }


def inspect_dataset_sync(root: str | Path) -> dict:
    """Return a read-only inventory of coarse capture and fine alignment files."""

    root = Path(root)
    episodes = []
    counts: Counter[str] = Counter(
        {"aligned": 0, "capture_only": 0, "missing_capture": 0, "invalid_alignment": 0}
    )
    for episode_index in discover_episode_indices(root):
        capture_path = root / "sync" / f"episode_{episode_index:06d}.gopro_capture.json"
        alignment_path = default_alignment_path(root, episode_index=episode_index)
        item = {
            "episode_index": episode_index,
            "capture_path": str(capture_path),
            "alignment_path": str(alignment_path),
        }
        if alignment_path.exists():
            try:
                payload = json.loads(alignment_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                item.update({
                    "state": "aligned",
                    "confidence": payload.get("confidence"),
                    "offset_s": payload.get("offset_s"),
                    "estimated_sync_error_s": payload.get("estimated_sync_error_s"),
                })
                counts["aligned"] += 1
            else:
                item["state"] = "invalid_alignment"
                counts["invalid_alignment"] += 1
        elif capture_path.exists():
            item.update({"state": "capture_only"})
            counts["capture_only"] += 1
        else:
            item.update({"state": "missing_capture"})
            counts["missing_capture"] += 1
        episodes.append(item)
    return {
        "dataset_root": str(root),
        "total_episodes": len(episodes),
        "counts": dict(counts),
        "episodes": episodes,
    }


def align_dataset(
    root: str | Path,
    *,
    episode_indices: Iterable[int] | None = None,
    overwrite: bool = False,
    pico_joint: str = "chest",
    pico_motion_source: str = "pose_delta",
    max_offset_s: float = 1.0,
    resample_hz: float = RESAMPLE_HZ,
    window_s: float = 3.0,
    step_s: float | None = 1.5,
    session_auto_fallback: bool = False,
) -> dict:
    """Align selected episodes and write only JSON files under ``sync/``."""

    root = Path(root)
    indices = (
        discover_episode_indices(root)
        if episode_indices is None
        else sorted(set(int(index) for index in episode_indices))
    )
    aligned = []
    skipped = []
    failures = []
    pending_indices = []
    for episode_index in indices:
        output_path = default_alignment_path(root, episode_index=episode_index)
        if output_path.exists() and not overwrite:
            skipped.append({"episode_index": episode_index, "reason": "already_aligned"})
        else:
            pending_indices.append(episode_index)

    # ``align_lerobot_episode`` imports telemetry_parser lazily.  Without this
    # preflight a missing processing dependency is caught once per episode and
    # later looks like a dataset-wide alignment-quality failure.  Fail before
    # doing any episode work so an environment fault cannot be misclassified as
    # bad source data.  Fully prealigned datasets do not need this dependency.
    if pending_indices and importlib.util.find_spec("telemetry_parser") is None:
        raise RuntimeError(
            "GoPro batch alignment requires telemetry-parser before processing "
            "any episodes; install the EH2 human processing dependencies"
        )

    pending: dict[int, dict] = {}
    for episode_index in pending_indices:
        try:
            result = align_lerobot_episode(
                root=root,
                episode_index=episode_index,
                pico_joint=pico_joint,
                pico_motion_source=pico_motion_source,
                max_offset_s=max_offset_s,
                resample_hz=resample_hz,
                window_s=window_s,
                step_s=step_s,
            )
            pending[episode_index] = result
        except Exception as exc:
            failures.append({"episode_index": episode_index, "error": str(exc)})

    reference_alignments: dict[int, dict] = {}
    for reference_index in discover_episode_indices(root):
        alignment_path = default_alignment_path(root, episode_index=reference_index)
        if not alignment_path.exists():
            continue
        try:
            payload = json.loads(alignment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            reference_alignments[reference_index] = payload
    reference_alignments.update(pending)

    for episode_index, initial_result in pending.items():
        output_path = default_alignment_path(root, episode_index=episode_index)
        try:
            result = dict(initial_result)
            session_prior = estimate_session_offset_prior(
                reference_alignments,
                episode_index=episode_index,
            )
            if session_prior:
                result.update(session_prior)
                if isinstance(result.get("segment_offsets"), list):
                    result = apply_session_consensus_prior(
                        result,
                        prior_offset_s=float(session_prior["session_prior_offset_s"]),
                    )
            initial_score = _alignment_score(result)
            if (
                session_auto_fallback
                and pico_joint == "chest"
                and initial_score < 2
                and session_prior
            ):
                try:
                    fallback = align_lerobot_episode(
                        root=root,
                        episode_index=episode_index,
                        pico_joint="auto",
                        pico_motion_source=pico_motion_source,
                        max_offset_s=max_offset_s,
                        resample_hz=resample_hz,
                        window_s=window_s,
                        step_s=step_s,
                        session_offset_prior_s=float(
                            session_prior["session_prior_offset_s"]
                        ),
                    )
                    fallback.update(session_prior)
                    fallback["session_prior_delta_s"] = abs(
                        float(fallback["offset_s"])
                        - float(session_prior["session_prior_offset_s"])
                    )
                    if isinstance(fallback.get("segment_offsets"), list):
                        fallback = apply_session_consensus_prior(
                            fallback,
                            prior_offset_s=float(
                                session_prior["session_prior_offset_s"]
                            ),
                        )
                    if _alignment_score(fallback) > initial_score:
                        fallback.update(
                            {
                                "session_auto_fallback_selected": True,
                                "session_auto_fallback_previous_confidence": result.get("confidence"),
                                "session_auto_fallback_previous_joint": result.get(
                                    "pico_motion_joint"
                                ),
                                "session_auto_fallback_previous_offset_s": float(
                                    result["offset_s"]
                                ),
                            }
                        )
                        result = fallback
                except Exception as exc:
                    result["session_auto_fallback_error"] = str(exc)
            _write_json_atomic(output_path, result)
            aligned.append(
                {
                    "episode_index": episode_index,
                    "offset_s": float(result["offset_s"]),
                    "confidence": float(result["confidence"]),
                    "estimated_sync_error_s": float(result["estimated_sync_error_s"]),
                    "pico_motion_joint": result["pico_motion_joint"],
                }
            )
        except Exception as exc:
            failures.append({"episode_index": episode_index, "error": str(exc)})
    return {
        "dataset_root": str(root),
        "requested": len(indices),
        "aligned": aligned,
        "skipped": skipped,
        "failures": failures,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def metadata_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit or batch-align Pico and GoPro timestamps without rewriting media."
    )
    parser.add_argument("--root", required=True, type=Path, help="LeRobot dataset root")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--audit", action="store_true", help="Only inspect existing sync metadata")
    action.add_argument("--align", action="store_true", help="Run gyro alignment")
    parser.add_argument(
        "--episode-index",
        action="append",
        type=int,
        default=None,
        help="Only align this episode; repeat to select multiple episodes",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing alignment JSON")
    parser.add_argument(
        "--pico-joint",
        choices=("auto", "chest", "head"),
        default="chest",
    )
    parser.add_argument(
        "--pico-motion-source",
        choices=("pose_delta", "sdk_velocity"),
        default="pose_delta",
    )
    parser.add_argument(
        "--max-offset",
        type=float,
        default=1.0,
        help="Residual gyro search radius after applying capture timing",
    )
    parser.add_argument("--resample-hz", type=float, default=RESAMPLE_HZ)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.5)
    parser.add_argument(
        "--session-auto-fallback",
        action="store_true",
        help="Retry weak chest results with head/chest auto selection constrained by the session prior",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional report JSON path")
    args = parser.parse_args(argv)

    if args.audit:
        report = inspect_dataset_sync(args.root)
        print(json.dumps(report["counts"], ensure_ascii=False))
        if args.out is not None:
            _write_json_atomic(args.out, report)
        return 0

    report = align_dataset(
        args.root,
        episode_indices=args.episode_index,
        overwrite=args.overwrite,
        pico_joint=args.pico_joint,
        pico_motion_source=args.pico_motion_source,
        max_offset_s=args.max_offset,
        resample_hz=args.resample_hz,
        window_s=args.window_sec,
        step_s=args.step_sec,
        session_auto_fallback=args.session_auto_fallback,
    )
    output_path = args.out or args.root / "sync" / "alignment_summary.json"
    _write_json_atomic(output_path, report)
    for item in report["aligned"]:
        print(
            f"episode_{item['episode_index']:06d}: confidence={item['confidence']:.3f}, "
            f"offset={item['offset_s'] * 1000.0:.1f} ms, "
            f"signal={item['pico_motion_joint']}"
        )
    for failure in report["failures"]:
        print(
            f"episode_{failure['episode_index']:06d}: failed: {failure['error']}",
            file=sys.stderr,
        )
    print(f"wrote {output_path}")
    return 1 if report["failures"] else 0


def run_sync_pipeline(
    root: str | Path,
    *,
    output_root: str | Path | None = None,
    output_fps: float = 50.0,
    overwrite_alignments: bool = True,
    overwrite_output: bool = False,
    workers: int = 1,
    session_auto_fallback: bool = True,
    pico_joint: str = "auto",
    max_offset_s: float = 1.0,
    window_s: float = 3.0,
    step_s: float = 1.5,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict:
    """Check, synchronize and audit a dataset without changing recorded motion."""

    root = Path(root).resolve()
    destination = (
        Path(output_root).resolve()
        if output_root is not None
        else root.with_name(f"{root.name}_synced")
    )
    if root == destination or root.is_relative_to(destination) or destination.is_relative_to(root):
        raise ValueError("sync output must not overlap source")
    if destination.exists() and not overwrite_output:
        raise FileExistsError(destination)
    if not math.isfinite(output_fps) or output_fps <= 0 or workers < 1:
        raise ValueError("output_fps and workers must be positive")
    raw_quality = audit_dataset(root, ffprobe=ffprobe, workers=workers)
    alignment_report = align_dataset(
        root,
        overwrite=overwrite_alignments,
        pico_joint=pico_joint,
        max_offset_s=max_offset_s,
        window_s=window_s,
        step_s=step_s,
        session_auto_fallback=session_auto_fallback,
    )
    if alignment_report["failures"]:
        details = "; ".join(
            f"episode {failure['episode_index']}: {failure['error']}"
            for failure in alignment_report["failures"]
        )
        raise RuntimeError(f"Alignment execution failed: {details}")

    audit_report = inspect_dataset_sync(root)
    episodes = audit_report["episodes"]
    if sorted(item["episode_index"] for item in episodes) != list(range(raw_quality["total_episodes"])):
        raise RuntimeError("Alignment must cover every input episode exactly once")
    episode_diagnostics = []
    for item in episodes:
        payload = json.loads(Path(item["alignment_path"]).read_text())
        model = payload.get("alignment_model", "constant")
        offset = payload.get("affine_offset_s") if model == "affine" else payload.get("offset_s")
        drift = payload.get("drift_s_per_s", 0.0) if model == "affine" else 0.0
        if (model not in {"constant", "affine"}
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (offset, drift))
                or 1 + drift <= 0):
            raise ValueError(f"Invalid synchronization clock: episode {item['episode_index']}")
        episode_diagnostics.append({
            "episode_index": item["episode_index"],
            "offset_s": offset,
            "drift_s_per_s": drift,
            "confidence": payload.get("confidence"),
            "estimated_sync_error_s": payload.get("estimated_sync_error_s"),
            "alignment_evidence": payload.get("alignment_evidence", "motion_correlation"),
        })

    application_report = apply_dataset_sync(
        root,
        output_root=destination,
        output_fps=output_fps,
        overwrite=overwrite_output,
        workers=workers,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
    )
    applied = application_report["applied"]
    quality = audit_dataset(destination, expected_fps=output_fps,
                            require_exact_video_frames=True, ffprobe=ffprobe, workers=workers)
    exact = (sorted(item["episode_index"] for item in applied) == list(range(len(episodes)))
             and all(item.get("frame_count_match") is True for item in applied))
    report = {
        "status": "completed",
        "source_root": str(root),
        "output_root": str(destination),
        "episodes": episode_diagnostics,
        "alignment": alignment_report,
        "application": application_report,
        "raw_quality": raw_quality,
        "processed_episodes": len(applied),
        "all_frame_counts_match": exact,
    }
    for name, content in (("sync_report.json", report), ("data_quality_report.json", quality)):
        write_report(destination / "meta" / name, content)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Align Pico/GoPro timestamps and export an exact-frame synchronized dataset."
        )
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--reuse-alignments", dest="overwrite_alignments", action="store_false",
                        help="Reuse existing estimates only when the source recordings are unchanged.")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-session-auto-fallback", action="store_true")
    parser.add_argument("--pico-joint", choices=("chest", "head", "auto"), default="auto")
    parser.add_argument("--max-offset", type=float, default=1.0)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.5)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)

    report = run_sync_pipeline(
        args.root,
        output_root=args.output_root,
        output_fps=args.output_fps,
        overwrite_alignments=args.overwrite_alignments,
        overwrite_output=args.overwrite_output,
        workers=args.workers,
        session_auto_fallback=not args.no_session_auto_fallback,
        pico_joint=args.pico_joint,
        max_offset_s=args.max_offset,
        window_s=args.window_sec,
        step_s=args.step_sec,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )
    print(
        f"completed: {report['output_root']}, {report['processed_episodes']} episodes, "
        f"exact_frames={report['all_frame_counts_match']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
