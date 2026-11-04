"""OpenXR palm-center recovery and BrainCo endpoint geometry.

The PICO/OpenXR hand array follows ``XrHandJointEXT`` ordering.  Joint 0 is
``PALM_EXT`` and joint 1 is ``WRIST_EXT``.  This module deliberately uses the
former as the human endpoint; no robot rollout or SONIC-achieved endpoint is
allowed to enter target construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


OPENXR_PALM_JOINT = 0

# Triangle-area centroids of the manufacturer-provided BrainCo Revo2
# ``*_base_link.STL`` palm shells, expressed in each base_link frame.  The
# base_link origin is at the wrist-side end of the palm shell, not its center.
BRAINCO_BASE_TO_PALM_CENTER_TRANSLATIONS_M = {
    "left": np.array([-0.00467923, 0.00023722, 0.04666757], dtype=np.float64),
    "right": np.array([-0.00466300, -0.00024704, 0.04666762], dtype=np.float64),
}

# Anatomical frame columns are [wrist->fingers, index->pinky, palm normal].
# These fixed rotations express those axes in the mirrored Revo2 base frames.
_BRAINCO_BASE_TO_ANATOMICAL_ROTATIONS = {
    "left": np.array(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    ),
    "right": np.array(
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    ),
}


@dataclass(frozen=True)
class PalmReconstruction:
    """A complete per-frame PICO palm pose plus source diagnostics."""

    positions_world: np.ndarray
    rotations_world: np.ndarray
    direct_valid: np.ndarray
    fallback_used: np.ndarray
    outlier_rejected: np.ndarray
    unresolved: np.ndarray


def _pose_arrays(poses: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"{name} must have shape (frames, 7), got {values.shape}")
    return values[:, :3], values[:, 3:7]


def _valid_pose_mask(positions: np.ndarray, quaternions_xyzw: np.ndarray) -> np.ndarray:
    return (
        np.all(np.isfinite(positions), axis=1)
        & np.all(np.isfinite(quaternions_xyzw), axis=1)
        & (np.linalg.norm(quaternions_xyzw, axis=1) > 1e-8)
    )


def _rolling_translation_outliers(
    values: np.ndarray,
    candidates: np.ndarray,
    *,
    window: int,
    threshold_m: float,
) -> np.ndarray:
    if window < 3 or window % 2 == 0:
        raise ValueError("outlier_window must be an odd integer >= 3")
    if threshold_m <= 0.0:
        raise ValueError("translation_outlier_m must be positive")
    rejected = np.zeros(len(values), dtype=bool)
    radius = window // 2
    for index in np.flatnonzero(candidates):
        start = max(0, int(index) - radius)
        stop = min(len(values), int(index) + radius + 1)
        neighbours = values[start:stop][candidates[start:stop]]
        if len(neighbours) < 3:
            continue
        reference = np.median(neighbours, axis=0)
        rejected[index] = np.linalg.norm(values[index] - reference) > threshold_m
    return rejected


def _interpolate_transforms(
    relative_positions: np.ndarray,
    relative_rotations: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return None

    times = np.arange(len(valid), dtype=np.float64)
    translations = np.column_stack(
        [
            np.interp(times, valid_indices, relative_positions[valid_indices, axis])
            for axis in range(3)
        ]
    )
    if valid_indices.size == 1:
        rotations = np.tile(relative_rotations[valid_indices[0]], (len(valid), 1, 1))
    else:
        interpolator = Slerp(
            valid_indices.astype(np.float64),
            R.from_matrix(relative_rotations[valid_indices]),
        )
        clipped_times = np.clip(times, valid_indices[0], valid_indices[-1])
        rotations = interpolator(clipped_times).as_matrix()
    return translations, rotations


def reconstruct_openxr_palm_series(
    hand_poses: np.ndarray,
    hand_active: np.ndarray,
    body_hand_poses: np.ndarray,
    *,
    outlier_window: int = 21,
    translation_outlier_m: float = 0.03,
    maximum_body_relative_translation_m: float = 0.15,
    cohort_relative_position_m: np.ndarray | None = None,
    cohort_relative_rotation: np.ndarray | None = None,
    minimum_direct_samples_for_episode_transform: int = 1,
) -> PalmReconstruction:
    """Recover OpenXR Palm(0), using only the PICO body-hand pose for gaps.

    The time-varying body-hand-to-palm transform is measured on valid Palm(0)
    frames, interpolated through internal gaps, and held at the first/last
    valid transform for boundary gaps.  A centered body-relative position gate
    rejects the common burst of bad samples when optical hand tracking returns.
    A separate generous absolute gate rejects persistent but internally smooth
    tracking-origin or unit errors, which a rolling median cannot detect.
    Direct valid samples remain numerically unchanged.
    """

    hands = np.asarray(hand_poses, dtype=np.float64)
    if hands.ndim != 3 or hands.shape[1:] != (26, 7):
        raise ValueError(f"hand_poses must have shape (frames, 26, 7), got {hands.shape}")
    active = np.asarray(hand_active, dtype=bool).reshape(-1)
    if len(active) != len(hands):
        raise ValueError("hand_active length must match hand_poses")
    palm_positions, palm_quaternions = _pose_arrays(
        hands[:, OPENXR_PALM_JOINT], name="OpenXR palm poses"
    )
    body_positions, body_quaternions = _pose_arrays(
        body_hand_poses, name="PICO body-hand poses"
    )
    if len(body_positions) != len(hands):
        raise ValueError("body_hand_poses length must match hand_poses")
    if (
        not np.isfinite(maximum_body_relative_translation_m)
        or maximum_body_relative_translation_m <= 0.0
    ):
        raise ValueError(
            "maximum_body_relative_translation_m must be finite and positive"
        )

    palm_pose_valid = _valid_pose_mask(palm_positions, palm_quaternions)
    body_pose_valid = _valid_pose_mask(body_positions, body_quaternions)
    candidate = active & palm_pose_valid & body_pose_valid

    safe_palm_quaternions = palm_quaternions.copy()
    safe_body_quaternions = body_quaternions.copy()
    safe_palm_quaternions[~palm_pose_valid] = [0.0, 0.0, 0.0, 1.0]
    safe_body_quaternions[~body_pose_valid] = [0.0, 0.0, 0.0, 1.0]
    palm_rotation = R.from_quat(safe_palm_quaternions)
    body_rotation = R.from_quat(safe_body_quaternions)
    relative_positions = body_rotation.inv().apply(palm_positions - body_positions)
    relative_rotations = (body_rotation.inv() * palm_rotation).as_matrix()

    outlier_rejected = _rolling_translation_outliers(
        relative_positions,
        candidate,
        window=outlier_window,
        threshold_m=translation_outlier_m,
    )
    outlier_rejected |= candidate & (
        np.linalg.norm(relative_positions, axis=1)
        > float(maximum_body_relative_translation_m)
    )
    direct_valid = candidate & ~outlier_rejected
    interpolated = _interpolate_transforms(
        relative_positions,
        relative_rotations,
        direct_valid,
    )
    if minimum_direct_samples_for_episode_transform < 1:
        raise ValueError(
            "minimum_direct_samples_for_episode_transform must be at least 1"
        )
    cohort_supplied = (
        cohort_relative_position_m is not None
        or cohort_relative_rotation is not None
    )
    if cohort_supplied:
        if cohort_relative_position_m is None or cohort_relative_rotation is None:
            raise ValueError(
                "cohort relative position and rotation must be supplied together"
            )
        cohort_position = np.asarray(
            cohort_relative_position_m, dtype=np.float64
        ).reshape(-1)
        cohort_rotation = np.asarray(
            cohort_relative_rotation, dtype=np.float64
        )
        if cohort_position.shape != (3,) or not np.isfinite(cohort_position).all():
            raise ValueError("cohort_relative_position_m must be finite shape (3,)")
        if cohort_rotation.shape != (3, 3) or not np.isfinite(
            cohort_rotation
        ).all():
            raise ValueError("cohort_relative_rotation must be finite shape (3, 3)")
        if not np.allclose(
            cohort_rotation.T @ cohort_rotation,
            np.eye(3),
            atol=1e-6,
        ) or not np.isclose(np.linalg.det(cohort_rotation), 1.0, atol=1e-6):
            raise ValueError("cohort_relative_rotation must be a rotation matrix")
        if (
            interpolated is None
            or np.count_nonzero(direct_valid)
            < minimum_direct_samples_for_episode_transform
        ):
            interpolated = (
                np.tile(cohort_position, (len(hands), 1)),
                np.tile(cohort_rotation, (len(hands), 1, 1)),
            )

    output_positions = np.full((len(hands), 3), np.nan, dtype=np.float64)
    output_rotations = np.tile(np.eye(3), (len(hands), 1, 1))
    output_positions[direct_valid] = palm_positions[direct_valid]
    output_rotations[direct_valid] = palm_rotation.as_matrix()[direct_valid]

    fallback_used = np.zeros(len(hands), dtype=bool)
    if interpolated is not None:
        relative_position_filled, relative_rotation_filled = interpolated
        fallback_used = ~direct_valid & body_pose_valid
        output_positions[fallback_used] = (
            body_positions[fallback_used]
            + body_rotation[fallback_used].apply(relative_position_filled[fallback_used])
        )
        output_rotations[fallback_used] = (
            body_rotation[fallback_used]
            * R.from_matrix(relative_rotation_filled[fallback_used])
        ).as_matrix()

    unresolved = ~direct_valid & ~fallback_used
    return PalmReconstruction(
        positions_world=output_positions,
        rotations_world=output_rotations,
        direct_valid=direct_valid,
        fallback_used=fallback_used,
        outlier_rejected=outlier_rejected,
        unresolved=unresolved,
    )


def _yaw_rotations(yaw_rad: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw_rad, dtype=np.float64).reshape(-1)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    rotations = np.zeros((len(yaw), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = cosine
    rotations[:, 0, 1] = -sine
    rotations[:, 1, 0] = sine
    rotations[:, 1, 1] = cosine
    rotations[:, 2, 2] = 1.0
    return rotations


def pico_palm_pose_to_current_root_local(
    palm_positions_pico_world: np.ndarray,
    palm_rotations_pico_world: np.ndarray,
    root_positions_pico_world: np.ndarray,
    heading_yaw_robot_world_rad: np.ndarray,
    *,
    pico_to_robot_frame: np.ndarray,
    vertical_registration_m: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Express physical palm poses in the current human-root gravity frame."""

    palm_positions = np.asarray(palm_positions_pico_world, dtype=np.float64)
    palm_rotations = np.asarray(palm_rotations_pico_world, dtype=np.float64)
    root_positions = np.asarray(root_positions_pico_world, dtype=np.float64)
    frame_map = np.asarray(pico_to_robot_frame, dtype=np.float64).reshape(3, 3)
    heading = np.asarray(heading_yaw_robot_world_rad, dtype=np.float64).reshape(-1)
    frames = len(palm_positions)
    if palm_positions.shape != (frames, 3):
        raise ValueError("palm_positions_pico_world must have shape (frames, 3)")
    if root_positions.shape != (frames, 3):
        raise ValueError("root_positions_pico_world must have shape (frames, 3)")
    if palm_rotations.shape != (frames, 3, 3) or len(heading) != frames:
        raise ValueError("palm rotations and heading must match position frames")
    registration = np.broadcast_to(
        np.asarray(vertical_registration_m, dtype=np.float64).reshape(-1),
        (frames,),
    )

    relative_robot_world = np.einsum(
        "ij,tj->ti", frame_map, palm_positions - root_positions
    )
    robot_world_rotations = np.einsum(
        "ij,tjk->tik", frame_map, palm_rotations
    )
    world_to_local = _yaw_rotations(-heading)
    local_positions = np.einsum(
        "tij,tj->ti", world_to_local, relative_robot_world
    )
    local_positions[:, 2] += registration
    local_rotations = np.einsum(
        "tij,tjk->tik", world_to_local, robot_world_rotations
    )
    return local_positions, local_rotations


def align_openxr_palm_rotations_to_anatomical_frame(
    openxr_palm_rotations: np.ndarray,
    anatomical_rotations: np.ndarray,
    calibration_valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | list[list[float]]]]:
    """Fit one per-episode OpenXR-joint-to-anatomical rotation.

    OpenXR Palm(0) provides a considerably smoother orientation than rebuilding
    a frame from four finger points every sample.  Its device-local axes are not
    the semantic axes needed by the robot, so the constant offset is estimated
    only from simultaneously valid PICO hand keypoints in the same episode.
    """

    joint = np.asarray(openxr_palm_rotations, dtype=np.float64)
    anatomical = np.asarray(anatomical_rotations, dtype=np.float64)
    valid = np.asarray(calibration_valid, dtype=bool).reshape(-1)
    if joint.ndim != 3 or joint.shape[1:] != (3, 3):
        raise ValueError("openxr_palm_rotations must have shape (frames, 3, 3)")
    if anatomical.shape != joint.shape or len(valid) != len(joint):
        raise ValueError("anatomical rotations and validity must match OpenXR frames")
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        raise ValueError("at least one valid anatomical calibration frame is required")
    offsets = R.from_matrix(joint[indices]).inv() * R.from_matrix(anatomical[indices])
    offset = offsets.mean()
    residual = (offsets * offset.inv()).magnitude()
    aligned = (R.from_matrix(joint) * offset).as_matrix()
    return aligned, {
        "samples": int(indices.size),
        "joint_to_anatomical_rotation": offset.as_matrix().tolist(),
        "residual_mean_deg": float(np.degrees(np.mean(residual))),
        "residual_max_deg": float(np.degrees(np.max(residual))),
    }


def _brainco_center_translation(side: str) -> np.ndarray:
    try:
        return BRAINCO_BASE_TO_PALM_CENTER_TRANSLATIONS_M[side]
    except KeyError as error:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}") from error


def brainco_base_rotations_from_anatomical_palm(
    anatomical_rotations: np.ndarray,
    *,
    side: str,
) -> np.ndarray:
    """Map semantic human palm axes to the mirrored BrainCo base-link axes."""

    values = np.asarray(anatomical_rotations, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError("anatomical_rotations must have shape (frames, 3, 3)")
    try:
        base_to_anatomical = _BRAINCO_BASE_TO_ANATOMICAL_ROTATIONS[side]
    except KeyError as error:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}") from error
    return np.einsum("tij,kj->tik", values, base_to_anatomical)


def brainco_palm_center_pose_from_base_pose(
    base_positions: np.ndarray,
    base_rotations: np.ndarray,
    *,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the IK-controlled BrainCo base_link pose to physical palm center."""

    positions = np.asarray(base_positions, dtype=np.float64)
    rotations = np.asarray(base_rotations, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("base_positions must have shape (frames, 3)")
    if rotations.shape != (len(positions), 3, 3):
        raise ValueError("base_rotations must have shape (frames, 3, 3)")
    translation = _brainco_center_translation(side)
    center_positions = positions + np.einsum("tij,j->ti", rotations, translation)
    return center_positions, rotations.copy()


def brainco_base_pose_from_palm_center_pose(
    center_positions: np.ndarray,
    center_rotations: np.ndarray,
    *,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a physical palm-center target to the base_link target used by IK."""

    positions = np.asarray(center_positions, dtype=np.float64)
    rotations = np.asarray(center_rotations, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("center_positions must have shape (frames, 3)")
    if rotations.shape != (len(positions), 3, 3):
        raise ValueError("center_rotations must have shape (frames, 3, 3)")
    translation = _brainco_center_translation(side)
    base_positions = positions - np.einsum("tij,j->ti", rotations, translation)
    return base_positions, rotations.copy()
