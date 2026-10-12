"""Continuous palm IK with staged waist assistance and temporal arm bounds."""
from typing import Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from .arm_ik import ArmIK


def gravity_local_rotation_from_body_quat(base_quat_world_wxyz):
    """Remove world heading while retaining pelvis roll and pitch."""
    quat = np.asarray(base_quat_world_wxyz, dtype=np.float64).reshape(4)
    world_from_body = Rotation.from_quat(quat[[1, 2, 3, 0]])
    matrix = world_from_body.as_matrix()
    heading = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    return Rotation.from_euler("z", heading).inv() * world_from_body


def poses_gravity_local_to_body(poses_local, base_quat_world_wxyz):
    """Transform gravity-aligned palm poses into pelvis-body coordinates."""
    body_from_local = gravity_local_rotation_from_body_quat(base_quat_world_wxyz).inv()
    source = np.asarray(poses_local, dtype=np.float64).reshape(2, 7)
    output = np.empty_like(source)
    for side in range(2):
        rotation = Rotation.from_quat(source[side, [4, 5, 6, 3]])
        output[side, :3] = body_from_local.apply(source[side, :3])
        output[side, 3:] = (body_from_local * rotation).as_quat()[[3, 0, 1, 2]]
    return output.reshape(14)


def transition_schedule(
    rows: int,
    *,
    prefix_frames: int,
    ramp_frames: int,
) -> np.ndarray:
    """Causal smoothstep from the executable start pose to semantic targets."""

    count = int(rows)
    prefix = int(prefix_frames)
    ramp_count = int(ramp_frames)
    if count < 0 or prefix < 0 or ramp_count < 0:
        raise ValueError("rows, prefix_frames, and ramp_frames must be non-negative")
    schedule = np.ones(count, dtype=np.float64)
    prefix = min(prefix, count)
    schedule[:prefix] = 0.0
    ramp_count = min(ramp_count, count - prefix)
    if ramp_count:
        ramp = np.linspace(0.0, 1.0, ramp_count + 2, dtype=np.float64)[1:-1]
        schedule[prefix : prefix + ramp_count] = ramp * ramp * (3.0 - 2.0 * ramp)
    return schedule


def _unwrap_rotation_vector_trajectory(rotvec: np.ndarray) -> np.ndarray:
    values = np.asarray(rotvec, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("rotvec must be finite (N,3)")
    output = values.copy()
    for frame_index in range(1, len(output)):
        principal = values[frame_index]
        angle = float(np.linalg.norm(principal))
        if angle <= 1e-12:
            continue
        axis = principal / angle
        candidates = np.stack(
            (principal, principal + 2.0 * np.pi * axis, principal - 2.0 * np.pi * axis)
        )
        output[frame_index] = candidates[
            np.argmin(np.linalg.norm(candidates - output[frame_index - 1], axis=1))
        ]
    return output


def transition_palm_targets(
    baseline_body_q_mujoco: np.ndarray,
    base_quat_world_wxyz: np.ndarray,
    semantic_targets_local: np.ndarray,
    *,
    solver: Any,
    prefix_frames: int,
    ramp_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Causally introduce semantic palm targets from an executable trajectory."""

    baseline = np.asarray(baseline_body_q_mujoco, dtype=np.float64)
    base_quat = np.asarray(base_quat_world_wxyz, dtype=np.float64)
    targets = np.asarray(semantic_targets_local, dtype=np.float64)
    rows = len(baseline)
    if (
        baseline.shape != (rows, 29)
        or base_quat.shape != (rows, 4)
        or targets.shape != (rows, 14)
        or not all(np.isfinite(value).all() for value in (baseline, base_quat, targets))
    ):
        raise ValueError(
            "transition inputs must be finite aligned (N,29), (N,4), and (N,14)"
        )
    schedule = transition_schedule(
        rows, prefix_frames=prefix_frames, ramp_frames=ramp_frames
    )
    reference_targets = np.empty_like(targets)
    for frame_index in range(rows):
        palm_pose_body = np.asarray(
            solver.fk_from_mujoco_body(baseline[frame_index]), dtype=np.float64
        )
        local_from_body = gravity_local_rotation_from_body_quat(base_quat[frame_index])
        for position_start, rotation_start in ((0, 3), (7, 10)):
            reference_targets[frame_index, position_start : position_start + 3] = (
                local_from_body.apply(
                    palm_pose_body[position_start : position_start + 3]
                )
            )
            reference_targets[frame_index, rotation_start : rotation_start + 4] = (
                local_from_body
                * Rotation.from_quat(
                    palm_pose_body[rotation_start : rotation_start + 4],
                    scalar_first=True,
                )
            ).as_quat(scalar_first=True)
    blended = targets.copy()
    for position_start, rotation_start in ((0, 3), (7, 10)):
        blended[:, position_start : position_start + 3] = (
            reference_targets[:, position_start : position_start + 3]
            + schedule[:, None]
            * (
                targets[:, position_start : position_start + 3]
                - reference_targets[:, position_start : position_start + 3]
            )
        )
        reference_rotation = Rotation.from_quat(
            reference_targets[:, rotation_start : rotation_start + 4],
            scalar_first=True,
        )
        semantic_rotation = Rotation.from_quat(
            targets[:, rotation_start : rotation_start + 4], scalar_first=True
        )
        relative = _unwrap_rotation_vector_trajectory(
            (semantic_rotation * reference_rotation.inv()).as_rotvec()
        )
        for frame_index, alpha in enumerate(schedule):
            blended[frame_index, rotation_start : rotation_start + 4] = (
                Rotation.from_rotvec(alpha * relative[frame_index])
                * reference_rotation[frame_index]
            ).as_quat(scalar_first=True)
    return blended, schedule


def temporal_joint_bounds(
    previous_active_q: np.ndarray,
    max_joint_step_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a hard per-frame interval around the previous IK solution."""

    previous = np.asarray(previous_active_q, dtype=np.float64).reshape(-1)
    limit = float(max_joint_step_rad)
    if not np.isfinite(previous).all():
        raise ValueError("previous active joints must be finite")
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("max arm joint step must be finite and > 0")
    return previous - limit, previous + limit


def solve_arm_trajectory(
    baseline_body_q_mujoco: np.ndarray,
    base_quat_world_wxyz: np.ndarray,
    palm_targets_local: np.ndarray,
    *,
    solver: ArmIK,
    waist_stage_solver: ArmIK | None = None,
    waist_solver: ArmIK | None = None,
    iterations: int = 80,
    waist_iterations: int = 120,
    orientation_weight: float = 0.0,
    waist_activation_error_m: float = 0.005,
    waist_completion_error_m: float = 0.001,
    waist_delta_limits_rad: np.ndarray | None = None,
    waist_smoothing_frames: int = 13,
    waist_repair_passes: int = 1,
    max_arm_joint_step_rad: float = 0.08,
) -> np.ndarray:
    """Solve targets with arms, then yaw/roll, then all waist axes as needed."""

    baseline = np.asarray(baseline_body_q_mujoco, dtype=np.float64)
    base_quat = np.asarray(base_quat_world_wxyz, dtype=np.float64)
    targets = np.asarray(palm_targets_local, dtype=np.float64)
    rows = len(baseline)
    if baseline.shape != (rows, 29) or not np.isfinite(baseline).all():
        raise ValueError(f"baseline body_q must be finite (N, {29})")
    if base_quat.shape != (rows, 4) or not np.isfinite(base_quat).all():
        raise ValueError("base quaternion must be finite (N, 4)")
    if targets.shape != (rows, 14) or not np.isfinite(targets).all():
        raise ValueError("palm targets must be finite (N, 14)")
    if rows == 0:
        raise ValueError("IK reference requires at least one frame")

    arm_solver = solver
    waist_limits = np.full(3, np.deg2rad(8.0), dtype=np.float64)
    if waist_solver is not None:
        if waist_delta_limits_rad is not None:
            waist_limits = np.asarray(
                waist_delta_limits_rad,
                dtype=np.float64,
            ).reshape(3)
        if not np.isfinite(waist_limits).all() or np.any(waist_limits < 0.0):
            raise ValueError("waist delta limits must be finite and >= 0")
    activation_error = float(waist_activation_error_m)
    completion_error = float(waist_completion_error_m)
    orientation_regularization = float(orientation_weight)
    arm_step_limit = float(max_arm_joint_step_rad)
    if not np.isfinite(activation_error) or activation_error < 0.0:
        raise ValueError("waist activation error must be finite and >= 0")
    if not np.isfinite(completion_error) or completion_error < 0.0:
        raise ValueError("waist completion error must be finite and >= 0")
    if not np.isfinite(orientation_regularization) or orientation_regularization < 0.0:
        raise ValueError("orientation weight must be finite and >= 0")
    if not np.isfinite(arm_step_limit) or arm_step_limit <= 0.0:
        raise ValueError("max arm joint step must be finite and > 0")
    arm_reference = baseline.copy()
    arm_previous = baseline[0].copy()
    arm_active_indices = np.asarray(
        getattr(
            arm_solver,
            "active_body_mujoco_indices",
            np.arange(15, 29, dtype=np.int64),
        ),
        dtype=np.int64,
    ).reshape(-1)
    if (
        len(arm_active_indices) == 0
        or np.any(arm_active_indices < 15)
        or np.any(arm_active_indices >= 29)
    ):
        raise ValueError("arm solver active indices must select body joints 15:29")
    arm_errors = np.empty((rows, 2), dtype=np.float64)
    for frame_index in range(rows):
        seed = baseline[frame_index].copy()
        seed[15:29] = arm_previous[15:29]
        target_body = poses_gravity_local_to_body(
            targets[frame_index],
            base_quat[frame_index],
        )
        temporal_lower = None
        temporal_upper = None
        if frame_index > 0:
            temporal_lower, temporal_upper = temporal_joint_bounds(
                arm_previous[arm_active_indices],
                arm_step_limit,
            )
        solved, diagnostics = arm_solver.solve(
            seed,
            target_body,
            iterations=max(1, int(iterations)),
            orientation_weight=orientation_regularization,
            rest_body_q_mujoco=baseline[frame_index],
            active_lower=temporal_lower,
            active_upper=temporal_upper,
        )
        solved = np.asarray(solved, dtype=np.float64).reshape(29)
        if not np.isfinite(solved).all():
            raise RuntimeError(f"IK returned non-finite joints at frame {frame_index}")
        frame_errors = np.array(
            [
                float(diagnostics["position_error_left_m"]),
                float(diagnostics["position_error_right_m"]),
            ],
            dtype=np.float64,
        )
        solved[:15] = baseline[frame_index, :15]
        arm_reference[frame_index] = solved
        arm_previous = solved
        arm_errors[frame_index] = frame_errors

    def solve_waist_trajectory(
        active_solver: ArmIK,
        initial_reference: np.ndarray,
        rest_reference: np.ndarray,
        initial_errors: np.ndarray,
        solve_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        active_indices = np.asarray(
            active_solver.active_body_mujoco_indices,
            dtype=np.int64,
        )
        active_waist_mask = active_indices < 15
        active_waist_indices = active_indices[active_waist_mask]
        inactive_waist_indices = np.setdiff1d(
            np.arange(12, 15, dtype=np.int64),
            active_waist_indices,
        )
        trajectory = np.asarray(initial_reference, dtype=np.float64).copy()
        trajectory_errors = np.asarray(initial_errors, dtype=np.float64).copy()
        selected_frames = np.asarray(solve_mask, dtype=bool).reshape(rows)
        if trajectory_errors.shape != (rows, 2):
            raise ValueError("initial waist errors must have shape (N, 2)")
        for frame_index in np.flatnonzero(selected_frames):
            target_body = poses_gravity_local_to_body(
                targets[frame_index],
                base_quat[frame_index],
            )
            active_lower = np.asarray(active_solver.lower, dtype=np.float64).copy()
            active_upper = np.asarray(active_solver.upper, dtype=np.float64).copy()
            active_lower[active_waist_mask] = np.maximum(
                active_lower[active_waist_mask],
                baseline[frame_index, active_waist_indices]
                - waist_limits[active_waist_indices - 12],
            )
            active_upper[active_waist_mask] = np.minimum(
                active_upper[active_waist_mask],
                baseline[frame_index, active_waist_indices]
                + waist_limits[active_waist_indices - 12],
            )
            candidate_seed = np.asarray(
                initial_reference[frame_index],
                dtype=np.float64,
            ).copy()
            candidate_seed[:12] = baseline[frame_index, :12]
            candidate_seed[inactive_waist_indices] = baseline[
                frame_index, inactive_waist_indices
            ]
            candidate_seed[active_indices] = np.clip(
                candidate_seed[active_indices],
                active_lower,
                active_upper,
            )
            candidate, candidate_diagnostics = active_solver.solve(
                candidate_seed,
                target_body,
                iterations=max(1, int(waist_iterations)),
                orientation_weight=orientation_regularization,
                rest_body_q_mujoco=rest_reference[frame_index],
                active_lower=active_lower,
                active_upper=active_upper,
            )
            candidate = np.asarray(candidate, dtype=np.float64).reshape(29)
            candidate[:12] = baseline[frame_index, :12]
            candidate[inactive_waist_indices] = baseline[
                frame_index, inactive_waist_indices
            ]
            trajectory[frame_index] = candidate
            trajectory_errors[frame_index] = np.array(
                [
                    float(candidate_diagnostics["position_error_left_m"]),
                    float(candidate_diagnostics["position_error_right_m"]),
                ],
                dtype=np.float64,
            )
        return trajectory, trajectory_errors

    reference = arm_reference.copy()
    errors = arm_errors.copy()
    waist_trajectory_selected = False
    stage_errors: np.ndarray | None = None
    if waist_solver is not None and float(np.max(arm_errors)) > activation_error:
        arm_needs_assistance = np.max(arm_errors, axis=1) > activation_error
        completion_seed = baseline
        completion_rest = baseline
        if waist_stage_solver is not None:
            stage_reference, stage_errors = solve_waist_trajectory(
                waist_stage_solver,
                arm_reference,
                baseline,
                arm_errors,
                arm_needs_assistance,
            )
            completion_seed = stage_reference
            completion_rest = stage_reference
        if stage_errors is not None:
            needs_pitch = arm_needs_assistance & (
                np.max(stage_errors, axis=1) > completion_error
            )
            full_reference = stage_reference
            full_errors = stage_errors
            if np.any(needs_pitch):
                full_reference, full_errors = solve_waist_trajectory(
                    waist_solver,
                    stage_reference,
                    stage_reference,
                    stage_errors,
                    needs_pitch,
                )
            waist_reference = stage_reference.copy()
            waist_errors = stage_errors.copy()
            use_full = needs_pitch & (
                np.max(full_errors, axis=1)
                < np.max(stage_errors, axis=1)
            )
            waist_reference[use_full] = full_reference[use_full]
            waist_errors[use_full] = full_errors[use_full]
        else:
            waist_reference, waist_errors = solve_waist_trajectory(
                waist_solver,
                completion_seed,
                completion_rest,
                arm_errors,
                arm_needs_assistance,
            )

        arm_frame_max = np.max(arm_errors, axis=1)
        waist_frame_max = np.max(waist_errors, axis=1)
        assistance_mask = (
            (arm_frame_max > activation_error)
            & (waist_frame_max < arm_frame_max - 1e-4)
            & (np.max(waist_errors - arm_errors, axis=1) <= 0.005)
        )
        reference[assistance_mask] = waist_reference[assistance_mask]
        errors[assistance_mask] = waist_errors[assistance_mask]
        waist_trajectory_selected = bool(np.any(assistance_mask))

    repair_passes = max(0, int(waist_repair_passes))

    def smooth_waist_and_resolve_arms(
        candidate_reference: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        waist_residual = candidate_reference[:, 12:15] - baseline[:, 12:15]
        smoothing_window = min(
            max(0, int(waist_smoothing_frames)),
            rows if rows % 2 else rows - 1,
        )
        if smoothing_window >= 3:
            if smoothing_window % 2 == 0:
                smoothing_window -= 1
            waist_residual = savgol_filter(
                waist_residual,
                window_length=smoothing_window,
                polyorder=min(2, smoothing_window - 1),
                axis=0,
                mode="interp",
            )
        waist_residual = np.clip(
            waist_residual,
            -waist_limits,
            waist_limits,
        )
        continuous_reference = baseline.copy()
        continuous_reference[:, 12:15] += waist_residual
        continuous_errors = np.empty((rows, 2), dtype=np.float64)
        previous_arms = candidate_reference[0, 15:29].copy()
        for frame_index in range(rows):
            seed = continuous_reference[frame_index].copy()
            seed[15:29] = previous_arms
            target_body = poses_gravity_local_to_body(
                targets[frame_index],
                base_quat[frame_index],
            )
            temporal_lower = None
            temporal_upper = None
            if frame_index > 0:
                temporal_lower, temporal_upper = temporal_joint_bounds(
                    previous_arms,
                    arm_step_limit,
                )
            solved, diagnostics = arm_solver.solve(
                seed,
                target_body,
                iterations=max(1, int(iterations)),
                orientation_weight=orientation_regularization,
                rest_body_q_mujoco=baseline[frame_index],
                active_lower=temporal_lower,
                active_upper=temporal_upper,
            )
            solved = np.asarray(solved, dtype=np.float64).reshape(29)
            solved[:15] = continuous_reference[frame_index, :15]
            continuous_reference[frame_index] = solved
            previous_arms = solved[15:29].copy()
            continuous_errors[frame_index] = [
                float(diagnostics["position_error_left_m"]),
                float(diagnostics["position_error_right_m"]),
            ]
        return continuous_reference, continuous_errors

    if waist_trajectory_selected:
        reference, errors = smooth_waist_and_resolve_arms(reference)
        for _repair_pass in range(repair_passes):
            current_frame_max = np.max(errors, axis=1)
            if float(np.max(current_frame_max)) <= activation_error:
                break
            repair_reference, repair_errors = solve_waist_trajectory(
                waist_solver,
                reference,
                baseline,
                errors,
                current_frame_max > activation_error,
            )
            repair_frame_max = np.max(repair_errors, axis=1)
            repair_mask = (
                (current_frame_max > activation_error)
                & (repair_frame_max < current_frame_max - 1e-4)
                & (np.max(repair_errors - errors, axis=1) <= 0.005)
            )
            if not np.any(repair_mask):
                break
            candidate = reference.copy()
            candidate[repair_mask] = repair_reference[repair_mask]
            candidate_reference, candidate_errors = (
                smooth_waist_and_resolve_arms(candidate)
            )
            current_max = float(np.max(errors))
            candidate_max = float(np.max(candidate_errors))
            hand_max_degradation = float(
                np.max(
                    np.max(candidate_errors, axis=0)
                    - np.max(errors, axis=0)
                )
            )
            if (
                candidate_max >= current_max - 1e-5
                or hand_max_degradation > 0.005
            ):
                break
            reference = candidate_reference
            errors = candidate_errors

    return reference.astype(np.float32)
