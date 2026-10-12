"""Bound-constrained palm IK with arm-first, optional waist assistance."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ArmIKConfig:
    iterations: int = 24
    damping: float = 1e-4
    max_step_norm_rad: float = .08
    max_joint_step_rad: float = .04
    posture_weight: float = 2e-4
    wrist_posture_weight: float = 2e-3
    waist_posture_weight: float = 5e-3
    orientation_weight: float = .02
    stop_tolerance_m: float = 5e-4


class ArmIK:
    """Solve in pelvis-body coordinates; active waist indices precede arms."""

    def __init__(self, kinematics, config=None, *, waist_indices=()):
        self.kinematics = kinematics
        self.config = config or ArmIKConfig()
        if len(set(waist_indices)) != len(waist_indices) or not set(waist_indices) <= {12, 13, 14}:
            raise ValueError('waist indices must select yaw, roll or pitch')
        self.active_body_mujoco_indices = np.array((*waist_indices, *range(15, 29)))
        self.lower = kinematics.lower[self.active_body_mujoco_indices]
        self.upper = kinematics.upper[self.active_body_mujoco_indices]

    def fk_from_mujoco_body(self, joints):
        return self.kinematics.evaluate(joints)[0][:2].reshape(14)

    def solve(self, body_q_mujoco, target_eef_pelvis, *, iterations=None,
              orientation_weight=None, rest_body_q_mujoco=None,
              active_lower=None, active_upper=None):
        q = np.asarray(body_q_mujoco, dtype=np.float64).reshape(29).copy()
        target = np.asarray(target_eef_pelvis, dtype=np.float64).reshape(2, 7)
        if not np.isfinite(q).all() or not np.isfinite(target).all():
            raise ValueError('joints and targets must be finite')
        rest = q.copy() if rest_body_q_mujoco is None else np.asarray(
            rest_body_q_mujoco, dtype=np.float64).reshape(29)
        cfg = self.config
        active = self.active_body_mujoco_indices
        count = cfg.iterations if iterations is None else max(1, int(iterations))
        ori_weight = cfg.orientation_weight if orientation_weight is None else max(0., float(orientation_weight))
        weights = np.full(len(active), cfg.posture_weight)
        weights[np.isin(active, [19, 20, 21, 26, 27, 28])] = cfg.wrist_posture_weight
        weights[active < 15] = cfg.waist_posture_weight
        lower = np.minimum(self.lower, q[active])
        upper = np.maximum(self.upper, q[active])
        if active_lower is not None:
            lower = np.maximum(lower, np.asarray(active_lower).reshape(len(active)))
        if active_upper is not None:
            upper = np.minimum(upper, np.asarray(active_upper).reshape(len(active)))
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
            raise ValueError('active IK bounds must be finite and ordered')
        if active_lower is not None or active_upper is not None:
            q[active] = np.clip(q[active], lower, upper)
        free = upper - lower > 1e-12
        indices, lower, upper, weights = active[free], lower[free], upper[free], weights[free]
        for _ in range(count):
            poses, jacobian = self.kinematics.evaluate(q)
            errors, jacobians, norms = [], [], []
            for side in range(2):
                position_error = target[side, :3] - poses[side, :3]
                norms.append(float(np.linalg.norm(position_error)))
                errors.append(position_error)
                jacobians.append(jacobian[side*6:side*6+3, indices])
                if ori_weight > 0:
                    desired = Rotation.from_quat(target[side, [4, 5, 6, 3]])
                    actual = Rotation.from_quat(poses[side, [4, 5, 6, 3]])
                    errors.append(np.sqrt(ori_weight) * (desired * actual.inv()).as_rotvec())
                    jacobians.append(np.sqrt(ori_weight) * jacobian[side*6+3:side*6+6, indices])
            if max(norms) <= cfg.stop_tolerance_m or indices.size == 0:
                break
            task_jacobian, task_error = np.vstack(jacobians), np.concatenate(errors)
            available = np.ones(len(indices), dtype=bool)
            delta = np.zeros(len(indices))
            for _ in range(len(indices) + 1):
                if not np.any(available):
                    break
                sqrt_posture = np.sqrt(weights[available])
                lhs = np.vstack([task_jacobian[:, available], np.diag(sqrt_posture)])
                rhs = np.concatenate([task_error, sqrt_posture * (rest[indices[available]] - q[indices[available]])])
                normal = lhs.T @ lhs + cfg.damping * np.eye(np.count_nonzero(available))
                try:
                    available_delta = np.linalg.solve(normal, lhs.T @ rhs)
                except np.linalg.LinAlgError:
                    available_delta = np.linalg.lstsq(normal, lhs.T @ rhs, rcond=None)[0]
                proposal = np.zeros(len(indices))
                proposal[available] = available_delta
                saturated = available & (((q[indices] <= lower + 1e-9) & (proposal < 0)) |
                                         ((q[indices] >= upper - 1e-9) & (proposal > 0)))
                if not np.any(saturated):
                    delta = proposal
                    break
                available[saturated] = False
            norm = float(np.linalg.norm(delta))
            if norm > cfg.max_step_norm_rad:
                delta *= cfg.max_step_norm_rad / norm
            delta = np.clip(delta, -cfg.max_joint_step_rad, cfg.max_joint_step_rad)
            q[indices] = np.clip(q[indices] + delta, lower, upper)
        achieved = self.kinematics.evaluate(q)[0][:2]
        errors = np.linalg.norm(target[:, :3] - achieved[:, :3], axis=1)
        return q.astype(np.float32), {
            'position_error_left_m': float(errors[0]),
            'position_error_right_m': float(errors[1]),
        }
