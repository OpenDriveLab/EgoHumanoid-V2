"""Rollout-conditioned token optimization in linearized task space."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from align.refinement.decoder import JointTargetDecoder


def _bounded_delta(
    parameter: torch.Tensor,
    *,
    max_element: float,
    max_row_norm: float,
) -> torch.Tensor:
    delta = float(max_element) * torch.tanh(parameter)
    norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    scale = torch.clamp(float(max_row_norm) / torch.clamp(norm, min=1e-12), max=1.0)
    return delta * scale


def optimize_tokens_for_linearized_task(
    decoder: JointTargetDecoder,
    observations: np.ndarray,
    *,
    initial_tokens: np.ndarray,
    baseline_joint_target: np.ndarray,
    task_jacobians: np.ndarray,
    task_residuals: np.ndarray,
    task_predictor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    joint_preservation_weights: np.ndarray,
    joint_velocity_weights: np.ndarray | None = None,
    joint_acceleration_weights: np.ndarray | None = None,
    joint_teacher_target: np.ndarray | None = None,
    joint_teacher_weights: np.ndarray | None = None,
    joint_teacher_weight: float = 0.0,
    ridge_weight: float = 1e-5,
    temporal_weight: float = 1e-5,
    max_element_delta: float = 0.1,
    max_row_norm_delta: float = 0.4,
    iterations: int = 200,
    learning_rate: float = 0.02,
    device: torch.device,
) -> np.ndarray:
    """Fit tokens directly in a linearized physical task space.

    This objective permits decoder joint combinations that are equivalent for
    the endpoint task. Per-joint preservation weights independently protect
    legs, waist, and other non-task joints.
    """

    base = np.asarray(observations, dtype=np.float32)
    initial = np.asarray(initial_tokens, dtype=np.float32)
    baseline = np.asarray(baseline_joint_target, dtype=np.float32)
    jacobians = np.asarray(task_jacobians, dtype=np.float32)
    residuals = np.asarray(task_residuals, dtype=np.float32)
    preservation = np.asarray(joint_preservation_weights, dtype=np.float32)
    velocity_weights = (
        np.zeros_like(preservation)
        if joint_velocity_weights is None
        else np.asarray(joint_velocity_weights, dtype=np.float32)
    )
    acceleration_weights = (
        np.zeros_like(preservation)
        if joint_acceleration_weights is None
        else np.asarray(joint_acceleration_weights, dtype=np.float32)
    )
    teacher = (
        None
        if joint_teacher_target is None
        else np.asarray(joint_teacher_target, dtype=np.float32)
    )
    teacher_weights = (
        None
        if joint_teacher_weights is None
        else np.asarray(joint_teacher_weights, dtype=np.float32)
    )
    if (
        base.ndim != 2
        or initial.ndim != 2
        or baseline.ndim != 2
        or jacobians.ndim != 3
        or residuals.ndim != 2
        or preservation.ndim != 1
        or len(base) != len(initial)
        or len(baseline) != len(initial)
        or jacobians.shape[:2] != residuals.shape
        or jacobians.shape[0] != len(initial)
        or jacobians.shape[2] != baseline.shape[1]
        or preservation.shape[0] != baseline.shape[1]
        or velocity_weights.shape != preservation.shape
        or acceleration_weights.shape != preservation.shape
        or initial.shape[1] > base.shape[1]
    ):
        raise ValueError("linearized task inputs must have aligned dimensions")
    if (teacher is None) != (teacher_weights is None):
        raise ValueError("joint teacher target and weights must be supplied together")
    if teacher is not None and (
        teacher.shape != baseline.shape
        or teacher_weights.shape != (baseline.shape[1],)
    ):
        raise ValueError("joint teacher inputs must align with the joint target")
    if not all(
        np.isfinite(value).all()
        for value in (
            base,
            initial,
            baseline,
            jacobians,
            residuals,
            preservation,
            velocity_weights,
            acceleration_weights,
        )
    ):
        raise ValueError("linearized task inputs must be finite")
    if (
        np.any(preservation < 0.0)
        or np.any(velocity_weights < 0.0)
        or np.any(acceleration_weights < 0.0)
    ):
        raise ValueError("joint preservation and temporal weights must be >= 0")
    if teacher is not None and (
        not np.isfinite(teacher).all()
        or not np.isfinite(teacher_weights).all()
        or np.any(teacher_weights < 0.0)
    ):
        raise ValueError("joint teacher inputs must be finite and nonnegative")
    if not np.isfinite(joint_teacher_weight) or joint_teacher_weight < 0.0:
        raise ValueError("joint_teacher_weight must be finite and >= 0")
    if joint_teacher_weight > 0.0 and teacher is None:
        raise ValueError("positive joint_teacher_weight requires a joint teacher")
    positive = (max_element_delta, max_row_norm_delta, learning_rate)
    if any(not np.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("bounds and learning rate must be finite and > 0")
    if (
        not np.isfinite(ridge_weight)
        or not np.isfinite(temporal_weight)
        or ridge_weight < 0.0
        or temporal_weight < 0.0
        or int(iterations) <= 0
    ):
        raise ValueError("regularizers must be >= 0 and iterations must be > 0")

    observation = torch.as_tensor(base, device=device)
    initial_token = torch.as_tensor(initial, device=device)
    baseline_joint = torch.as_tensor(baseline, device=device)
    task_jacobian = torch.as_tensor(jacobians, device=device)
    task_residual = torch.as_tensor(residuals, device=device)
    preserve_weight = torch.as_tensor(preservation, device=device)
    joint_velocity_weight = torch.as_tensor(velocity_weights, device=device)
    joint_acceleration_weight = torch.as_tensor(
        acceleration_weights, device=device
    )
    teacher_joint = (
        None if teacher is None else torch.as_tensor(teacher, device=device)
    )
    teacher_joint_weight = (
        None
        if teacher_weights is None
        else torch.as_tensor(teacher_weights, device=device)
    )
    token_dim = int(initial.shape[1])
    def bounded_parameter_delta(parameter: torch.Tensor) -> torch.Tensor:
        return _bounded_delta(
            parameter,
            max_element=max_element_delta,
            max_row_norm=max_row_norm_delta,
        )

    def with_token(token: torch.Tensor) -> torch.Tensor:
        value = observation.clone()
        value[:, :token_dim] = token
        return value

    with torch.no_grad():
        decoded_baseline = decoder.joint_target(with_token(initial_token))
    if decoded_baseline.shape != baseline_joint.shape:
        raise ValueError("decoder output must match baseline_joint_target")

    parameter = torch.zeros_like(initial_token, requires_grad=True)
    optimizer = torch.optim.Adam([parameter], lr=float(learning_rate))
    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        delta = bounded_parameter_delta(parameter)
        output = decoder.joint_target(with_token(initial_token + delta))
        joint_delta = output - baseline_joint
        predicted_task_delta = (
            torch.einsum("ntj,nj->nt", task_jacobian, joint_delta)
            if task_predictor is None else task_predictor(joint_delta)
        )
        if predicted_task_delta.shape != task_residual.shape:
            raise ValueError("response prediction must match task residual dimensions")
        task_error = predicted_task_delta - task_residual
        task_loss = torch.mean(task_error**2)
        preservation_loss = torch.mean(
            preserve_weight[None, :] * joint_delta**2
        )
        joint_velocity_loss = (
            torch.mean(
                joint_velocity_weight[None, :]
                * (joint_delta[1:] - joint_delta[:-1]) ** 2
            )
            if len(joint_delta) > 1
            else joint_delta.new_zeros(())
        )
        joint_acceleration_loss = (
            torch.mean(
                joint_acceleration_weight[None, :]
                * (
                    joint_delta[2:]
                    - 2.0 * joint_delta[1:-1]
                    + joint_delta[:-2]
                )
                ** 2
            )
            if len(joint_delta) > 2
            else joint_delta.new_zeros(())
        )
        if teacher_joint is None or float(joint_teacher_weight) == 0.0:
            teacher_loss = output.new_zeros(())
        else:
            teacher_error = output - teacher_joint
            teacher_loss = torch.sum(
                teacher_joint_weight[None, :] * teacher_error**2
            ) / torch.clamp(
                float(len(output)) * torch.sum(teacher_joint_weight), min=1e-12
            )
        ridge_loss = torch.mean(delta**2)
        temporal_loss = (
            torch.mean((delta[1:] - delta[:-1]) ** 2)
            if len(delta) > 1
            else delta.new_zeros(())
        )
        loss = (
            task_loss
            + preservation_loss
            + joint_velocity_loss
            + joint_acceleration_loss
            + float(joint_teacher_weight) * teacher_loss
            + float(ridge_weight) * ridge_loss
            + float(temporal_weight) * temporal_loss
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_delta = bounded_parameter_delta(parameter)
        tokens = (initial_token + final_delta).cpu().numpy().astype(np.float32)
    return tokens
