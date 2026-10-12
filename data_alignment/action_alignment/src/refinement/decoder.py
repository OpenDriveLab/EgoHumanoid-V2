"""Differentiable SONIC release decoder in MuJoCo joint order."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


ISAACLAB_TO_MUJOCO = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int64,
)


DEFAULT_ANGLES_MUJOCO = np.array(
    [
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


def g1_action_scale() -> np.ndarray:
    natural_freq = 10.0 * 2.0 * math.pi
    stiffness_5020 = 0.003609725 * natural_freq * natural_freq
    stiffness_7520_14 = 0.010177520 * natural_freq * natural_freq
    stiffness_7520_22 = 0.025101925 * natural_freq * natural_freq
    stiffness_4010 = 0.00425 * natural_freq * natural_freq
    scale_5020 = 0.25 * 25.0 / stiffness_5020
    scale_7520_14 = 0.25 * 88.0 / stiffness_7520_14
    scale_7520_22 = 0.25 * 139.0 / stiffness_7520_22
    scale_4010 = 0.25 * 5.0 / stiffness_4010
    return np.array(
        [
            scale_7520_22,
            scale_7520_22,
            scale_7520_14,
            scale_7520_22,
            scale_5020,
            scale_5020,
            scale_7520_22,
            scale_7520_22,
            scale_7520_14,
            scale_7520_22,
            scale_5020,
            scale_5020,
            scale_7520_14,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_4010,
            scale_4010,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_5020,
            scale_4010,
            scale_4010,
        ],
        dtype=np.float32,
    )


G1_ACTION_SCALE_MUJOCO = g1_action_scale()


def rollout_observations(capture, tokens) -> np.ndarray:
    """Reconstruct the release decoder's causal ten-frame observation history.

    Capture joints and previous joint targets are in MuJoCo order. Prefixes
    outside the recording are zero-padded, matching the offline decoder input;
    those first ten frames must remain frozen during token refinement.
    """
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 64 or not len(tokens) or not np.isfinite(tokens).all():
        raise ValueError('expected finite nonempty (N,64) motion tokens')
    arrays = {}
    for name, width in (('body_q', 29), ('body_dq', 29), ('last_action', 29),
                        ('base_ang_vel', 3), ('base_quat', 4)):
        value = np.asarray(capture[name], dtype=np.float32)
        if value.shape != (len(tokens), width) or not np.isfinite(value).all():
            raise ValueError(f'{name} must be finite and match token rows')
        arrays[name] = value
    order = np.argsort(ISAACLAB_TO_MUJOCO)
    q = (arrays['body_q'] - DEFAULT_ANGLES_MUJOCO)[:, order]
    dq = arrays['body_dq'][:, order]
    action = ((arrays['last_action'] - DEFAULT_ANGLES_MUJOCO) / G1_ACTION_SCALE_MUJOCO)[:, order]
    gravity = Rotation.from_quat(arrays['base_quat'][:, [1, 2, 3, 0]]).inv().apply([0, 0, -1]).astype(np.float32)
    histories = []
    indices = np.arange(len(tokens))[:, None] + np.arange(10)
    for value in (arrays['base_ang_vel'], q, dq, action, gravity):
        padded = np.pad(value, ((9, 0), (0, 0)))
        histories.append(padded[indices].reshape(len(tokens), -1))
    return np.concatenate((tokens, *histories), axis=1)


class JointTargetDecoder(Protocol):
    def joint_target(self, observation: torch.Tensor) -> torch.Tensor: ...


class TorchOnnxControlDecoder:
    """Differentiable copy of the released G1 decoder ONNX MLP."""

    def __init__(
        self,
        layers: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        device: torch.device,
    ) -> None:
        self.layers = [
            (
                weight.to(device=device, dtype=torch.float32),
                bias.to(device=device, dtype=torch.float32),
            )
            for weight, bias in layers
        ]
        self.order = torch.as_tensor(
            ISAACLAB_TO_MUJOCO, dtype=torch.long, device=device
        )
        self.default = torch.as_tensor(
            DEFAULT_ANGLES_MUJOCO, dtype=torch.float32, device=device
        )
        self.scale = torch.as_tensor(
            G1_ACTION_SCALE_MUJOCO, dtype=torch.float32, device=device
        )

    def action(self, observation: torch.Tensor) -> torch.Tensor:
        value = observation
        for index, (weight, bias) in enumerate(self.layers):
            value = F.linear(value, weight, bias)
            if index + 1 < len(self.layers):
                value = F.silu(value)
        return value

    def joint_target(self, observation: torch.Tensor) -> torch.Tensor:
        action = self.action(observation)
        return self.default + action[..., self.order] * self.scale


def load_torch_decoder_from_onnx(
    model_path: Path,
    *,
    device: torch.device,
) -> TorchOnnxControlDecoder:
    """Load the seven released linear layers without the trainer checkpoint."""

    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(Path(model_path).expanduser().resolve()))
    initializers = {
        value.name: numpy_helper.to_array(value).copy()
        for value in model.graph.initializer
    }
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index, weight_index in zip((0, 2, 4, 6, 8, 10, 12), range(136, 143)):
        weight_name = f"onnx::MatMul_{weight_index}"
        bias_name = f"module.decoders.g1_dyn.module.{layer_index}.bias"
        if weight_name not in initializers or bias_name not in initializers:
            raise ValueError(
                f"{model_path}: missing released decoder layer {layer_index}"
            )
        weight = torch.from_numpy(initializers[weight_name].T)
        bias = torch.from_numpy(initializers[bias_name])
        layers.append((weight, bias))
    return TorchOnnxControlDecoder(layers, device=device)
