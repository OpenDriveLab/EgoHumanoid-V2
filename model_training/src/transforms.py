"""Shared training and inference schema: 43 state values and 76 action values."""

import dataclasses

import numpy as np
from openpi import transforms

STATE_DIM = 43
ACTION_DIM = 76
ACTION_KEYS = (
    "action.motion_token",
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
)


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclasses.dataclass(frozen=True)
class Repack(transforms.DataTransformFn):
    def __call__(self, data):
        image_key = "observation.images.ego_view"
        if image_key not in data:
            image_key = "observation.images.gopro"
        actions = [array(data[key]) for key in ACTION_KEYS]
        for action, size in zip(actions, (64, 6, 6), strict=True):
            if action.shape[-1] != size or not np.isfinite(action).all():
                raise ValueError(f"Expected finite action channels of width {size}")
        result = {
            "image": data[image_key],
            "state": data["observation.state"],
            "actions": np.concatenate(actions, axis=-1),
        }
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class Inputs(transforms.DataTransformFn):
    def __call__(self, data):
        state = array(data["state"]).astype(np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError("Expected a finite 43-dimensional G1 state")
        image = array(data["image"])
        if image.ndim != 3:
            raise ValueError("Expected an RGB image")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        if image.shape[-1] != 3:
            raise ValueError("Expected three RGB channels")
        if np.issubdtype(image.dtype, np.floating):
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            raise ValueError("Image must be uint8 or floating point in [0, 1]")
        result = {
            "image": {"base_0_rgb": image},
            "image_mask": {"base_0_rgb": np.True_},
            "state": transforms.pad_to_dim(state, ACTION_DIM),
        }
        if "actions" in data:
            result["actions"] = array(data["actions"]).astype(np.float32)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class Outputs(transforms.DataTransformFn):
    def __call__(self, data):
        return {"actions": array(data["actions"])[..., :ACTION_DIM]}
