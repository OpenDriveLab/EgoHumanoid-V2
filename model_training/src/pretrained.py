"""Initialize a 76-dimensional action head from pi0.5 base weights."""

import logging

from safetensors import safe_open

ACTION_HEAD_KEYS = {
    "action_in_proj.weight",
    "action_out_proj.weight",
    "action_out_proj.bias",
}


def load_pretrained(model, path):
    expected = model.state_dict()
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        state = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}  # noqa: SIM118
        for alias, retained in (checkpoint.metadata() or {}).items():
            if alias in expected and retained in state:
                state[alias] = state[retained]
    unexpected = state.keys() - expected.keys()
    missing = expected.keys() - state.keys()
    mismatched = {
        key
        for key in state.keys() & expected.keys()
        if state[key].shape != expected[key].shape
    }
    if unexpected or missing - ACTION_HEAD_KEYS or mismatched - ACTION_HEAD_KEYS:
        raise ValueError(
            f"Incompatible pretrained weights: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}, shape={sorted(mismatched)}"
        )
    compatible = {key: value for key, value in state.items() if key not in mismatched}
    result = model.load_state_dict(compatible, strict=False)
    if set(result.missing_keys) - ACTION_HEAD_KEYS or result.unexpected_keys:
        raise ValueError("Incomplete pretrained backbone load")
    logging.getLogger(__name__).info(
        "Loaded pretrained backbone; initialized action parameters: %s",
        sorted(missing | mismatched),
    )
