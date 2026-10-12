"""BrainCo command conversion at the STARK driver boundary; no device I/O."""

import numpy as np

BRAINCO_COMMAND_MAX = np.asarray(
    [0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
)
BRAINCO_TO_STARK_SCALE = np.asarray(
    [2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
)
BRAINCO_STARK_COMMAND_MAX = np.ones(6, dtype=np.float32)


def brainco_command_to_stark(value) -> np.ndarray:
    """Convert project BrainCo units to the STARK driver's 0..1 percentages."""
    command = np.asarray(value, dtype=np.float32).reshape(-1)
    if command.size < 6:
        raise ValueError("BrainCo command must contain at least six values")
    command = command[:6]
    if not np.isfinite(command).all():
        raise ValueError("BrainCo command contains non-finite values")
    project_command = np.clip(command, 0.0, BRAINCO_COMMAND_MAX)
    return np.clip(
        project_command * BRAINCO_TO_STARK_SCALE,
        0.0,
        BRAINCO_STARK_COMMAND_MAX,
    ).astype(np.float32, copy=False)
