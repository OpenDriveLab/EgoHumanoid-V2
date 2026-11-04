"""Composite generated background and arm layers while preserving source gloves.

Used by render.py; operates on image arrays and does not infer robot geometry.
"""

from __future__ import annotations

import numpy as np


def composite_locked_frame(
    raw_rgb: np.ndarray,
    inpainted_rgb: np.ndarray,
    removal_mask: np.ndarray,
    glove_mask: np.ndarray,
    mesh_premultiplied_rgb: np.ndarray,
    mesh_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use generated pixels only in the hole, then lock mesh and gloves."""
    raw = np.asarray(raw_rgb)
    generated = np.asarray(inpainted_rgb)
    mesh = np.asarray(mesh_premultiplied_rgb)
    removal = np.asarray(removal_mask, dtype=bool)
    gloves = np.asarray(glove_mask, dtype=bool)
    alpha = np.asarray(mesh_alpha, dtype=np.float32).copy()
    if raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.uint8:
        raise ValueError("raw_rgb must be uint8 with shape (H, W, 3)")
    if generated.shape != raw.shape or generated.dtype != np.uint8:
        raise ValueError("inpainted_rgb must match raw_rgb")
    if mesh.shape != raw.shape or mesh.dtype != np.uint8:
        raise ValueError("mesh RGB must match raw_rgb")
    if removal.shape != raw.shape[:2] or gloves.shape != raw.shape[:2]:
        raise ValueError("removal and glove masks must match the RGB image")
    if alpha.shape != (2,) + raw.shape[:2] or not np.isfinite(alpha).all():
        raise ValueError("mesh_alpha must be finite with shape (2, H, W)")
    np.clip(alpha, 0.0, 1.0, out=alpha)
    output = raw.copy()
    hole = removal & ~gloves
    output[hole] = generated[hole]
    alpha[:, gloves] = 0.0
    combined_alpha = np.max(alpha, axis=0)
    layer = mesh.astype(np.float32)
    layer[combined_alpha <= 0.0] = 0.0
    layer[gloves] = 0.0
    output = np.clip(
        np.rint(output.astype(np.float32) * (1.0 - combined_alpha[..., None]) + layer),
        0,
        255,
    ).astype(np.uint8)
    output[gloves] = raw[gloves]
    return (output, alpha >= 0.2)
