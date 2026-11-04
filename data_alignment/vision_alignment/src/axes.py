"""Refine 2D wrist and arm-entry anchors from removal and glove masks.

Used by prepare.py after joint projection and RGB cuff alignment. Roots here
are image-space arm-entry points, not the robot base or pelvis pose.
"""

from __future__ import annotations

import cv2
import numpy as np


def smooth_track(values: np.ndarray, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = values.copy()
    for index in range(len(values)):
        lo, hi = (max(0, index - radius), min(len(values), index + radius + 1))
        for side in range(values.shape[1]):
            window = values[lo:hi, side]
            centre = np.median(window, axis=0)
            distance = np.linalg.norm(window - centre, axis=1)
            keep = distance <= max(12.0, 3.0 * float(np.median(distance)))
            output[index, side] = np.mean(window[keep], axis=0)
    return output


def fit_frame_axes(
    removal: np.ndarray,
    gloves: np.ndarray,
    prior_cuffs: np.ndarray,
    prior_roots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return visual cuffs and boundary roots for both body sides."""
    removal = np.asarray(removal, dtype=bool)
    gloves = np.asarray(gloves, dtype=bool)
    height, width = removal.shape
    cuffs = np.asarray(prior_cuffs, dtype=np.float64).copy()
    roots = np.asarray(prior_roots, dtype=np.float64).copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(removal.astype(np.uint8), 8)
    glove_count, glove_labels, glove_stats, _ = cv2.connectedComponentsWithStats(
        gloves.astype(np.uint8), 8
    )
    side_gloves = np.zeros((2, height, width), dtype=bool)
    for label in range(1, glove_count):
        if glove_stats[label, cv2.CC_STAT_AREA] < 32:
            continue
        gy, gx = np.nonzero(glove_labels == label)
        distances = [np.min((gx - cuff[0]) ** 2 + (gy - cuff[1]) ** 2) for cuff in cuffs]
        side_gloves[int(np.argmin(distances))] |= glove_labels == label
    for side in range(2):
        side_glove = side_gloves[side]
        if not side_glove.any():
            continue
        near_glove = cv2.dilate(
            side_glove.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
        ).astype(bool)
        candidates: list[tuple[float, int, int, int]] = []
        prior_direction = prior_roots[side] - prior_cuffs[side]
        prior_direction /= max(float(np.linalg.norm(prior_direction)), 1e-06)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 128:
                continue
            component = labels == label
            contact = int(np.count_nonzero(component & near_glove))
            if contact:
                cy, cx = np.nonzero(component)
                boundary = (cx <= 3) | (cx >= width - 4) | (cy <= 3) | (cy >= height - 4)
                if np.any(boundary):
                    vector = (
                        np.asarray([np.median(cx[boundary]), np.median(cy[boundary])])
                        - prior_cuffs[side]
                    )
                else:
                    distance = np.hypot(cx - prior_cuffs[side, 0], cy - prior_cuffs[side, 1])
                    far = distance >= np.percentile(distance, 75)
                    vector = (
                        np.asarray([np.median(cx[far]), np.median(cy[far])]) - prior_cuffs[side]
                    )
                cosine = float(
                    np.dot(
                        vector / max(float(np.linalg.norm(vector)), 1e-06),
                        prior_direction,
                    )
                )
                candidates.append((cosine, contact, area, label))
        if not candidates:
            continue
        component = labels == max(candidates)[3]
        interface = component & near_glove
        iy, ix = np.nonzero(interface)
        if ix.size:
            cuffs[side] = [float(np.median(ix)), float(np.median(iy))]
        yy, xx = np.nonzero(component)
        boundary_groups = [xx <= 3, xx >= width - 4, yy <= 3, yy >= height - 4]
        nonempty = [group for group in boundary_groups if np.any(group)]
        if nonempty:
            prior_direction = prior_roots[side] - cuffs[side]
            prior_direction /= max(float(np.linalg.norm(prior_direction)), 1e-06)
            scored = []
            diagonal = float(np.hypot(width, height))
            for group in nonempty:
                point = np.asarray([float(np.median(xx[group])), float(np.median(yy[group]))])
                vector = point - cuffs[side]
                length = float(np.linalg.norm(vector))
                cosine = float(np.dot(vector / max(length, 1e-06), prior_direction))
                scored.append((cosine + 0.2 * length / diagonal, group))
            group = max(scored, key=lambda item: item[0])[1]
            bx, by = (xx[group], yy[group])
            distance = np.hypot(bx - cuffs[side, 0], by - cuffs[side, 1])
            far = distance >= np.percentile(distance, 50)
            roots[side] = [float(np.median(bx[far])), float(np.median(by[far]))]
            continue
        distance = np.hypot(xx - cuffs[side, 0], yy - cuffs[side, 1])
        far = distance >= np.percentile(distance, 75)
        target = np.asarray([np.median(xx[far]), np.median(yy[far])])
        direction = target - cuffs[side]
        if np.linalg.norm(direction) < 8.0:
            direction = prior_roots[side] - cuffs[side]
        ts = []
        if direction[1] > 1e-06:
            ts.append((height - 1 - cuffs[side, 1]) / direction[1])
        if direction[1] < -1e-06:
            ts.append((0.0 - cuffs[side, 1]) / direction[1])
        if direction[0] < -1e-06:
            ts.append((0.0 - cuffs[side, 0]) / direction[0])
        if direction[0] > 1e-06:
            ts.append((width - 1 - cuffs[side, 0]) / direction[0])
        ts = [value for value in ts if value > 0.0]
        if ts:
            roots[side] = cuffs[side] + direction * min(ts)
    return (cuffs, roots)
