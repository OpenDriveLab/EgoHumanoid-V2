"""Locate palm and forearm anchors and bound image-space arm rendering."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ArmImageAnchors:
    """Palm and forearm anchor pairs in pixels, ordered left then right."""

    arm_points: np.ndarray


_REFERENCE_WIDTH = 1280.0
_REFERENCE_HEIGHT = 720.0
_ARM_POINTS_REFERENCE = np.asarray(
    [[[175.0, 460.0], [75.0, 660.0]], [[1005.0, 435.0], [1180.0, 650.0]]],
    dtype=np.float32,
)
DEFAULT_GEOMETRY_CUFF_FOREARM_FRACTION = 0.15


def default_arm_anchors(width: int, height: int) -> ArmImageAnchors:
    """Scale fallback image anchors to the video resolution."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    scale = np.asarray(
        [float(width) / _REFERENCE_WIDTH, float(height) / _REFERENCE_HEIGHT],
        dtype=np.float32,
    )
    return ArmImageAnchors(arm_points=_ARM_POINTS_REFERENCE * scale)


def cuff_anchor_points(
    anchors: ArmImageAnchors,
    *,
    forearm_fraction: float = DEFAULT_GEOMETRY_CUFF_FOREARM_FRACTION,
) -> np.ndarray:
    """Return stable sleeve-axis anchors for aligning the rendered wrists."""
    fraction = float(forearm_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("forearm_fraction must be finite and within [0, 1]")
    arms = np.asarray(anchors.arm_points, dtype=np.float32)
    if arms.shape != (2, 2, 2) or not np.isfinite(arms).all():
        raise ValueError("anchor arm_points must be finite with shape (2, 2, 2)")
    palms = arms[:, 0]
    sleeves = arms[:, 1]
    return palms + fraction * (sleeves - palms)


def _component_masks(mask: np.ndarray, minimum_area: int) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [
        labels == label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area
    ]


def _deep_point(component: np.ndarray, *, upper_fraction: float | None = None) -> np.ndarray:
    candidate = component.copy()
    if upper_fraction is not None:
        ys = np.nonzero(component)[0]
        if ys.size:
            cutoff = np.min(ys) + upper_fraction * max(1, np.max(ys) - np.min(ys))
            candidate &= np.indices(component.shape)[0] <= cutoff
    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    distance[~candidate] = -1.0
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return np.asarray([x, y], dtype=np.float32)


def _sleeve_point_from_glove_axis(
    glove_component: np.ndarray, sleeve_color: np.ndarray, diagonal: float
) -> np.ndarray | None:
    """Follow the glove's principal axis through its warm-sleeve endpoint."""
    glove_y, glove_x = np.nonzero(glove_component)
    sleeve_y, sleeve_x = np.nonzero(sleeve_color)
    if len(glove_x) < 16 or len(sleeve_x) < 16:
        return None
    glove_points = np.column_stack([glove_x, glove_y]).astype(np.float32)
    center = glove_points.mean(axis=0)
    covariance = np.cov((glove_points - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projection = (glove_points - center) @ axis
    low = glove_points[projection <= np.percentile(projection, 8)].mean(axis=0)
    high = glove_points[projection >= np.percentile(projection, 92)].mean(axis=0)
    sleeve_points = np.column_stack([sleeve_x, sleeve_y]).astype(np.float32)
    near_radius = 0.1 * diagonal
    nearby_counts = [
        int(np.sum(np.linalg.norm(sleeve_points - endpoint[None, :], axis=1) < near_radius))
        for endpoint in (low, high)
    ]
    wrist_index = int(np.argmax(nearby_counts))
    wrist = (low, high)[wrist_index]
    fingertips = (high, low)[wrist_index]
    direction = wrist - fingertips
    norm = float(np.linalg.norm(direction))
    if norm < 1.0:
        return None
    target = wrist + direction / norm * (0.13 * diagonal)
    yy, xx = np.indices(glove_component.shape)
    target_distance = np.hypot(xx - target[0], yy - target[1])
    interior = cv2.distanceTransform(sleeve_color.astype(np.uint8), cv2.DIST_L2, 5)
    score = interior - 0.55 * target_distance
    score[~sleeve_color] = -np.inf
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return np.asarray([x, y], dtype=np.float32)


def _plausible_or_fallback_sleeve_point(
    sleeve: np.ndarray | None, *, palm: np.ndarray, side: int, width: int, height: int
) -> np.ndarray:
    """Keep a lower, body-side sleeve click or derive one from the palm."""
    if side not in (0, 1):
        raise ValueError(f"side must be 0 or 1, got {side}")
    palm_value = np.asarray(palm, dtype=np.float32)
    if palm_value.shape != (2,) or not np.isfinite(palm_value).all():
        raise ValueError("palm must be a finite point with shape (2,)")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    diagonal = float(np.hypot(width, height))
    candidate = None if sleeve is None else np.asarray(sleeve, dtype=np.float32)
    below_palm = (
        candidate is not None
        and candidate.shape == (2,)
        and np.isfinite(candidate).all()
        and (candidate[1] > palm_value[1] + 0.02 * diagonal)
    )
    body_side = False
    if below_palm:
        tolerance = 0.02 * width
        body_side = (
            candidate[0] <= palm_value[0] + tolerance
            if side == 0
            else candidate[0] >= palm_value[0] - tolerance
        )
    outward_distance = (
        palm_value[0] - candidate[0]
        if candidate is not None and side == 0
        else candidate[0] - palm_value[0]
        if candidate is not None
        else -np.inf
    )
    bent_elbow_side_entry = (
        candidate is not None
        and candidate.shape == (2,)
        and np.isfinite(candidate).all()
        and (outward_distance >= 0.04 * diagonal)
    )
    if below_palm and body_side or bent_elbow_side_entry:
        return candidate.copy()
    if side == 0:
        horizontal_fraction, vertical_fraction = (-0.09, 0.28)
    elif palm_value[1] >= 0.6 * height:
        horizontal_fraction, vertical_fraction = (0.08, 0.24)
    else:
        horizontal_fraction, vertical_fraction = (0.18, 0.14)
    fallback = palm_value + np.asarray(
        [horizontal_fraction * width, vertical_fraction * height], dtype=np.float32
    )
    return np.clip(
        fallback,
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([width - 1.0, height - 1.0], dtype=np.float32),
    )


def detect_arm_anchors(
    rgb: np.ndarray, *, right_minimum_y_fraction: float = 0.12
) -> ArmImageAnchors:
    """Locate palm and forearm anchors from blue-glove and skin-color regions."""
    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("rgb must be a uint8 array with shape (H, W, 3)")
    if not 0.0 <= right_minimum_y_fraction < 1.0:
        raise ValueError("right_minimum_y_fraction must lie in [0, 1)")
    height, width = frame.shape[:2]
    fallback = default_arm_anchors(width, height)
    values = frame.astype(np.int16)
    chroma = values.max(axis=2) - values.min(axis=2)
    intensity = values.mean(axis=2)
    glove_color = (chroma <= 52) & (intensity >= 42) & (intensity <= 185)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    blue_glove_color = (
        (hsv[..., 0] >= 80) & (hsv[..., 0] <= 130) & (hsv[..., 1] >= 25) & (hsv[..., 2] >= 20)
    )
    sleeve_color = (
        (hsv[..., 0] <= 35) & (hsv[..., 1] >= 55) & (hsv[..., 2] >= 35) & (hsv[..., 2] <= 195)
    )
    skin_color = (
        (hsv[..., 0] <= 25) & (hsv[..., 1] >= 35) & (hsv[..., 1] <= 220) & (hsv[..., 2] >= 55)
    )
    morphology = np.ones((5, 5), dtype=np.uint8)
    glove_color = cv2.morphologyEx(
        glove_color.astype(np.uint8), cv2.MORPH_CLOSE, morphology
    ).astype(bool)
    blue_glove_color = cv2.morphologyEx(
        blue_glove_color.astype(np.uint8), cv2.MORPH_CLOSE, morphology
    ).astype(bool)
    distance_to_blue = cv2.distanceTransform((~blue_glove_color).astype(np.uint8), cv2.DIST_L2, 5)
    blue_guided_glove_color = blue_glove_color | glove_color & (
        distance_to_blue <= 0.04 * float(np.hypot(width, height))
    )
    sleeve_color = cv2.morphologyEx(
        sleeve_color.astype(np.uint8), cv2.MORPH_CLOSE, morphology
    ).astype(bool)
    yy, xx = np.indices((height, width))
    bare_hand_components: list[np.ndarray | None] = []
    for side in range(2):
        outer = xx < int(0.46 * width) if side == 0 else xx > int(0.54 * width)
        candidates = _component_masks(
            skin_color & outer & (yy > int(0.55 * height)),
            minimum_area=max(64, round(0.00012 * width * height)),
        )
        candidates = [
            component
            for component in candidates
            if component[-3:].any() and float(component.sum()) <= 0.06 * width * height
        ]
        bare_hand_components.append(
            max(candidates, key=lambda component: int(component.sum())) if candidates else None
        )
    use_bare_hands = all(component is not None for component in bare_hand_components) and all(
        float(distance_to_blue[component].min()) > 0.035 * float(np.hypot(width, height))
        for component in bare_hand_components
        if component is not None
    )
    palm_points: list[np.ndarray] = []
    sleeve_points: list[np.ndarray] = []
    diagonal = float(np.hypot(width, height))
    for side in range(2):
        side_region = xx < int(0.52 * width) if side == 0 else xx > int(0.48 * width)
        region = side_region & (yy > int(0.16 * height)) & (yy < int(0.86 * height))
        blue_components = _component_masks(
            blue_guided_glove_color & region,
            minimum_area=max(48, round(0.00015 * width * height)),
        )
        blue_components = [
            component for component in blue_components if np.any(component & blue_glove_color)
        ]
        components = blue_components or _component_masks(
            glove_color & region, minimum_area=max(48, round(0.00015 * width * height))
        )
        expected_x = (0.18 if side == 0 else 0.82) * width
        glove_component: np.ndarray | None = None
        if use_bare_hands:
            glove_component = bare_hand_components[side]
            assert glove_component is not None
            palm = (
                _deep_point(
                    np.pad(glove_component, 1, constant_values=False),
                    upper_fraction=0.82,
                )
                - 1.0
            )
        elif components:

            def glove_score(component: np.ndarray, expected_x: float = expected_x) -> float:
                _, xs = np.nonzero(component)
                area = float(len(xs))
                location_penalty = 1.0 + abs(float(np.median(xs)) - expected_x) / width
                return area / location_penalty

            ranked = sorted(components, key=glove_score, reverse=True)
            ranked_scores = [glove_score(component) for component in ranked]
            leading_dominance = (
                float("inf")
                if len(ranked_scores) == 1
                else ranked_scores[0] / max(ranked_scores[1], 1.0)
            )
            for rank, candidate in enumerate(ranked):
                candidate_palm = _deep_point(candidate, upper_fraction=0.74)
                candidate_y, candidate_x = np.nonzero(candidate)
                touches_same_side_edge = (
                    int(candidate_x.min()) == 0
                    if side == 0
                    else int(candidate_x.max()) == width - 1
                )
                has_lower_forearm_support = (
                    not touches_same_side_edge or float(candidate_y.max()) >= 0.64 * height
                )
                horizontal_span = float(candidate_x.max() - candidate_x.min() + 1)
                median_x = float(np.median(candidate_x))
                on_expected_outer_side = (
                    median_x <= 0.35 * width if side == 0 else median_x >= 0.65 * width
                )
                dominant_side_crop = (
                    rank == 0
                    and leading_dominance >= 2.0
                    and touches_same_side_edge
                    and has_lower_forearm_support
                    and (horizontal_span >= 0.12 * width)
                    and (float(candidate.sum()) >= 0.01 * width * height)
                    and on_expected_outer_side
                )
                if dominant_side_crop:
                    candidate_palm = (
                        _deep_point(
                            np.pad(candidate, 1, constant_values=False),
                            upper_fraction=0.74,
                        )
                        - 1.0
                    )
                palm_is_interior = (
                    candidate_palm[0] > 0.02 * width
                    if side == 0
                    else candidate_palm[0] < 0.98 * width
                )
                if side == 1:
                    palm_is_interior &= candidate_palm[1] > right_minimum_y_fraction * height
                if palm_is_interior and has_lower_forearm_support:
                    glove_component = candidate
                    palm = candidate_palm
                    break
        if glove_component is None:
            palm = fallback.arm_points[side, 0].copy()
        palm_points.append(palm)
        sleeve = (
            _sleeve_point_from_glove_axis(glove_component, sleeve_color & side_region, diagonal)
            if glove_component is not None
            else None
        )
        sleeve = _plausible_or_fallback_sleeve_point(
            sleeve, palm=palm, side=side, width=width, height=height
        )
        sleeve_points.append(sleeve)
    palms = np.stack(palm_points).astype(np.float32)
    sleeves = np.stack(sleeve_points).astype(np.float32)
    arm_points = np.stack([palms, sleeves], axis=1)
    return ArmImageAnchors(arm_points=arm_points)


def clip_render_alpha_to_removal_corridor(
    arm_alpha: np.ndarray,
    removal_mask: np.ndarray,
    *,
    expansion_px: int = 14,
    feather_px: int = 3,
) -> np.ndarray:
    """Limit projected robot geometry to a soft source-arm neighborhood."""
    alpha = np.asarray(arm_alpha)
    removal = np.asarray(removal_mask, dtype=bool)
    if alpha.ndim != 3 or alpha.shape[1:] != removal.shape:
        raise ValueError("arm_alpha must have shape (arms, H, W) matching removal")
    if expansion_px < 0 or feather_px < 1:
        raise ValueError("expansion must be non-negative and feather positive")
    corridor = removal.astype(np.uint8)
    if expansion_px:
        size = 2 * expansion_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        corridor = cv2.dilate(corridor, kernel)
    distance_inside = cv2.distanceTransform(corridor, cv2.DIST_L2, 5)
    weight = np.clip(distance_inside / float(feather_px), 0.0, 1.0)
    return alpha * weight[None].astype(alpha.dtype, copy=False)
