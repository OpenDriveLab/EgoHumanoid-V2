"""Alignment candidate metrics and selection."""

import numpy as np

PELVIS_MIN_MARGIN_CM = 5.
PELVIS_RELATIVE_MARGIN = .2
FEET_PELVIS_EXCESS_WEIGHT = .05


def pelvis_limit_cm(human_displacement_cm):
    """Return the horizontal pelvis displacement limit in cm."""
    human = float(human_displacement_cm)
    if not np.isfinite(human) or human < 0:
        raise ValueError('human displacement must be finite and nonnegative')
    return human + max(PELVIS_MIN_MARGIN_CM, PELVIS_RELATIVE_MARGIN * human)


def rollout_metrics(actual_endpoints, target_endpoints, pelvis_world) -> tuple[float, float]:
    """Return mean bilateral endpoint error and max horizontal displacement in cm.

    Endpoints are synchronized (frames, 2, 3) positions in the same coordinate
    frame, in meters. Pelvis positions are (frames, 3) world coordinates.
    Displacement is measured from the first frame, not accumulated path length.
    Compare identical time intervals and rollout conditions across candidates.
    """
    actual = np.asarray(actual_endpoints, dtype=float)
    target = np.asarray(target_endpoints, dtype=float)
    pelvis = np.asarray(pelvis_world, dtype=float)
    if (actual.ndim != 3 or actual.shape[1:] != (2, 3)
            or len(actual) == 0 or target.shape != actual.shape
            or pelvis.shape != (len(actual), 3)):
        raise ValueError('expected matching nonempty endpoints (N, 2, 3) and pelvis (N, 3)')
    if not all(np.isfinite(values).all() for values in (actual, target, pelvis)):
        raise ValueError('rollout positions must be finite')
    error = np.linalg.norm(actual - target, axis=-1).mean() * 100
    displacement = np.linalg.norm(pelvis[:, :2] - pelvis[0, :2], axis=-1).max() * 100
    return float(error), float(displacement)


def rank_candidates(scores, *, human_displacement_cm, orientation_errors_deg=None) -> list[int]:
    """Return eligible candidate indices in score order.

    Scores contain (mean bilateral position error, maximum horizontal pelvis
    displacement from the initial position), both in cm.
    Human displacement uses the same interval and its own starting point.
    Feet combine position error with weighted excess pelvis displacement.
    Hands supply mean bilateral orientation errors and use the paper's score:
    0.75 * position_cm / 5 + 0.25 * orientation_deg / 15. Lower is better;
    ties retain input order. An empty result means no eligible candidate.
    """
    limit = pelvis_limit_cm(human_displacement_cm)
    values = np.asarray(scores, dtype=float)
    if values.shape == (0,):
        return []
    if (values.ndim != 2 or values.shape[1] != 2
            or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('expected finite nonnegative (N, 2) scores')
    eligible = np.flatnonzero(values[:, 1] <= limit)
    error = values[:, 0]
    if orientation_errors_deg is not None:
        orientation = np.asarray(orientation_errors_deg, dtype=float)
        if (orientation.shape != (len(values),) or not np.isfinite(orientation).all()
                or (orientation < 0).any()):
            raise ValueError('expected one finite nonnegative orientation error per candidate')
        error = .75 * error / 5. + .25 * orientation / 15.
    else:
        excess = np.maximum(0., values[:, 1] - float(human_displacement_cm))
        error = error + FEET_PELVIS_EXCESS_WEIGHT * excess
    order = np.argsort(error[eligible], kind='stable')
    return eligible[order].tolist()
