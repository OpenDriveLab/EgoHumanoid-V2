"""K stage: joint-space trajectory fitting before SONIC joint encoding."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr


def solve_foot_trajectory(baseline, targets, evaluate, *, lower, upper):
    """Fit sole positions with bounded joint corrections around initial intent.

    Positions and Jacobians use the body frame. The callback returns (4,3)
    positions and (4,3,29) Jacobians, ordered palms then soles.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if baseline.ndim != 2 or baseline.shape[1] != 29 or not len(baseline):
        raise ValueError('expected a nonempty (N,29) joint reference')
    rows, joints = baseline.shape
    if targets.shape != (rows, 4, 3) or not all(np.isfinite(a).all() for a in (baseline, targets)):
        raise ValueError('expected finite reference and (N,4,3) targets')
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != (29,) or upper.shape != (29,)
            or not np.isfinite([lower, upper]).all() or np.any(lower >= upper)):
        raise ValueError('expected ordered finite joint limits')
    radius = np.full(29, np.inf)
    radius[:15] = .08
    radius[[2, 8, 12]] = np.deg2rad(2.)
    lower = np.maximum(np.minimum(lower, baseline), baseline-radius)
    upper = np.minimum(np.maximum(upper, baseline), baseline+radius)
    current = np.clip(baseline, lower, upper)
    scale = np.repeat(np.sqrt([0., 0., 1., 1.]), 3)
    identity = sparse.eye(rows*joints, format='csr')
    posture = sparse.diags(np.tile(np.sqrt(np.full(joints, .001)), rows), format='csr')
    regularizers = [posture]
    for order, weight in ((1, .002), (2, .004)):
        if rows > order:
            coefficients = (-1., 1.) if order == 1 else (1., -2., 1.)
            difference = sparse.diags(
                [np.full(rows-order, value) for value in coefficients],
                range(order+1), shape=(rows-order, rows), format='csr')
            regularizers.append(sparse.kron(
                difference, sparse.diags(np.sqrt(np.full(joints, weight)), format='csr'),
                format='csr'))
    damping = np.sqrt(1e-5)*identity
    for _ in range(30):
        # Keep the residual's floating-point evaluation around the fixed intent.
        predicted = baseline + (current-baseline)*np.ones((1, joints))
        positions, jacobians = zip(*(evaluate(q) for q in predicted))
        positions, jacobians = np.asarray(positions), np.asarray(jacobians)
        if (positions.shape != (rows, 4, 3) or jacobians.shape != (rows, 4, 3, 29)
                or not np.isfinite(positions).all() or not np.isfinite(jacobians).all()):
            raise ValueError('invalid endpoint positions or Jacobians')
        task = sparse.block_diag(
            [sparse.csr_matrix(scale[:, None]*j.reshape(12, joints)) for j in jacobians],
            format='csr') @ identity
        correction = (current-baseline).reshape(-1)
        matrices = [task, *regularizers, damping]
        rhs = np.concatenate([
            np.tile(scale, rows)*(targets-positions).reshape(-1),
            *[-matrix@correction for matrix in regularizers], np.zeros(rows*joints)])
        step = lsmr(sparse.vstack(matrices, format='csr'), rhs,
                    atol=1e-7, btol=1e-7, maxiter=300)[0].reshape(rows, joints)
        step = np.clip(step, -.04, .04)
        norms = np.linalg.norm(step, axis=1)
        over = norms > .16
        step[over] *= (.16/norms[over])[:, None]
        candidate = np.clip(current+step, lower, upper)
        change = float(np.max(np.abs(candidate-current)))
        current = candidate
        if change <= 2e-5:
            break
    current = np.clip(current, lower, upper)
    limit = max(.16, float(np.max(np.abs(np.diff(baseline, axis=0))))) if rows > 1 else .16
    peak = float(np.max(np.abs(np.diff(current, axis=0)))) if rows > 1 else 0.
    if peak > limit:
        projected = current.copy()
        local_limit = max(limit-1e-6, .5*limit)
        for _ in range(200):
            previous = projected.copy()
            for indices in (range(1, rows), range(rows-1, 0, -1)):
                for index in indices:
                    difference = projected[index]-projected[index-1]
                    adjustment = .5*np.sign(difference)*np.maximum(np.abs(difference)-local_limit, 0.)
                    projected[index] -= adjustment
                    projected[index-1] += adjustment
                np.clip(projected, lower, upper, out=projected)
            serialized = projected.astype(np.float32).astype(np.float64)
            local_step = float(np.max(np.abs(np.diff(serialized, axis=0))))
            if float(np.max(np.abs(projected-previous))) <= 1e-10:
                break
        if local_step <= limit+1e-9:
            current, peak = projected, local_step
        if peak > limit:
            correction = current-baseline
            low, high = 0., 1.
            for _ in range(40):
                middle = .5*(low+high)
                trial = (baseline+middle*correction).astype(np.float32).astype(np.float64)
                if float(np.max(np.abs(np.diff(trial, axis=0)))) <= limit:
                    low = middle
                else:
                    high = middle
            current = baseline+low*correction
    return current.astype(np.float32)
