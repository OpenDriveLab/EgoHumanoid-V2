"""H → K → D alignment from prepared human references and fresh rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
import torch

from .refinement.decoder import load_torch_decoder_from_onnx, rollout_observations
from .data.dataset import read_reference, write_episode
from .refinement.dynamics import optimize_tokens_for_linearized_task
from .correction.feet import solve_foot_trajectory
from .kinematics import G1Kinematics
from .refinement.response import EndpointResponse, ResponseTask, measured_pair_delta
from .simulation.rollout import encode_joints, encode_poses, replay_tokens
from .selection import pelvis_limit_cm, rank_candidates, rollout_metrics


def register_targets(targets, human_height, odometry):
    """Keep physical endpoint height while changing from human to robot pelvis."""
    targets = np.asarray(targets, dtype=float)
    height = np.asarray(human_height, dtype=float)
    odometry = np.asarray(odometry, dtype=float)
    if (targets.ndim != 3 or targets.shape[1:] != (4, 7)
            or height.shape != (len(targets),) or odometry.shape != (len(targets), 7)
            or not all(np.isfinite(a).all() for a in (targets, height, odometry))
            or np.any(height <= 0) or np.any(odometry[:, 2] <= 0)):
        raise ValueError('expected aligned target poses and positive human/robot heights')
    result = targets.copy()
    result[..., 2] += height[:, None] - odometry[:, 2, None]
    return result


def endpoint_weights(endpoints):
    weights = np.zeros((4, 6))
    if endpoints == 'hands':
        weights[:2, :3], weights[:2, 3:] = 1, .1
    elif endpoints == 'feet':
        weights[2:, :3] = 1
    else:
        raise ValueError('endpoints must be hands or feet')
    return weights


def endpoint_task(actual, targets, weights):
    """Project physical endpoint deltas onto absolute and bilateral objectives."""
    feet = bool(np.any(weights[2:]))
    left = 2 if feet else 0
    n = len(actual)
    blocks, residuals = [], []
    delta = measured_pair_delta(targets, actual)
    identity = np.eye(24)
    positions = [*range(6*left, 6*left+3), *range(6*(left+1), 6*(left+1)+3)]
    rotations = [index+3 for index in positions]
    blocks.append(np.broadcast_to(identity[positions], (n, 6, 24)).copy())
    residuals.append(delta[:, positions])
    if feet:
        relative = np.sqrt(.5)*(identity[positions[3:]]-identity[positions[:3]])
        blocks.append(np.broadcast_to(relative, (n, 3, 24)).copy())
        residuals.append(np.sqrt(.5)*(delta[:, positions[3:]]-delta[:, positions[:3]]))
    else:
        angular_scale = np.sqrt(weights[left, 3])
        blocks.append(np.broadcast_to(angular_scale*identity[rotations], (n, 6, 24)).copy())
        residuals.append(angular_scale*delta[:, rotations])
    return np.concatenate(blocks, axis=1), np.concatenate(residuals, axis=1)


def blend_startup_reference(initial, candidate):
    """Retain the fixed initial token prefix, then linearly introduce refinement."""
    initial = np.asarray(initial, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if initial.shape != candidate.shape or initial.ndim != 2:
        raise ValueError('startup token sources must have matching 2-D shapes')
    blend = np.clip((np.arange(len(initial), dtype=float)-10)/50, 0, 1)[:, None]
    return (initial + blend*(candidate-initial)).astype(np.float32)


def smooth_correction(optimized, baseline, *, safe=False):
    delta = np.asarray(optimized, dtype=float) - baseline
    element, norm_limit, slew = (.005, .02, .002) if safe else (.015, .05, .005)
    window = min(21 if safe else 15, len(delta) if len(delta) % 2 else len(delta) - 1)
    if window >= 3:
        delta = savgol_filter(delta, window, 2, axis=0, mode='interp')
    delta = np.clip(delta, -element, element)
    delta *= np.minimum(1, norm_limit / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12))
    delta[:47] = 0
    for index in range(47, len(delta)):
        delta[index] = np.clip(delta[index], delta[index-1]-slew, delta[index-1]+slew)
    return (baseline + delta).astype(np.float32)


def preservation_weights(weights):
    """Protect the joints that do not actuate the selected endpoint family."""
    result = np.full(29, 20.)
    if np.any(weights[:2]):
        result[15:] = .1
    if np.any(weights[2:]):
        result[:12] = .1
    return result


def task_joint_mask(weights):
    """Palm corrections actuate arms; sole corrections retain full-body authority."""
    mask = np.ones(29)
    if np.any(weights[:2]) and not np.any(weights[2:]):
        mask[:15] = 0
    return mask


def temporal_multipliers():
    """Penalize proximal joint motion more strongly than distal adjustments."""
    velocity, acceleration = np.ones(29), np.ones(29)
    for indices, v, a in [
        ([0, 1, 2, 6, 7, 8, 15, 16, 17, 22, 23, 24], 8, 32),
        ([3, 9, 18, 25], 3, 8),
        ([12, 13, 14], 16, 64),
    ]:
        velocity[indices], acceleration[indices] = v, a
    return velocity, acceleration


def endpoint_series(robot, capture):
    poses, jacobians = zip(*(robot.evaluate(q, quat) for q, quat in
                            zip(capture['body_q'], capture['base_quat'])))
    return np.asarray(poses), np.asarray(jacobians)


def initialize_hands(robot, generation, replay, targets):
    """Solve semantic palms from the initial reference in the replay body frame."""
    from .correction.arm_ik import ArmIK
    from .correction.arm_trajectory import transition_palm_targets, solve_arm_trajectory

    arms = ArmIK(robot)
    baseline = generation['body_q']
    root = replay['base_quat']
    palms, _ = transition_palm_targets(
        baseline, root, targets[:, :2].reshape(-1, 14), solver=arms,
        prefix_frames=10, ramp_frames=50)
    return solve_arm_trajectory(
        baseline, root, palms, solver=arms,
        waist_stage_solver=ArmIK(robot, waist_indices=(12, 13)),
        waist_solver=ArmIK(robot, waist_indices=(12, 13, 14)),
        iterations=80, waist_iterations=120, orientation_weight=.1)


def initialize_feet(robot, generation, replay, targets, encoder_root):
    """Correct measured sole error around the fixed initial joint reference."""
    from .correction.arm_trajectory import gravity_local_rotation_from_body_quat
    baseline = np.asarray(generation['body_q'], dtype=np.float64)
    positions = []
    for index, q in enumerate(baseline):
        anchor, _ = robot.evaluate(q, encoder_root[index])
        actual, _ = robot.evaluate(replay['body_q'][index], replay['base_quat'][index])
        desired = anchor[:, :3] + (targets[index, :, :3]-actual[:, :3])
        positions.append(gravity_local_rotation_from_body_quat(encoder_root[index]).inv().apply(desired))
    def evaluate(q):
        poses, jacobian = robot.evaluate(q)
        return poses[:, :3], jacobian.reshape(4, 6, 29)[:, :3]
    return solve_foot_trajectory(baseline, np.asarray(positions), evaluate,
                                 lower=robot.lower, upper=robot.upper)


def hand_teacher_correction(actual, targets, jacobians, orientation_weight):
    """Delayed per-arm damped correction used as a decoded-joint reference."""
    errors = measured_pair_delta(targets, actual)
    correction = np.zeros((len(actual), 29))
    scale = np.r_[np.ones(3), np.full(3, np.sqrt(orientation_weight))]
    for index in range(len(actual)):
        future = min(index+3, len(actual)-1)
        for side, start in enumerate((15, 22)):
            matrix = jacobians[future, side*6:side*6+6, start:start+7]*scale[:, None]
            residual = errors[future, side*6:side*6+6]*scale
            delta = matrix.T @ np.linalg.solve(matrix@matrix.T + 2e-4*np.eye(6), residual)
            delta *= .55
            delta *= min(1., .25/max(float(np.linalg.norm(delta)), 1e-12))
            correction[index, start:start+7] = np.clip(delta, -.12, .12)
    window = min(21, len(actual) if len(actual)%2 else len(actual)-1)
    if window >= 3:
        correction = savgol_filter(correction, window, min(2, window-1), axis=0, mode='interp')
    correction *= np.clip((np.arange(len(actual))-46)/50, 0, 1)[:, None]
    return correction.astype(np.float32)


def foot_teacher_correction(jacobians, residuals):
    """Compute a bilateral leg reference from the physical endpoint task."""
    correction = np.zeros((len(jacobians), 12))
    for index, (jacobian, residual) in enumerate(zip(jacobians, residuals)):
        matrix = jacobian[:, :12]
        delta = matrix.T @ np.linalg.solve(matrix@matrix.T + 2e-4*np.eye(len(matrix)), residual)
        delta *= .55
        delta *= min(1., .25/max(float(np.linalg.norm(delta)), 1e-12))
        correction[index] = np.clip(delta, -.12, .12)
    window = min(21, len(correction) if len(correction)%2 else len(correction)-1)
    if window >= 3:
        correction = savgol_filter(correction, window, min(2, window-1), axis=0, mode='interp')
    correction *= np.clip((np.arange(len(correction))-46)/50, 0, 1)[:, None]
    return correction


def refine_round(current, targets, robot, decoder, weights, *, response_model, device, safe=False):
    """Optimize with the frozen response model, then preserve startup."""
    weights = weights.copy()
    if safe:
        weights[:, 3:] *= 2
    actual, jacobians = endpoint_series(robot, current)
    tokens = current['token_state']
    observations = rollout_observations(current, tokens)
    projection, residual = endpoint_task(actual, targets, weights)
    mask = task_joint_mask(weights)
    task_jacobians = np.einsum('nre,nej->nrj', projection, jacobians)*mask
    response = EndpointResponse(response_model, observations, device=device, fps=50)
    blend = (.5 if safe else 1.) if np.any(weights[2:]) else .15
    rotation_blend = .6 if np.any(weights[2:]) else blend
    predictor = ResponseTask(response, jacobians, [(projection, mask)],
                             blend=blend, rotation_blend=rotation_blend)
    velocity, acceleration = temporal_multipliers()
    with torch.no_grad():
        baseline = decoder.joint_target(torch.as_tensor(observations, device=device)).cpu().numpy()
    teacher, teacher_weights = None, None
    if np.any(weights[:2]):
        teacher = baseline + hand_teacher_correction(actual, targets, jacobians, weights[0, 3])
        teacher_weights = task_joint_mask(weights)
    else:
        teacher = baseline.copy()
        teacher[:, :12] += foot_teacher_correction(task_jacobians, residual)
        teacher_weights = np.r_[np.ones(12), np.zeros(17)]
    optimized = optimize_tokens_for_linearized_task(
        decoder, observations, initial_tokens=tokens, baseline_joint_target=baseline,
        task_jacobians=task_jacobians,
        task_residuals=residual,
        task_predictor=predictor,
        joint_preservation_weights=preservation_weights(weights),
        joint_teacher_target=teacher, joint_teacher_weights=teacher_weights,
        joint_teacher_weight=.1 if teacher is not None else 0.,
        joint_velocity_weights=velocity * (.006 if safe else .002),
        joint_acceleration_weights=acceleration * (.012 if safe else .004),
        ridge_weight=.005 if safe else .002, temporal_weight=.008 if safe else .002,
        max_element_delta=.005 if safe else .015,
        max_row_norm_delta=.02 if safe else .05, iterations=400,
        learning_rate=.008 if safe else .01,
        device=torch.device(device))
    return smooth_correction(optimized, tokens, safe=safe)


def run_prepared(prepared, *, sonic_root, urdf, scene, output, endpoints, rounds=8,
                 fps=50, device=None, preparation_report=None):
    """Run every stage anew; a missing eligible result is recorded, never replaced."""
    if fps != 50:
        raise ValueError('alignment uses the response model at 50 Hz')
    weights = endpoint_weights(endpoints)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if rounds < 1:
        raise ValueError('rounds must be positive')
    count = len(prepared['joints'])
    if count <= 10:
        raise ValueError('alignment requires more than ten frames of observation history')
    human_displacement = float(prepared['human_displacement_cm'])
    if 'human_root_position' in prepared:
        root = np.asarray(prepared['human_root_position'], dtype=float)
        if root.shape != (count, 3) or not np.isfinite(root).all():
            raise ValueError('human root positions must match the input frames')
        human_displacement = float(np.linalg.norm(root[:, :2]-root[0, :2], axis=1).max()*100)
    if not np.isfinite(human_displacement) or human_displacement < 0:
        raise ValueError('human displacement must be finite and nonnegative')
    output, sonic_root = Path(output), Path(sonic_root)
    output.mkdir(parents=True, exist_ok=False)
    robot = G1Kinematics(urdf, control_limits=True)
    decoder = load_torch_decoder_from_onnx(
        sonic_root / 'gear_sonic_deploy/policy/release/model_decoder.onnx',
        device=torch.device(device))
    options = dict(fps=fps, scene=scene, hands=prepared['hands'])
    generated = encode_poses(sonic_root, prepared['joints'], prepared['poses'],
                             prepared['quaternions'], output/'H'/'encode', **options)
    current = replay_tokens(sonic_root, generated['token_state'], output/'H'/'replay', **options)
    targets = register_targets(prepared['targets'], prepared['human_height'], generated['odometry_world'])
    np.savez_compressed(output/'targets.npz', poses=targets)
    report = dict(endpoints=endpoints, fps=fps, human_displacement_cm=human_displacement,
                  pelvis_limit_cm=pelvis_limit_cm(human_displacement), rounds=[], selected=None)
    if preparation_report is not None:
        report['reference'] = preparation_report
    captures = {}

    def record(name, capture):
        captures[name] = capture
        actual, _ = endpoint_series(robot, capture)
        chosen = slice(0, 2) if endpoints == 'hands' else slice(2, 4)
        position, pelvis = rollout_metrics(actual[:, chosen, :3], targets[:, chosen, :3],
                                           capture['odometry_world'][:, :3])
        row = dict(stage=name, position_cm=position, pelvis_cm=pelvis)
        if endpoints == 'hands':
            difference = measured_pair_delta(actual, targets).reshape(-1, 4, 6)
            row['orientation_deg'] = float(np.linalg.norm(difference[:, :2, 3:], axis=-1).mean() * 180 / np.pi)
        report['rounds'].append(row)
        values = report['rounds']
        ranked = rank_candidates([(v['position_cm'], v['pelvis_cm']) for v in values],
                                  human_displacement_cm=human_displacement,
                                  orientation_errors_deg=(
                                      [v['orientation_deg'] for v in values] if endpoints == 'hands' else None))
        report['selected'] = values[ranked[0]]['stage'] if ranked else None
        (output/'run.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)

    record('H', current)
    encoder_root = np.asarray(prepared['quaternions'], dtype=float)
    encoder_root = encoder_root / np.linalg.norm(encoder_root, axis=1, keepdims=True)
    if endpoints == 'hands':
        q = initialize_hands(robot, generated, current, targets)
    else:
        q = initialize_feet(robot, generated, current, targets, encoder_root)
    velocity = np.gradient(q, 1/fps, axis=0)
    generated = encode_joints(sonic_root, q, velocity, encoder_root,
                              output/'K'/'encode', **options)
    current = replay_tokens(sonic_root, generated['token_state'],
                                               output/'K'/'replay', **options)
    record('K', current)
    for round_index in range(1, rounds + 1):
        name = f'D{round_index}'
        model_path = sonic_root / 'models/response' / ('hands.pt' if endpoints == 'hands' else 'feet.pt')
        tokens = refine_round(current, targets, robot, decoder,
                                              weights, response_model=model_path,
                                              device=device, safe=round_index == 8)
        tokens = blend_startup_reference(captures['H']['token_state'], tokens)
        (output/name).mkdir()
        trial = replay_tokens(sonic_root, tokens, output/name/'replay', **options)
        record(name, trial)
        current = trial
    if report['selected'] is not None:
        write_episode(prepared, captures[report['selected']], targets, output/'data.parquet', fps=fps)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--sonic-root', type=Path, default=Path('sonic'))
    parser.add_argument('--urdf', type=Path)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--endpoints', choices=('hands', 'feet'), required=True)
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--fps', type=float, default=50)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    arguments = vars(parser.parse_args())
    model = arguments['sonic_root'] / 'models/g1_revo2'
    arguments['urdf'] = arguments['urdf'] or arguments['sonic_root'] / 'gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf'
    arguments['scene'] = arguments['scene'] or model / 'scene.xml'
    source = arguments.pop('input')
    if source.suffix == '.parquet':
        import pyarrow.parquet as pq
        if b'eh2.reference' in (pq.read_schema(source).metadata or {}):
            prepared = read_reference(source, fps=arguments['fps'])
        else:
            from .data.reference import prepare_raw
            prepared, arguments['preparation_report'] = prepare_raw(
                source, arguments['sonic_root'], fps=arguments['fps'])
    else:
        with np.load(source, allow_pickle=False) as archive:
            prepared = dict(archive)
    run_prepared(prepared, **arguments)


if __name__ == '__main__':
    main()
