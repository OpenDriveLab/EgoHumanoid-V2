"""Human-only PICO geometry; no task profiles, time alignment or robot feedback."""

import numpy as np
from pathlib import Path
import hashlib
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


PICO_TO_ROBOT = np.array([[-1., 0, 0], [0, 0, 1], [0, 1, 0]])
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22])


def _heading(body, left=1, right=2):
    lateral = (body[:, left, :3] - body[:, right, :3]) @ PICO_TO_ROBOT.T
    if np.any(np.linalg.norm(lateral[:, :2], axis=1) < 1e-6):
        raise ValueError('body lateral axis is undefined')
    return np.unwrap(np.arctan2(-lateral[:, 0], lateral[:, 1]))


def body_geometry(body):
    body = np.asarray(body, dtype=float)
    if body.ndim != 3 or body.shape[1:] != (24, 7) or not len(body) or not np.isfinite(body).all():
        raise ValueError('expected finite body poses (N,24,7)')
    floor = np.median(np.min(body[:, [7, 8, 10, 11], 1], axis=1))
    height = body[:, 0, 1] - floor
    if np.any(height <= 0):
        raise ValueError('human root must be above the measured floor')
    heading = _heading(body)
    window = min(29, len(body) if len(body) % 2 else len(body)-1)
    if window >= 5:
        heading = savgol_filter(heading, window, 3)
    feet = (body[:, [7, 8], :3] + body[:, [10, 11], :3]) / 2
    feet[:, :, 1] -= np.median(np.min(feet[:, :, 1], axis=1)) - floor
    relative = (feet - body[:, 0, None, :3]) @ PICO_TO_ROBOT.T
    soles = np.einsum('nij,nkj->nki', Rotation.from_euler('z', -heading).as_matrix(), relative)
    return height, heading, soles, body[:, 0, :3] @ PICO_TO_ROBOT.T


def normalize_body(body):
    """Use initial hip and shoulder axes to calibrate the encoder reference."""
    result = np.asarray(body, dtype=float).copy()
    hip = float(np.median(_heading(result)[:10]))
    shoulder = float(np.median(_heading(result, 16, 17)[:10]))
    upper = np.array([3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    for indices, angle in ((np.arange(24), -hip), (upper, hip-shoulder)):
        matrix = PICO_TO_ROBOT.T @ Rotation.from_euler('z', angle).as_matrix() @ PICO_TO_ROBOT
        root = result[:, 0, None, :3].copy()
        result[:, indices, :3] = root + (result[:, indices, :3]-root) @ matrix.T
        rotations = Rotation.from_quat(result[:, indices, 3:].reshape(-1, 4))
        result[:, indices, 3:] = (Rotation.from_matrix(matrix) * rotations).as_quat().reshape(len(result), len(indices), 4)
    return result


def anatomical_frames(hand, valid):
    positions = np.asarray(hand)[..., :3]
    # OpenXR proximal finger bases, matching the original target convention.
    forward = positions[:, [6, 11, 16, 21]].mean(axis=1) - positions[:, 1]
    lateral = positions[:, 21] - positions[:, 6]
    def unit(vector):
        norm = np.linalg.norm(vector, axis=1)
        return vector / np.maximum(norm[:, None], 1e-12), np.isfinite(norm) & (norm > 1e-8)
    forward, first = unit(forward)
    lateral -= np.sum(lateral * forward, axis=1)[:, None] * forward
    lateral, second = unit(lateral)
    normal, third = unit(np.cross(forward, lateral))
    valid = np.asarray(valid, dtype=bool) & first & second & third & np.isfinite(positions).all(axis=(1, 2))
    matrices = np.stack([forward, np.cross(normal, forward), normal], axis=2)
    matrices[~valid] = np.eye(3)
    return matrices, valid


def smpl_inputs(body, sonic_root):
    """Use SONIC's released human skeleton and coordinate conventions."""
    body = np.asarray(body, dtype=float)
    if (body.ndim != 3 or body.shape[1:] != (24, 7) or not len(body)
            or not np.isfinite(body).all() or np.any(np.linalg.norm(body[..., 3:], axis=-1) < 1e-8)):
        raise ValueError('expected nonempty finite body poses with valid quaternions')
    import torch
    from gear_sonic.trl.utils.torch_transform import (
        angle_axis_to_quaternion, compute_human_joints, quat_apply, quat_inv,
        quaternion_to_angle_axis,
    )
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
    global_rotations = (Rotation.from_quat(body[..., 3:].reshape(-1, 4))
                        * Rotation.from_euler('y', 180, degrees=True)).as_matrix().reshape(-1, 24, 3, 3)
    local = np.einsum('njba,njbc->njac', global_rotations[:, PARENTS[1:22]], global_rotations[:, 1:22])
    poses = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 63).astype(np.float32)
    root = torch.from_numpy(Rotation.from_matrix(global_rotations[:, 0]).as_rotvec().astype(np.float32))
    with torch.no_grad():
        quaternion = smpl_root_ytoz_up(angle_axis_to_quaternion(root))
        joints = compute_human_joints(torch.from_numpy(poses), quaternion_to_angle_axis(quaternion),
                                      human_joints_info_path=str(Path(sonic_root)/'gear_sonic/data/human/human_joints_info.pkl'))
        quaternion = remove_smpl_base_rot(quaternion, w_last=False)
        joints = quat_apply(quat_inv(quaternion)[:, None, :].expand(-1, joints.shape[1], -1), joints)
    return dict(joints=joints.cpu().numpy(), poses=poses.reshape(-1, 21, 3),
                quaternions=quaternion.cpu().numpy())


def palm_targets(body, hands, active, heading):
    """Reuse collection's optical-palm recovery and anatomical calibration."""
    from human.processing.palm import (
        reconstruct_openxr_palm_series, align_openxr_palm_rotations_to_anatomical_frame,
        pico_palm_pose_to_current_root_local, brainco_base_rotations_from_anatomical_palm,
    )
    targets, diagnostics = [], {}
    for index, side in enumerate(('left', 'right')):
        hand = hands[:, index]
        recovered = reconstruct_openxr_palm_series(hand, active[:, index], body[:, 22+index])
        if recovered.unresolved.any():
            raise ValueError(f'{side} palm has no recoverable physical reference')
        anatomy, valid = anatomical_frames(hand, recovered.direct_valid)
        rotations, report = align_openxr_palm_rotations_to_anatomical_frame(
            recovered.rotations_world, anatomy, valid)
        position, rotation = pico_palm_pose_to_current_root_local(
            recovered.positions_world, rotations, body[:, 0, :3], heading,
            pico_to_robot_frame=PICO_TO_ROBOT)
        rotation = brainco_base_rotations_from_anatomical_palm(rotation, side=side)
        quaternion = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
        window = min(25, len(body) if len(body) % 2 else len(body)-1)
        if window >= 5:
            position = savgol_filter(position, window, 2, axis=0, mode='interp')
        for frame in range(1, len(quaternion)):
            if np.dot(quaternion[frame-1], quaternion[frame]) < 0:
                quaternion[frame] *= -1
        quaternion = gaussian_filter1d(quaternion, 8., axis=0, mode='nearest')
        norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError('palm smoothing produced an undefined orientation')
        targets.append(np.c_[position, quaternion/norm])
        diagnostics[side] = dict(report, fallback_frames=int(recovered.fallback_used.sum()))
    return np.stack(targets, axis=1), diagnostics


def hand_commands(hands, active):
    from gear_sonic.scripts.pico_manager_thread_server import retarget_openxr_hand_to_brainco
    commands = np.zeros((len(hands), 2, 6), dtype=np.float32)
    for frame, side in np.argwhere(active):
        commands[frame, side] = retarget_openxr_hand_to_brainco(hands[frame, side])[:6]
    if not np.isfinite(commands).all():
        raise ValueError('hand retargeting produced nonfinite commands')
    return commands


def prepare_raw(path, sonic_root, *, fps=50):
    """Normalize an already synchronized PICO episode, without cropping or retiming."""
    import pyarrow.parquet as pq
    columns = ['timestamp', 'observation.pico.body_pose']
    columns += [f'observation.pico.{side}_hand_{field}' for side in ('left', 'right')
                for field in ('pose', 'active')]
    arrays = pq.read_table(path, columns=columns).to_pydict()
    timestamp = np.asarray(arrays['timestamp'], dtype=float).reshape(-1)
    if len(timestamp) < 2 or not np.isfinite(timestamp).all() or not np.allclose(np.diff(timestamp), 1/fps, atol=2e-4, rtol=0):
        raise ValueError(f'input must already be synchronized at {fps:g} Hz')
    n = len(timestamp)
    body = np.asarray(arrays['observation.pico.body_pose'], dtype=float).reshape(n, 24, 7)
    hands = np.stack([np.asarray(arrays[f'observation.pico.{side}_hand_pose'], dtype=float).reshape(n, 26, 7)
                      for side in ('left', 'right')], axis=1)
    active = np.stack([np.asarray(arrays[f'observation.pico.{side}_hand_active'], dtype=bool).reshape(n)
                       for side in ('left', 'right')], axis=1)
    height, heading, soles, root = body_geometry(body)
    palms, diagnostics = palm_targets(body, hands, active, heading)
    targets = np.zeros((n, 4, 7))
    targets[..., 3] = 1
    targets[:, :2] = palms
    targets[:, 2:, :3] = soles
    prepared = smpl_inputs(normalize_body(body), sonic_root)
    prepared.update(hands=hand_commands(hands, active), targets=targets, human_height=height,
                    human_root_position=root,
                    human_displacement_cm=float(np.linalg.norm(root[:, :2]-root[0, :2], axis=1).max()*100))
    report = dict(frames=n, fps=fps, source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  palm_calibration=diagnostics, reference_frame='human_root_gravity_local')
    return prepared, report
