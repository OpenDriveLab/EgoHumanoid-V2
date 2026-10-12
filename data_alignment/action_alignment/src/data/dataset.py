"""Portable reference and measured episode files for alignment examples."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REFERENCE_FIELDS = {
    'joints': ('teleop.smpl_joints', (24, 3)),
    'poses': ('teleop.smpl_pose', (21, 3)),
    'quaternions': ('teleop.body_quat_w', (4,)),
    'hands': ('teleop.hand_commands', (2, 6)),
    'targets': ('reference.endpoint_poses', (4, 7)),
    'human_height': ('reference.root_height', ()),
}


def _array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    return array


def write_episode(prepared, capture, targets, output, *, fps=50):
    """Write the immutable human reference and this exact measured rollout."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    n = len(prepared['joints'])
    displacement = float(prepared['human_displacement_cm'])
    if n < 1 or not np.isfinite(fps) or fps <= 0 or not np.isfinite(displacement) or displacement < 0:
        raise ValueError('invalid frame count, frame rate or human displacement')
    columns = dict(timestamp=np.arange(n)/fps, frame_index=np.arange(n),
                   episode_index=np.zeros(n, dtype=int), index=np.arange(n),
                   task_index=np.zeros(n, dtype=int))
    for key, (column, shape) in REFERENCE_FIELDS.items():
        array = _array(prepared[key], (n, *shape), key)
        columns[column] = array.reshape(n, -1).tolist() if shape else array
    if 'human_root_position' in prepared:
        root = _array(prepared['human_root_position'], (n, 3), 'human root position')
        measured = float(np.linalg.norm(root[:, :2]-root[0, :2], axis=1).max()*100)
        if not np.isclose(measured, displacement, rtol=1e-7, atol=1e-7):
            raise ValueError('human displacement does not match the reference trajectory')
        columns['reference.root_position'] = root.tolist()
    targets = _array(targets, (n, 4, 7), 'registered targets')
    columns['action.palm_center_target_local'] = targets[:, :2].reshape(n, 14).tolist()
    columns['action.sole_center_target_local'] = targets[:, 2:, :3].reshape(n, 6).tolist()
    measured = {
        'token_state': ('action.motion_token', (64,)),
        'body_q': ('debug.runtime_body_q_mujoco', (29,)),
        'body_dq': ('debug.runtime_body_dq_mujoco', (29,)),
        'odometry_world': ('debug.runtime_odometry_world', (7,)),
        'state_tick': ('debug.runtime_state_tick', ()),
        'capture_time': ('debug.runtime_capture_time', ()),
    }
    for key, (column, shape) in measured.items():
        array = _array(capture[key], (n, *shape), key)
        columns[column] = array.tolist()
    qpos = np.asarray(capture['simulation_qpos'])
    if qpos.ndim != 2 or qpos.shape[0] != n or qpos.shape[1] < 7 or not np.isfinite(qpos).all():
        raise ValueError('simulation_qpos must contain an exact configuration for every frame')
    columns['debug.runtime_simulation_qpos'] = qpos.tolist()
    metadata = dict(version=1, fps=fps, human_displacement_cm=displacement,
                    reference_frame='human_root_gravity_local', quaternion_order='wxyz')
    table = pa.table(columns).replace_schema_metadata({b'eh2.reference': json.dumps(metadata).encode()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        pq.write_table(table, stream, compression='zstd')


def read_reference(path, *, fps=50):
    """Load the human reference independently of stored robot motion tokens."""
    table = pq.read_table(path)
    encoded = (table.schema.metadata or {}).get(b'eh2.reference')
    if encoded is None:
        raise ValueError('not an alignment reference episode')
    metadata = json.loads(encoded)
    if metadata.get('version') != 1 or metadata.get('quaternion_order') != 'wxyz':
        raise ValueError('unsupported reference version or quaternion convention')
    if metadata.get('fps') != fps:
        raise ValueError('reference frame rate does not match the requested rollout')
    prepared = {}
    for key, (column, shape) in REFERENCE_FIELDS.items():
        prepared[key] = np.asarray(table[column].to_pylist()).reshape(len(table), *shape)
        _array(prepared[key], (len(table), *shape), key)
    prepared['human_displacement_cm'] = float(metadata['human_displacement_cm'])
    if 'reference.root_position' in table.column_names:
        root = np.asarray(table['reference.root_position'].to_pylist())
        prepared['human_root_position'] = _array(root, (len(table), 3), 'human root position')
        prepared['human_displacement_cm'] = float(np.linalg.norm(root[:, :2]-root[0, :2], axis=1).max()*100)
    return prepared
