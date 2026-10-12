"""SONIC pose, joint and latent packets for the local simulation runner."""
import json

import numpy as np

from ..refinement.decoder import ISAACLAB_TO_MUJOCO

MUJOCO_TO_ISAACLAB = np.argsort(ISAACLAB_TO_MUJOCO)

def pack(fields, *, topic='pose', version=3):
    metadata, payload = [], []
    types = {'f': {4: 'f32', 8: 'f64'}, 'i': {4: 'i32', 8: 'i64'}, 'u': {1: 'u8'}}
    for name, value in fields.items():
        value = np.asarray(value)
        dtype = types[value.dtype.kind][value.dtype.itemsize]
        little = np.ascontiguousarray(value.astype(value.dtype.newbyteorder('<')))
        metadata.append(dict(name=name, dtype=dtype, shape=list(little.shape)))
        payload.append(little.tobytes())
    header = json.dumps(dict(v=version, endian='le', count=1, fields=metadata),
                        separators=(',', ':')).encode()
    if len(header) > 1280:
        raise ValueError('SONIC packet header exceeds 1280 bytes')
    return topic.encode() + header.ljust(1280, b'\0') + b''.join(payload)


def command_packet(*, start=False, stop=False, planner=False):
    return pack({key: np.array([value], dtype=np.uint8)
                 for key, value in (('start', start), ('stop', stop), ('planner', planner))},
                topic='command', version=1)


def idle_planner_packet():
    return pack(dict(mode=np.array([0], dtype=np.int32),
                     movement=np.zeros(3, dtype=np.float32),
                     facing=np.array([1, 0, 0], dtype=np.float32),
                     speed=np.array([-1], dtype=np.float32),
                     height=np.array([-1], dtype=np.float32)), topic='planner', version=1)


def _hand(value):
    hand = np.zeros(7, dtype=np.float32) if value is None else np.asarray(value, dtype=np.float32)
    if hand.shape not in ((6,), (7,)) or not np.isfinite(hand).all():
        raise ValueError('expected finite six- or seven-component hand commands')
    return np.pad(hand, (0, 7-len(hand))).reshape(1, 7)


def _array(value, shape):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'expected finite array with shape {shape}')
    return array


def latent_packet(token, index, *, left_hand=None, right_hand=None):
    return pack(dict(token_state=_array(token, (64,)).reshape(1, 64),
                     frame_index=np.array([index], dtype=np.int64),
                     left_hand_joints=_hand(left_hand), right_hand_joints=_hand(right_hand)), version=4)


def joint_packet(joints, velocities, quaternions, index, *, left_hand=None, right_hand=None,
                 wire_index=None):
    """Convert absolute MuJoCo joints to encoder order with 46 future frames."""
    rows = len(joints)
    q = _array(joints, (rows, 29))[:, MUJOCO_TO_ISAACLAB]
    dq = _array(velocities, (rows, 29))[:, MUJOCO_TO_ISAACLAB]
    quat = _array(quaternions, (rows, 4))
    if not 0 <= index < rows:
        raise ValueError('joint frame index outside trajectory')
    frames = np.arange(index, index+46, dtype=np.int64)
    selected = np.minimum(frames, rows-1)
    velocity = dq[selected].copy()
    velocity[frames >= rows] = 0
    return pack(dict(joint_pos=q[selected], joint_vel=velocity, body_quat_w=quat[selected],
                     frame_index=frames + (0 if wire_index is None else wire_index-index),
                     catch_up=np.array([0], dtype=np.float32),
                     left_hand_joints=_hand(left_hand), right_hand_joints=_hand(right_hand)), version=1)


def pose_packet(joints, poses, quaternions, index, *, left_hand=None, right_hand=None,
                wire_index=None):
    """Human encoder stream with the original five-frame SMPL history."""
    rows = len(joints)
    joints = _array(joints, (rows, 24, 3))
    poses = _array(poses, (rows, 21, 3))
    quaternions = _array(quaternions, (rows, 4))
    if not 0 <= index < rows:
        raise ValueError('pose frame index outside trajectory')
    frames = np.arange(max(0, index-4), index+1, dtype=np.int64)
    return pack(dict(smpl_pose=poses[frames], smpl_joints=joints[frames],
                     body_quat_w=quaternions[frames], joint_pos=np.zeros((len(frames), 29), dtype=np.float32),
                     joint_vel=np.zeros((len(frames), 29), dtype=np.float32),
                     frame_index=frames + (0 if wire_index is None else wire_index-index),
                     left_hand_joints=_hand(left_hand), right_hand_joints=_hand(right_hand)), version=3)
