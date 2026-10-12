"""G1 physical endpoint geometry in gravity-aligned pelvis coordinates."""
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation


JOINT_NAMES = tuple(
    f'{side}_{joint}_joint'
    for side in ('left', 'right')
    for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')
) + ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint') + tuple(
    f'{side}_{joint}_joint'
    for side in ('left', 'right')
    for joint in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                  'wrist_roll', 'wrist_pitch', 'wrist_yaw')
)


def gravity_rotation(quaternion) -> np.ndarray:
    """Remove world heading while retaining measured pelvis roll and pitch."""
    quat = np.asarray(quaternion, dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12:
        raise ValueError('expected a finite nonzero wxyz quaternion')
    world = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    heading = np.arctan2(world[1, 0], world[0, 0])
    return Rotation.from_euler('z', -heading).as_matrix() @ world


class G1Kinematics:
    """Fixed-root URDF kinematics, with the deployed Revo2 palm transforms.

    Joint order: left leg, right leg, waist, left arm, right arm (29 DoF).
    Endpoint order: left palm, right palm, left sole, right sole.
    """

    def __init__(self, urdf: Path, *, control_limits=False):
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        ids = [self.model.getJointId(name) for name in JOINT_NAMES]
        if any(i >= self.model.njoints for i in ids):
            raise ValueError('URDF must contain all 29 G1 body joints')
        if any(self.model.joints[i].nq != 1 or self.model.joints[i].nv != 1 for i in ids):
            raise ValueError('G1 body joints must be single-axis joints')
        self.q_indices = np.array([self.model.joints[i].idx_q for i in ids])
        self.v_indices = np.array([self.model.joints[i].idx_v for i in ids])
        self.lower = self.model.lowerPositionLimit[self.q_indices].copy()
        self.upper = self.model.upperPositionLimit[self.q_indices].copy()
        if control_limits:
            self.lower[16] = max(self.lower[16], .19)
            self.upper[23] = min(self.upper[23], -.19)
        names = ('left_wrist_yaw_link', 'right_wrist_yaw_link',
                 'left_ankle_roll_link', 'right_ankle_roll_link')
        self.frames = [self.model.getFrameId(name) for name in names]
        if any(i >= self.model.nframes for i in self.frames):
            raise ValueError('URDF is missing a wrist or ankle frame')
        self.rotations = np.tile(np.eye(3), (4, 1, 1))
        self.rotations[:2] = Rotation.from_euler('xyz',
            [[-np.pi/2, 0, -np.pi/2], [np.pi/2, 0, np.pi/2]]).as_matrix()
        centers = np.array([[-.00467923, .00023722, .04666757],
                            [-.00466300, -.00024704, .04666762]])
        self.offsets = np.array([[.0591, 0, 0], [.0591, 0, 0],
                                 [.035, 0, -.035], [.035, 0, -.035]])
        self.offsets[:2] += np.einsum('nij,nj->ni', self.rotations[:2], centers)

    def evaluate(self, joints, base_quaternion=(1, 0, 0, 0)):
        """Return poses (4,7), wxyz, and xyz/rotvec Jacobian (24,29)."""
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (29,) or not np.isfinite(joints).all():
            raise ValueError('expected finite (29,) joint positions')
        local = gravity_rotation(base_quaternion)
        q = pin.neutral(self.model)
        q[self.q_indices] = joints
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        poses, jacobians = [], []
        for frame, offset, rotation in zip(self.frames, self.offsets, self.rotations):
            placement = self.data.oMf[frame]
            displacement = placement.rotation @ offset
            position = local @ (placement.translation + displacement)
            quat = Rotation.from_matrix(local @ placement.rotation @ rotation).as_quat()
            poses.append(np.r_[position, quat[[3, 0, 1, 2]]])
            jacobian = pin.getFrameJacobian(self.model, self.data, frame,
                                           pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, self.v_indices].copy()
            jacobian[:3] += np.cross(jacobian[3:].T, displacement).T
            jacobians.append(np.vstack((local @ jacobian[:3], local @ jacobian[3:])))
        return np.array(poses), np.vstack(jacobians)
