"""Loopback-only transport for SONIC's released MuJoCo environment."""
from dataclasses import asdict
from types import SimpleNamespace
import argparse
from pathlib import Path
import time

import msgpack
import numpy as np
import zmq
from scipy.spatial.transform import Rotation


class InitialSupport:
    """Anchor the initial pelvis pose, then fade support over 250 physics steps."""

    def __init__(self, ramp_steps=250):
        if ramp_steps < 1:
            raise ValueError('support ramp must contain at least one step')
        self.ramp_steps = int(ramp_steps)
        self._enable = True
        self._anchor = None
        self._scale = 1.
        self._release_left = 0

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, value):
        if value and (not self._enable or self._release_left):
            self._anchor = None
            if not self._enable:
                self._scale = 0.
        self._enable = bool(value)
        self._release_left = 0

    def release_smoothly(self):
        if self.enable and not self._release_left:
            self._release_left = self.ramp_steps
            self._release_scale = self._scale

    def Advance(self, pose):
        if not self.enable:
            return np.zeros(6)
        pose = np.asarray(pose, dtype=float)
        rotation = Rotation.from_quat(pose[[4, 5, 6, 3]])
        if self._anchor is None:
            self._anchor = pose[:3].copy(), rotation
        if self._release_left:
            self._release_left -= 1
            self._scale = self._release_scale*self._release_left/self.ramp_steps
            if not self._release_left:
                self._enable = False
        else:
            self._scale = min(1., self._scale+1/self.ramp_steps)
        position, orientation = self._anchor
        force = 10000*(position-pose[:3])-1000*pose[7:10]
        torque = -1000*(rotation*orientation.inv()).as_rotvec()-10*pose[10:13]
        return self._scale*np.r_[force, torque]


def initialize_stance(environment):
    """Set the controller's standing configuration once, before physics starts."""
    import mujoco
    from ..refinement.decoder import DEFAULT_ANGLES_MUJOCO
    model, data = environment.mj_model, environment.mj_data
    data.qpos[model.jnt_qposadr[environment.body_joint_index]] = DEFAULT_ANGLES_MUJOCO
    if environment.use_floating_root_link:
        data.qpos[2] = .8
    mujoco.mj_forward(model, data)
    environment.elastic_band = InitialSupport()


class SimulationTransport:
    """Use the upstream simulation bridge without creating any DDS channels."""

    def __init__(self, *, state_port, command_port):
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b'')
        self.handlers = {}
        self.snapshot = None
        for socket, port in ((self.publisher, state_port), (self.subscriber, command_port)):
            if port:
                socket.bind(f'tcp://127.0.0.1:{port}')
            else:
                socket.bind_to_random_port('tcp://127.0.0.1')
        self.state_endpoint = self.publisher.getsockopt(zmq.LAST_ENDPOINT).decode()
        self.command_endpoint = self.subscriber.getsockopt(zmq.LAST_ENDPOINT).decode()

    def channels(self):
        transport = self

        class Publisher:
            def __init__(self, topic, message_type=None):
                self.topic = topic.encode()

            def Init(self):
                pass

            def Write(self, message):
                values = asdict(message)
                if self.topic == b'rt/odostate' and transport.snapshot is not None:
                    values['simulation_qpos'] = transport.snapshot()
                payload = msgpack.packb(values, use_bin_type=True,
                                       default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item())
                transport.publisher.send_multipart([self.topic, payload])

        class Subscriber:
            def __init__(self, topic, message_type=None):
                self.topic = topic.encode()

            def Init(self, handler, queueLen=1):
                transport.handlers[self.topic] = handler

        return Publisher, Subscriber

    def poll(self):
        latest = {}
        while True:
            try:
                parts = self.subscriber.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            if len(parts) >= 2 and parts[0] in self.handlers:
                latest[parts[0]] = parts[1]
        for topic, payload in latest.items():
            message = msgpack.unpackb(payload, raw=False,
                                      object_hook=lambda value: SimpleNamespace(**value))
            self.handlers[topic](message)

    def close(self):
        self.publisher.close(0)
        self.subscriber.close(0)
        self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def apply_control(environment, command):
    if command not in ('hold', 'release'):
        raise ValueError('unknown simulation control')
    if command == 'hold':
        environment.elastic_band.enable = True
    else:
        environment.elastic_band.release_smoothly()
    return 'ok'


def revo2_targets(command):
    """Six Revo2 channels to the model's sixteen coupled finger positions."""
    values = np.asarray(command, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError('expected six finite Revo2 commands')
    flex = np.clip(values / [.5, .5, 1, 1, 1, 1], 0, 1)
    return np.r_[flex[1] * 1.5184, flex[0] * 1.0472, flex[0] * 1.0472, 0,
                 (flex[2:, None] * [1.4661, 1.6930, 1]).ravel()]


def create_environment(config, scene=None):
    """Use upstream dynamics, with explicit Revo2 actuator indexing when requested."""
    import mujoco
    from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
    from ..kinematics import JOINT_NAMES

    class ContinuousEnvironment(DefaultEnv):
        def check_fall(self):
            # A captured rollout must never silently reset to an upright pose.
            self.fall = bool(self.mj_data.qpos[2] < .2)

    if scene is None:
        return ContinuousEnvironment(config, 'default', onscreen=False, offscreen=False)

    class Revo2Environment(ContinuousEnvironment):
        def init_scene(self):
            self.robot.NUM_HAND_JOINTS = self.num_hand_dof = 16
            self.torques = np.zeros(61)
            body_limits = self.torque_limit[np.r_[0:22, 29:36]].copy()
            super().init_scene()
            self.body_joint_index = np.array([self.mj_model.joint(name).id for name in JOINT_NAMES])
            joint_to_actuator = {int(joint): i for i, (joint, _) in
                                 enumerate(self.mj_model.actuator_trnid)}
            self.actuators = [np.array([joint_to_actuator[int(j)] for j in indices])
                              for indices in (self.body_joint_index, self.left_hand_index,
                                              self.right_hand_index)]
            self.torque_limit = np.full(self.mj_model.nu, 1e6)
            self.torque_limit[self.actuators[0]] = body_limits
            self.finger_targets = np.zeros((2, 16))

        def prepare_obs(self):
            observation = super().prepare_obs()
            for name, indices in zip(('body', 'left_hand', 'right_hand'), self.actuators):
                observation[name + '_tau_est'] = self.mj_data.actuator_force[indices]
            return observation

        def sim_step(self):
            self.obs = self.prepare_obs()
            self.unitree_bridge.PublishLowState(self.obs)
            force = np.zeros(6)
            if self.elastic_band.enable:
                pose = np.r_[self.mj_data.xpos[self.band_attached_link],
                             self.mj_data.xquat[self.band_attached_link], np.zeros(6)]
                mujoco.mj_objectVelocity(self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY,
                                        self.band_attached_link, pose[7:13], 0)
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                force = self.elastic_band.Advance(pose)
            self.mj_data.xfrc_applied[self.band_attached_link] = force
            self.torques[self.actuators[0]] = self.compute_body_torques()
            for indices, target in zip(self.actuators[1:], self.finger_targets):
                self.torques[indices] = target
            self.mj_data.ctrl[:] = np.clip(self.torques, -self.torque_limit, self.torque_limit)
            mujoco.mj_step(self.mj_model, self.mj_data)
            self.check_fall()

    config = dict(config, ROBOT_SCENE=str(Path(scene).resolve()))
    return Revo2Environment(config, 'default', onscreen=False, offscreen=False)


def run(*, state_port, command_port, control_port, scene=None):
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim import unitree_sdk2py_bridge as sdk_bridge

    config = SimLoopConfig(interface='lo', enable_onscreen=False).load_wbc_yaml()
    config['ENABLE_ELASTIC_BAND'] = True
    config['USE_JOYSTICK'] = False
    with SimulationTransport(state_port=state_port, command_port=command_port) as transport:
        # SDK message definitions are reused; DDS networking is never initialized.
        originals = sdk_bridge.ChannelPublisher, sdk_bridge.ChannelSubscriber
        sdk_bridge.ChannelPublisher, sdk_bridge.ChannelSubscriber = transport.channels()
        try:
            bridge = sdk_bridge.UnitreeSdk2Bridge(config)
        finally:
            sdk_bridge.ChannelPublisher, sdk_bridge.ChannelSubscriber = originals
        environment = create_environment(config, scene)
        initialize_stance(environment)
        environment.set_unitree_bridge(bridge)
        transport.snapshot = lambda: environment.mj_data.qpos.copy()
        if scene is not None:
            for side, name in enumerate(('left', 'right')):
                def set_fingers(message, side=side):
                    environment.finger_targets[side] = revo2_targets(message.position)
                transport.handlers[f'rt/brainco/{name}/cmd'.encode()] = set_fingers
        control = transport.context.socket(zmq.REP)
        control.bind(f'tcp://127.0.0.1:{control_port}')
        print('Local simulation ready', flush=True)
        try:
            while True:
                start = time.monotonic()
                transport.poll()
                if control.poll(0):
                    command = control.recv_string()
                    if command == 'stop':
                        control.send_string('ok')
                        break
                    try:
                        control.send_string(apply_control(environment, command))
                    except ValueError as exc:
                        control.send_string(str(exc))
                environment.sim_step()
                time.sleep(max(0, config['SIMULATE_DT'] - (time.monotonic() - start)))
        finally:
            control.close(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run an isolated local SONIC simulation.')
    parser.add_argument('--state-port', type=int, required=True)
    parser.add_argument('--command-port', type=int, required=True)
    parser.add_argument('--control-port', type=int, required=True)
    parser.add_argument('--scene', type=Path, help='G1 with Revo2 MuJoCo scene')
    args = parser.parse_args(argv)
    run(**vars(args))


if __name__ == '__main__':
    main()
