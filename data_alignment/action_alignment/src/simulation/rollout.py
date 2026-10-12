"""Fresh, loopback-only SONIC rollouts with measured state capture."""

from contextlib import ExitStack
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import msgpack
import numpy as np
import zmq

from .capture import FrameMatcher
from .protocol import command_packet, idle_planner_packet, joint_packet, latent_packet, pose_packet


def pose_startup(joints, pose, quaternion, fps):
    """One-second planner anchor followed by a five-second SMPL transition."""
    neutral_pose = np.zeros((21, 3), dtype=np.float32)
    neutral_pose[[8, 12, 13, 15, 16, 17, 18, 19, 20]] = [
        [.442231, .160744, -.035954], [.013171, -.502350, -.535810],
        [.118388, .489804, .542290], [-.103324, -.395104, -.382225],
        [-.176980, .386330, .394228], [.657332, -.232433, -.312498],
        [.552432, .229996, .242589], [.599486, .022090, -.134601],
        [.583013, -.097687, .055269]]
    neutral_joints = np.array([
        [0., 0., 0.], [-.026001, .058189, -.092763], [-.021250, -.063267, -.103908],
        [-.027618, -.002763, .109891], [-.035397, .112885, -.471517],
        [-.038074, -.107477, -.466288], [-.033558, .006685, .241744],
        [-.067273, .069431, -.874576], [-.058267, -.092061, -.877015],
        [-.005111, -.004645, .293979], [.050943, .116689, -.932573],
        [.060782, -.130873, -.935344], [.038580, -.009814, .456742],
        [.016857, .045945, .373824], [.026392, -.047812, .375903],
        [.120930, .028844, .592805], [.035703, .174406, .343294],
        [.079706, -.147891, .349648], [.042223, .220460, .079812],
        [.118046, -.206888, .083935], [.118445, .223487, -.161483],
        [.162034, -.203140, -.161911], [.090299, .161501, -.241099],
        [.128001, -.152199, -.246099]], dtype=np.float32) + joints[0]
    ticks = int(round(5*fps))
    phase = np.arange(1, ticks+1, dtype=float)/ticks
    blend = np.r_[np.zeros(int(round(fps))), phase*phase*(3-2*phase)]
    joint_rows = np.stack([((1-a)*neutral_joints+a*joints).astype(np.float32) for a in blend])
    pose_rows = np.stack([((1-a)*neutral_pose+a*pose).astype(np.float32) for a in blend])
    root = np.asarray(quaternion, dtype=np.float32)
    root = root/np.linalg.norm(root)
    return joint_rows, pose_rows, np.repeat(root[None], len(blend), axis=0), blend


def latent_startup(state, target, hand, fps):
    """Blend the live planner token and hand commands into the first reference."""
    target = np.asarray(target, dtype=np.float32)
    start = np.asarray(state.get('token_state', target), dtype=np.float32).reshape(-1)
    if start.shape != (64,) or not np.isfinite(start).all():
        start = target.copy()
    hand = np.asarray(hand, dtype=np.float32)
    if hand.shape == (2, 6):
        hand = np.pad(hand, ((0, 0), (0, 1)))
    start_hand = hand.copy()
    for index, side in enumerate(('left', 'right')):
        value = np.asarray(state.get(f'last_{side}_hand_action',
                                    state.get(f'{side}_hand_q', [])), dtype=np.float32).reshape(-1)
        if len(value) >= 7 and np.isfinite(value[:7]).all():
            start_hand[index] = value[:7]
    phase = np.arange(1, max(1, round(5*fps))+1, dtype=float)/max(1, round(5*fps))
    blend = phase*phase*(3-2*phase)
    tokens = np.stack([(1-float(a))*start+float(a)*target for a in blend])
    hands = np.stack([(1-float(a))*start_hand+float(a)*hand for a in blend])
    return tokens, hands


def controller_command(deploy, ports, *, hand_type='none'):
    motion, debug, state, command, _ = ports
    deploy = Path(deploy)
    return [str(deploy / 'target/release/g1_deploy_onnx_ref'), 'lo',
            str(deploy / 'policy/release/model_decoder.onnx'), str(deploy / 'reference/example'),
            '--obs-config', str(deploy / 'policy/release/observation_config.yaml'),
            '--encoder-file', str(deploy / 'policy/release/model_encoder.onnx'),
            '--planner-file', str(deploy / 'planner/target_vel/V2/planner_sonic.onnx'),
            '--input-type', 'zmq_manager', '--output-type', 'zmq',
            '--zmq-host', '127.0.0.1', '--zmq-port', str(motion),
            '--zmq-out-port', str(debug), '--zmq-robot-ip', '127.0.0.1',
            '--zmq-robot-state-port', str(state), '--zmq-robot-cmd-port', str(command),
            '--hand-type', hand_type, '--disable-crc-check']


def _warmup(step, fps):
    """Prime controller history without filtering intentionally moving references."""
    frames = int(round(5 * fps))
    origin = time.monotonic()
    for index in range(frames):
        step(0, holding=True, capture_after=origin+(index+1)/fps)
    return frames


def _settle(observe, send, fps):
    """Hold the first token until 75 consecutive full-body quiet samples."""
    quiet_frames = 0
    origin = time.monotonic()
    for elapsed in range(max(75, int(round(15 * fps)))):
        frame = observe()
        body = np.asarray(frame.get('body_dq', []), dtype=float).reshape(-1)
        base = np.asarray(frame.get('base_ang_vel', []), dtype=float).reshape(-1)
        quiet = (len(body) >= 29 and len(base) >= 3
                 and np.isfinite(body[:29]).all() and np.isfinite(base[:3]).all()
                 and np.max(np.abs(base[:3])) <= .03
                 and np.max(np.abs(body[:15])) <= .08
                 and np.max(np.abs(body[15:29])) <= .10)
        quiet_frames = quiet_frames + 1 if quiet else 0
        send()
        if quiet_frames >= 75:
            return elapsed + 1
        time.sleep(max(0, origin+(elapsed+1)/fps-time.monotonic()))
    raise TimeoutError('first token did not become quiet within the startup limit')


def _ports():
    # Reserve distinct ephemeral ports together. Bind failures never stop another run.
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(5)]
        for connection in sockets:
            connection.bind(('127.0.0.1', 0))
        return [connection.getsockname()[1] for connection in sockets]


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def replay_tokens(sonic_root, tokens, output, *, fps=50, scene=None, hands=None):
    """Start a fresh simulator, prime the first token, and replay every input row.

    Output is created exclusively, with one measured state per input frame.
    All sockets are loopback; no DDS is initialized.
    """
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 64 or not len(tokens) or not np.isfinite(tokens).all():
        raise ValueError('tokens must be a nonempty finite (N, 64) array')
    def packet(row, sequence, hand, holding):
        return latent_packet(tokens[row], sequence, left_hand=hand[0], right_hand=hand[1])
    return _rollout(sonic_root, len(tokens), packet, output, fps=fps, scene=scene,
                    hands=hands, expected_tokens=tokens)


def _trajectories(values, shapes):
    arrays = [np.asarray(value, dtype=np.float32) for value in values]
    if any(array.shape != (len(arrays[0]), *shape) or not len(array)
           or not np.isfinite(array).all() for array, shape in zip(arrays, shapes)):
        raise ValueError('trajectory arrays must have matching nonempty finite rows')
    if (np.linalg.norm(arrays[-1], axis=1) < 1e-8).any():
        raise ValueError('base quaternions must be nonzero')
    return arrays


def encode_joints(sonic_root, joints, velocities, quaternions, output, *, fps=50,
                  scene=None, hands=None):
    """Online K encoding with future joint references; replay its tokens separately."""
    q, dq, quat = _trajectories((joints, velocities, quaternions), ((29,), (29,), (4,)))
    def packet(row, sequence, hand, holding):
        arrays = (q[:1], np.zeros_like(dq[:1]), quat[:1]) if holding else (q, dq, quat)
        return joint_packet(*arrays, 0 if holding else row, wire_index=sequence,
                            left_hand=hand[0], right_hand=hand[1])
    return _rollout(sonic_root, len(q), packet, output, fps=fps, scene=scene, hands=hands)


def encode_poses(sonic_root, joints, poses, quaternions, output, *, fps=50,
                 scene=None, hands=None):
    """Online H encoding from SMPL history; replay its tokens separately."""
    joints, poses, quat = _trajectories((joints, poses, quaternions), ((24, 3), (21, 3), (4,)))
    startup = [np.repeat(value[:1], 5, axis=0) for value in (joints, poses, quat)]
    lead_joints, lead_poses, lead_quat, blend = pose_startup(joints[0], poses[0], quat[0], fps)
    lead = [np.concatenate((np.repeat(value[:1], 4, axis=0), value))
            for value in (lead_joints, lead_poses, lead_quat)]
    history = [np.concatenate((warm[-4:], values))
               for warm, values in zip(lead, (joints, poses, quat))]
    def packet(row, sequence, hand, holding):
        arrays = startup if holding else history
        return pose_packet(*arrays, 4 if holding else row+4, wire_index=sequence,
                           left_hand=hand[0], right_hand=hand[1])
    def prealign(row, sequence, hand, holding):
        return pose_packet(*lead, row+4, wire_index=sequence,
                           left_hand=blend[row]*hand[0], right_hand=blend[row]*hand[1])
    return _rollout(sonic_root, len(joints), packet, output, fps=fps, scene=scene,
                    hands=hands, startup_packet=prealign, startup_count=len(blend))


def _rollout(sonic_root, count, packet, output, *, fps, scene, hands, expected_tokens=None,
             startup_packet=None, startup_count=0):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive')
    hands = np.zeros((count, 2, 6)) if hands is None else np.asarray(hands, dtype=float)
    if hands.shape != (count, 2, 6) or not np.isfinite(hands).all():
        raise ValueError('hands must contain finite (N, 2, 6) commands')
    sonic_root = Path(sonic_root).resolve()
    deploy = sonic_root / 'gear_sonic_deploy'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ports = _ports()
    environment = os.environ.copy()
    environment.setdefault('MUJOCO_GL', 'disable')
    environment['PYTHONPATH'] = os.pathsep.join(
        [str(sonic_root), environment.get('PYTHONPATH', '')])

    with ExitStack() as stack:
        context = zmq.Context()
        stack.callback(context.term)
        publisher = context.socket(zmq.PUB)
        debug = context.socket(zmq.SUB)
        odometry = context.socket(zmq.SUB)
        control = context.socket(zmq.REQ)
        for connection in (publisher, debug, odometry, control):
            stack.callback(connection.close, 0)
        publisher.bind(f'tcp://127.0.0.1:{ports[0]}')
        debug.setsockopt(zmq.SUBSCRIBE, b'')
        debug.connect(f'tcp://127.0.0.1:{ports[1]}')
        odometry.setsockopt(zmq.SUBSCRIBE, b'rt/odostate')
        odometry.connect(f'tcp://127.0.0.1:{ports[2]}')
        control.setsockopt(zmq.RCVTIMEO, 3000)
        control.connect(f'tcp://127.0.0.1:{ports[4]}')
        simulator_log = stack.enter_context((output / 'simulation.log').open('w'))
        controller_log = stack.enter_context((output / 'controller.log').open('w'))
        simulator = subprocess.Popen(
            [sys.executable, '-m', 'align.simulation.runtime', '--state-port', str(ports[2]),
             '--command-port', str(ports[3]), '--control-port', str(ports[4])]
            + ([] if scene is None else ['--scene', str(Path(scene).resolve())]),
            cwd=sonic_root, env=environment, stdout=simulator_log, stderr=subprocess.STDOUT)
        stack.callback(_stop, simulator)
        controller = subprocess.Popen(controller_command(deploy, ports,
                                      hand_type='none' if scene is None else 'brainco'), cwd=deploy,
                                      env=environment, stdout=controller_log, stderr=subprocess.STDOUT)
        stack.callback(_stop, controller)
        matcher = FrameMatcher()
        pending = None

        def drain():
            nonlocal pending
            if simulator.poll() is not None or controller.poll() is not None:
                raise RuntimeError(f'rollout process exited; inspect {output}')
            while odometry.poll(0):
                _, payload = odometry.recv_multipart()
                matcher.add_odometry(msgpack.unpackb(payload, raw=False))
            while debug.poll(0):
                raw = debug.recv()
                if raw.startswith(b'g1_debug'):
                    pending = msgpack.unpackb(raw[8:], raw=False)

        sequence = 4

        def observe():
            drain()
            return pending or {}

        def hold_first():
            nonlocal sequence
            publisher.send(idle_planner_packet())
            publisher.send(packet(0, sequence, hands[0], True))
            sequence += 1

        def step(row, timeout=1.0, start=False, holding=False, prealign=False, capture_after=None):
            nonlocal sequence
            wire_sequence = sequence
            sequence += 1
            drain()
            if pending is not None:
                matcher.last_index = max(matcher.last_index, int(pending['index']))
            sent = time.monotonic()
            capture_after = sent+1/fps if capture_after is None else capture_after
            deadline = sent + timeout
            next_send = sent
            while time.monotonic() < deadline:
                now = time.monotonic()
                if now >= next_send:
                    send_packet = startup_packet if prealign else packet
                    publisher.send(send_packet(row, wire_sequence, hands[0 if holding else row], holding))
                    if start:
                        publisher.send(command_packet(start=True))
                    next_send = now + 1 / fps if start else deadline + 1
                drain()
                if pending is not None and time.monotonic() >= capture_after:
                    expected = None if expected_tokens is None or prealign else expected_tokens[row]
                    frame = matcher.capture(pending, expected_token=expected)
                    if frame is not None:
                        frame['capture_time'] = time.monotonic()
                        frame['response_seconds'] = frame['capture_time'] - sent
                        return frame
                time.sleep(.0005)
            raise TimeoutError(f'no matching measured frame; inspect {output}')

        try:
            deadline, next_start = time.monotonic()+60, 0.
            while pending is None:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(f'controller did not start; inspect {output}')
                if now >= next_start:
                    publisher.send(command_packet(start=True, planner=True))
                    next_start = now+.5
                drain()
                time.sleep(.05)
            control.send_string('hold')
            if control.recv_string() != 'ok':
                raise RuntimeError('simulator did not enable support')
            control.send_string('release')
            if control.recv_string() != 'ok':
                raise RuntimeError('simulator did not release support')
            release_deadline = time.monotonic()+1.
            while time.monotonic() < release_deadline:
                publisher.send(idle_planner_packet())
                time.sleep(min(1/fps, max(0, release_deadline-time.monotonic())))
            publisher.send(command_packet())
            time.sleep(.12)
            for _ in range(2):
                publisher.send(command_packet())
                time.sleep(.02)
            if expected_tokens is not None:
                drain()
                lead_tokens, lead_hands = latent_startup(pending, expected_tokens[0], hands[0], fps)
                def startup_packet(row, sequence, hand, holding):
                    return latent_packet(lead_tokens[row], sequence,
                                         left_hand=lead_hands[row, 0], right_hand=lead_hands[row, 1])
                startup_count = len(lead_tokens)
            if startup_packet is None:
                startup_frames = _warmup(step, fps)
            else:
                origin = time.monotonic()
                for row in range(startup_count):
                    step(row, holding=True, prealign=True, capture_after=origin+(row+1)/fps)
                startup_frames = startup_count
            if expected_tokens is not None:
                startup_frames += _settle(observe, hold_first, fps)
            origin = time.monotonic()
            frames = [step(row, capture_after=origin+(row+1)/fps) for row in range(count)]
            capture = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
            capture['capture_time'] -= capture['capture_time'][0]
            capture['startup_frames'] = np.asarray(startup_frames)
            if expected_tokens is not None:
                capture['input_tokens'] = expected_tokens.copy()
            capture['input_hands'] = hands.copy()
            np.savez_compressed(output / 'capture.npz', **capture)
            return capture
        finally:
            for _ in range(3):
                publisher.send(command_packet(stop=True))
                time.sleep(.02)
