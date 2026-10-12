#!/usr/bin/env python3
"""ZMQ-ROS2 Bridge for Unitree G1 + BrainCo hand.

Runs on the robot. Subscribes ROS2 topics (lowstate / dex3 / brainco motor_status),
forwards them to host via ZMQ; receives ZMQ commands from host and republishes them
as ROS2 topics (lowcmd / dex3 cmd / brainco joint_commands).

Wire ZMQ topic names are unchanged from the previous unitree_sdk2py-based bridge,
so host-side clients (zmq_channel.py, SONIC C++) require no change. New brainco
topics are added under `rt/brainco/...`.

Architecture:
    Robot                                       Host
    ┌─────────────────────────┐                 ┌─────────────────────────┐
    │  ROS2 (CycloneDDS)      │                 │  Control Loop           │
    │  ↕ rclpy node           │                 │  (zmq_channel etc)      │
    │  THIS SCRIPT            │  ←── ZMQ ──→    │                         │
    │  - PUB :5560 state      │                 │                         │
    │  - SUB :5561 cmd        │                 │                         │
    │  - UDP :5562 state      │                 │                         │
    └─────────────────────────┘                 └─────────────────────────┘

Requires sourced ROS 2 Foxy, Unitree messages and BrainCo messages.
Installation must not launch this module: it sends robot commands when run.
"""

import argparse
import os
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import zmq

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from unitree_hg.msg import HandCmd, HandState, IMUState, LowCmd, LowState
from sensor_msgs.msg import JointState

try:
    from unitree_go.msg import SportModeState
    HAS_SPORT_MODE_STATE = True
except ImportError:
    SportModeState = None
    HAS_SPORT_MODE_STATE = False

try:
    from ros2_stark_interfaces.msg import MotorStatus as BraincoMotorStatus
    HAS_BRAINCO = True
except ImportError:
    BraincoMotorStatus = None
    HAS_BRAINCO = False

# Keep unitree_sdk2py imports only for: (1) CRC over LowCmd, (2) one-shot MotionSwitcher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

from .wire import (
    ZmqOdoState,
    deserialize_message,
    serialize_message,
    sport_mode_state_to_odo_payload,
)
from ..hands.brainco import brainco_command_to_stark


# UDP State Protocol - fixed binary format for low-latency transmission
UDP_STATE_MAGIC = b"STAT"
UDP_STATE_FORMAT = "<4sI35f35f4f3f3f4f3f3f"
UDP_STATE_SIZE = struct.calcsize(UDP_STATE_FORMAT)  # 368 bytes


@dataclass
class BridgeConfig:
    state_port: int = 5560
    cmd_port: int = 5561
    udp_state_port: int = 5562
    loop_rate: float = 500.0
    enable_dex3: bool = True
    enable_brainco: bool = True
    verbose: bool = False
    enable_udp_state: bool = True
    udp_target_ip: str = ""
    enable_odometry: bool = True
    odometry_topic: str = "/sportmodestate"


class Ros2Bridge(Node):
    """rclpy node holding all subscribers and publishers used by the bridge."""

    def __init__(self, config: BridgeConfig):
        super().__init__("zmq_ros2_bridge")
        self.config = config

        # Caches (populated by subscription callbacks; consumed by main thread)
        self.cached_lowstate: Optional[LowState] = None
        self.cached_secondary_imu: Optional[IMUState] = None
        self.cached_left_hand: Optional[HandState] = None
        self.cached_right_hand: Optional[HandState] = None
        self.cached_brainco_left = None
        self.cached_brainco_right = None
        self.cached_odometry = None
        self.cached_lowstate_changed = False  # Whether main loop should send via UDP

        # Use BEST_EFFORT for high-rate state to match unitree firmware's QoS
        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscriptions: state in
        self.create_subscription(LowState, "/lowstate", self._lowstate_cb, state_qos)
        self.create_subscription(
            IMUState, "/secondary_imu", self._secondary_imu_cb, state_qos
        )
        if config.enable_dex3:
            self.create_subscription(
                HandState, "/dex3/left/state", self._left_hand_cb, state_qos
            )
            self.create_subscription(
                HandState, "/dex3/right/state", self._right_hand_cb, state_qos
            )
        if config.enable_brainco and HAS_BRAINCO:
            self.create_subscription(
                BraincoMotorStatus, "/motor_status", self._brainco_left_cb, state_qos
            )
            self.create_subscription(
                BraincoMotorStatus, "/motor_status_r", self._brainco_right_cb, state_qos
            )
        if config.enable_odometry and HAS_SPORT_MODE_STATE:
            self.create_subscription(
                SportModeState,
                config.odometry_topic,
                self._odometry_cb,
                state_qos,
            )

        # Publishers: command out
        self.lowcmd_pub = self.create_publisher(LowCmd, "/lowcmd", cmd_qos)
        if config.enable_dex3:
            self.left_hand_cmd_pub = self.create_publisher(
                HandCmd, "/dex3/left/cmd", cmd_qos
            )
            self.right_hand_cmd_pub = self.create_publisher(
                HandCmd, "/dex3/right/cmd", cmd_qos
            )
        else:
            self.left_hand_cmd_pub = None
            self.right_hand_cmd_pub = None
        if config.enable_brainco:
            self.brainco_left_cmd_pub = self.create_publisher(
                JointState, "/joint_commands_left", cmd_qos
            )
            self.brainco_right_cmd_pub = self.create_publisher(
                JointState, "/joint_commands_right", cmd_qos
            )
        else:
            self.brainco_left_cmd_pub = None
            self.brainco_right_cmd_pub = None

    # --- Subscription callbacks ---
    def _lowstate_cb(self, msg: LowState):
        self.cached_lowstate = msg
        self.cached_lowstate_changed = True

    def _secondary_imu_cb(self, msg: IMUState):
        self.cached_secondary_imu = msg

    def _left_hand_cb(self, msg: HandState):
        self.cached_left_hand = msg

    def _right_hand_cb(self, msg: HandState):
        self.cached_right_hand = msg

    def _brainco_left_cb(self, msg):
        self.cached_brainco_left = msg

    def _brainco_right_cb(self, msg):
        self.cached_brainco_right = msg

    def _odometry_cb(self, msg):
        self.cached_odometry = msg


class ZmqRos2Bridge:
    """Bridge between local ROS2 topics and remote ZMQ communication."""

    def __init__(self, config: BridgeConfig):
        self.config = config
        self.running = False

        self.context: Optional[zmq.Context] = None
        self.state_pub: Optional[zmq.Socket] = None
        self.cmd_sub: Optional[zmq.Socket] = None

        self.udp_socket: Optional[socket.socket] = None
        self.udp_target_addr: Optional[tuple] = None

        self.node: Optional[Ros2Bridge] = None
        self.spin_thread: Optional[threading.Thread] = None

        self.crc = CRC()

        # Cache last-received DDS-class LowCmd for high-rate publish (with CRC)
        self.last_lowcmd: Optional[LowCmd] = None

        self.stats = {
            "state_tx": 0,
            "cmd_rx": 0,
            "ros2_writes": 0,
            "errors": 0,
            "last_state_time": 0.0,
            "last_cmd_time": 0.0,
        }

    def init_zmq(self) -> bool:
        try:
            self.context = zmq.Context()

            self.state_pub = self.context.socket(zmq.PUB)
            self.state_pub.bind(f"tcp://*:{self.config.state_port}")
            self.state_pub.setsockopt(zmq.SNDHWM, 1000)
            print(f"[Bridge] State PUB bound to tcp://*:{self.config.state_port}")

            self.cmd_sub = self.context.socket(zmq.SUB)
            self.cmd_sub.bind(f"tcp://*:{self.config.cmd_port}")
            self.cmd_sub.setsockopt(zmq.RCVTIMEO, 1)
            self.cmd_sub.setsockopt(zmq.RCVHWM, 10)
            self.cmd_sub.setsockopt(zmq.CONFLATE, 1)
            self.cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "rt/lowcmd")
            self.cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "rt/dex3/left/cmd")
            self.cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "rt/dex3/right/cmd")
            self.cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "rt/brainco/left/cmd")
            self.cmd_sub.setsockopt_string(zmq.SUBSCRIBE, "rt/brainco/right/cmd")
            print(f"[Bridge] Command SUB bound to tcp://*:{self.config.cmd_port}")
            return True
        except zmq.ZMQError as e:
            print(f"[Bridge] ZMQ init error: {e}")
            return False

    def init_udp(self, target_ip: str) -> bool:
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
            self.udp_target_addr = (target_ip, self.config.udp_state_port)
            print(
                f"[Bridge] UDP state sender initialized, target: "
                f"{target_ip}:{self.config.udp_state_port}"
            )
            return True
        except Exception as e:
            print(f"[Bridge] UDP init error: {e}")
            return False

    def init_ros2(self) -> bool:
        try:
            if not rclpy.ok():
                rclpy.init()
            self.node = Ros2Bridge(self.config)
            self.spin_thread = threading.Thread(
                target=rclpy.spin, args=(self.node,), daemon=True
            )
            self.spin_thread.start()
            sub_summary = "/lowstate /secondary_imu"
            if self.config.enable_dex3:
                sub_summary += " /dex3/left/state /dex3/right/state"
            if self.config.enable_brainco:
                if HAS_BRAINCO:
                    sub_summary += " /motor_status /motor_status_r"
                else:
                    print(
                        "[Bridge] Warning: --enable-brainco set but ros2_stark_interfaces not found, "
                        "skipping brainco state subscriptions"
                    )
            if self.config.enable_odometry:
                if HAS_SPORT_MODE_STATE:
                    sub_summary += f" {self.config.odometry_topic}"
                else:
                    print(
                        "[Bridge] Warning: odometry requested but "
                        "unitree_go/SportModeState is unavailable; "
                        "fixed-R0 moving-base control will not start."
                    )
            print(f"[Bridge] ROS2 node initialized; subscribed: {sub_summary}")
            return True
        except Exception as e:
            print(f"[Bridge] ROS2 init error: {e}")
            return False

    @staticmethod
    def pack_state_udp(lowstate: LowState, secondary_imu: Optional[IMUState]) -> bytes:
        """Pack state into fixed binary format for low-latency UDP."""
        q = [lowstate.motor_state[i].q for i in range(35)]
        dq = [lowstate.motor_state[i].dq for i in range(35)]
        imu_quat = [lowstate.imu_state.quaternion[i] for i in range(4)]
        imu_gyro = [lowstate.imu_state.gyroscope[i] for i in range(3)]
        imu_accel = [lowstate.imu_state.accelerometer[i] for i in range(3)]
        if secondary_imu is not None:
            sec_quat = [secondary_imu.quaternion[i] for i in range(4)]
            sec_gyro = [secondary_imu.gyroscope[i] for i in range(3)]
            sec_accel = [secondary_imu.accelerometer[i] for i in range(3)]
        else:
            sec_quat = [1.0, 0.0, 0.0, 0.0]
            sec_gyro = [0.0, 0.0, 0.0]
            sec_accel = [0.0, 0.0, 0.0]
        return struct.pack(
            UDP_STATE_FORMAT,
            UDP_STATE_MAGIC,
            int(lowstate.tick),
            *q,
            *dq,
            *imu_quat,
            *imu_gyro,
            *imu_accel,
            *sec_quat,
            *sec_gyro,
            *sec_accel,
        )

    def send_state_udp(self, lowstate: LowState, secondary_imu: Optional[IMUState]):
        if self.udp_socket and self.udp_target_addr:
            try:
                data = self.pack_state_udp(lowstate, secondary_imu)
                self.udp_socket.sendto(data, self.udp_target_addr)
            except Exception as e:
                if self.config.verbose:
                    print(f"[Bridge] UDP send error: {e}")

    def publish_state_zmq(self, topic: str, msg) -> None:
        try:
            data = serialize_message(topic, msg)
            self.state_pub.send_multipart(
                [topic.encode("utf-8"), data], zmq.NOBLOCK
            )
            self.stats["state_tx"] += 1
            self.stats["last_state_time"] = time.time()
        except zmq.ZMQError as e:
            self.stats["errors"] += 1
            if self.config.verbose:
                print(f"[Bridge] State publish error: {e}")
        except Exception as e:
            self.stats["errors"] += 1
            if self.config.verbose:
                print(f"[Bridge] State serialize error ({topic}): {e}")

    def receive_command(self) -> Optional[tuple]:
        try:
            parts = self.cmd_sub.recv_multipart(zmq.NOBLOCK)
            if len(parts) >= 2:
                topic = parts[0].decode("utf-8")
                data = parts[1]
                self.stats["cmd_rx"] += 1
                self.stats["last_cmd_time"] = time.time()
                return topic, data
        except zmq.Again:
            pass
        except zmq.ZMQError as e:
            self.stats["errors"] += 1
            if self.config.verbose:
                print(f"[Bridge] Command receive error: {e}")
        return None

    def forward_command(self, topic: str, data: bytes):
        try:
            msg = deserialize_message(topic, data)

            if topic == "rt/lowcmd":
                ros2_cmd = LowCmd()
                ros2_cmd.mode_pr = int(msg.mode_pr)
                ros2_cmd.mode_machine = int(msg.mode_machine)
                for i in range(min(len(msg.motor_cmd), 35)):
                    ros2_cmd.motor_cmd[i].mode = int(msg.motor_cmd[i].mode)
                    ros2_cmd.motor_cmd[i].q = float(msg.motor_cmd[i].q)
                    ros2_cmd.motor_cmd[i].dq = float(msg.motor_cmd[i].dq)
                    ros2_cmd.motor_cmd[i].tau = float(msg.motor_cmd[i].tau)
                    ros2_cmd.motor_cmd[i].kp = float(msg.motor_cmd[i].kp)
                    ros2_cmd.motor_cmd[i].kd = float(msg.motor_cmd[i].kd)
                    ros2_cmd.motor_cmd[i].reserve = int(msg.motor_cmd[i].reserve)
                for i in range(min(len(msg.reserve), 4)):
                    ros2_cmd.reserve[i] = int(msg.reserve[i])
                self.last_lowcmd = ros2_cmd  # written at high rate by main loop

            elif (
                topic == "rt/dex3/left/cmd"
                and self.node
                and self.node.left_hand_cmd_pub
            ):
                ros2_cmd = HandCmd()
                for i in range(min(len(msg.motor_cmd), 7)):
                    ros2_cmd.motor_cmd[i].mode = int(msg.motor_cmd[i].mode)
                    ros2_cmd.motor_cmd[i].q = float(msg.motor_cmd[i].q)
                    ros2_cmd.motor_cmd[i].dq = float(msg.motor_cmd[i].dq)
                    ros2_cmd.motor_cmd[i].tau = float(msg.motor_cmd[i].tau)
                    ros2_cmd.motor_cmd[i].kp = float(msg.motor_cmd[i].kp)
                    ros2_cmd.motor_cmd[i].kd = float(msg.motor_cmd[i].kd)
                self.node.left_hand_cmd_pub.publish(ros2_cmd)
                self.stats["ros2_writes"] += 1

            elif (
                topic == "rt/dex3/right/cmd"
                and self.node
                and self.node.right_hand_cmd_pub
            ):
                ros2_cmd = HandCmd()
                for i in range(min(len(msg.motor_cmd), 7)):
                    ros2_cmd.motor_cmd[i].mode = int(msg.motor_cmd[i].mode)
                    ros2_cmd.motor_cmd[i].q = float(msg.motor_cmd[i].q)
                    ros2_cmd.motor_cmd[i].dq = float(msg.motor_cmd[i].dq)
                    ros2_cmd.motor_cmd[i].tau = float(msg.motor_cmd[i].tau)
                    ros2_cmd.motor_cmd[i].kp = float(msg.motor_cmd[i].kp)
                    ros2_cmd.motor_cmd[i].kd = float(msg.motor_cmd[i].kd)
                self.node.right_hand_cmd_pub.publish(ros2_cmd)
                self.stats["ros2_writes"] += 1

            elif topic == "rt/brainco/left/cmd" and self.node and self.node.brainco_left_cmd_pub:
                self.node.brainco_left_cmd_pub.publish(
                    self._build_joint_state(brainco_command_to_stark(msg.position))
                )
                self.stats["ros2_writes"] += 1

            elif topic == "rt/brainco/right/cmd" and self.node and self.node.brainco_right_cmd_pub:
                self.node.brainco_right_cmd_pub.publish(
                    self._build_joint_state(brainco_command_to_stark(msg.position))
                )
                self.stats["ros2_writes"] += 1

        except Exception as e:
            self.stats["errors"] += 1
            if self.config.verbose:
                print(f"[Bridge] Forward command error ({topic}): {e}")

    def _build_joint_state(self, position) -> JointState:
        js = JointState()
        js.header.stamp = self.node.get_clock().now().to_msg()
        js.name = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]
        js.position = [float(v) for v in position]
        return js

    def _publish_lowcmd_with_crc(self):
        """Compute CRC over LowCmd via DDS struct and publish to ROS2 /lowcmd."""
        if self.last_lowcmd is None or self.node is None:
            return
        # This intentionally allocates a fresh DDS object every tick. Reusing it
        # was part of the bridge optimization reverted after unsafe real-robot
        # motion (9ceaff6).
        dds = unitree_hg_msg_dds__LowCmd_()
        ros = self.last_lowcmd
        dds.mode_pr = ros.mode_pr
        dds.mode_machine = ros.mode_machine
        for i in range(35):
            dds.motor_cmd[i].mode = ros.motor_cmd[i].mode
            dds.motor_cmd[i].q = ros.motor_cmd[i].q
            dds.motor_cmd[i].dq = ros.motor_cmd[i].dq
            dds.motor_cmd[i].tau = ros.motor_cmd[i].tau
            dds.motor_cmd[i].kp = ros.motor_cmd[i].kp
            dds.motor_cmd[i].kd = ros.motor_cmd[i].kd
            dds.motor_cmd[i].reserve = ros.motor_cmd[i].reserve
        for i in range(4):
            dds.reserve[i] = ros.reserve[i]
        crc_value = self.crc.Crc(dds)
        ros.crc = int(crc_value)
        self.node.lowcmd_pub.publish(ros)
        self.stats["ros2_writes"] += 1

    def run(self):
        print("[Bridge] Starting main loop...")
        self.running = True
        loop_period = 1.0 / self.config.loop_rate
        last_stat_time = time.time()
        zmq_send_interval = 1.0 / 500.0
        last_zmq_send_time = 0.0

        # IDs of brainco state we last published, to avoid resending unchanged frames
        last_brainco_left_id = None
        last_brainco_right_id = None
        last_left_hand_id = None
        last_right_hand_id = None

        while self.running:
            loop_start = time.time()

            # --- State path: read caches populated by ROS2 callbacks, send via ZMQ/UDP ---
            now = time.time()

            # UDP low-latency: send on every change (callback set the flag)
            if (
                self.config.enable_udp_state
                and self.udp_socket
                and self.node is not None
                and self.node.cached_lowstate is not None
                and self.node.cached_lowstate_changed
            ):
                self.send_state_udp(
                    self.node.cached_lowstate, self.node.cached_secondary_imu
                )
                self.node.cached_lowstate_changed = False

            # ZMQ rate-limited (500Hz) send for the cached state. Keep all ZMQ
            # socket access on this thread; callback-thread publishing was part
            # of the unsafe bridge optimization reverted by 9ceaff6.
            if now - last_zmq_send_time >= zmq_send_interval:
                last_zmq_send_time = now
                if self.node is not None:
                    if self.node.cached_lowstate is not None:
                        self.publish_state_zmq("rt/lowstate", self.node.cached_lowstate)
                    if self.node.cached_secondary_imu is not None:
                        self.publish_state_zmq(
                            "rt/secondary_imu", self.node.cached_secondary_imu
                        )
                    if (
                        self.config.enable_odometry
                        and self.node.cached_odometry is not None
                    ):
                        try:
                            payload = sport_mode_state_to_odo_payload(
                                self.node.cached_odometry
                            )
                            self.publish_state_zmq(
                                "rt/odostate",
                                ZmqOdoState(**payload),
                            )
                        except Exception as exc:
                            self.stats["errors"] += 1
                            if self.config.verbose:
                                print(
                                    f"[Bridge] odometry conversion error: {exc}"
                                )

            # Hand state: send only on new frame (lower rate, identity by Python id())
            if self.node is not None:
                if (
                    self.node.cached_left_hand is not None
                    and id(self.node.cached_left_hand) != last_left_hand_id
                ):
                    self.publish_state_zmq(
                        "rt/dex3/left/state", self.node.cached_left_hand
                    )
                    last_left_hand_id = id(self.node.cached_left_hand)
                if (
                    self.node.cached_right_hand is not None
                    and id(self.node.cached_right_hand) != last_right_hand_id
                ):
                    self.publish_state_zmq(
                        "rt/dex3/right/state", self.node.cached_right_hand
                    )
                    last_right_hand_id = id(self.node.cached_right_hand)
                if (
                    self.node.cached_brainco_left is not None
                    and id(self.node.cached_brainco_left) != last_brainco_left_id
                ):
                    self.publish_state_zmq(
                        "rt/brainco/left/state", self.node.cached_brainco_left
                    )
                    last_brainco_left_id = id(self.node.cached_brainco_left)
                if (
                    self.node.cached_brainco_right is not None
                    and id(self.node.cached_brainco_right) != last_brainco_right_id
                ):
                    self.publish_state_zmq(
                        "rt/brainco/right/state", self.node.cached_brainco_right
                    )
                    last_brainco_right_id = id(self.node.cached_brainco_right)

            # Drain the non-blocking command queue and retain the newest message
            # for each topic before forwarding.
            latest_cmds = {}
            while True:
                cmd_result = self.receive_command()
                if cmd_result:
                    topic, data = cmd_result
                    latest_cmds[topic] = (topic, data)
                else:
                    break
            for topic, data in latest_cmds.values():
                self.forward_command(topic, data)

            # High-rate lowcmd repeat (CRC computed each time)
            if self.last_lowcmd is not None:
                self._publish_lowcmd_with_crc()

            # Stats every 5s
            now = time.time()
            if now - last_stat_time > 5.0:
                dt = now - last_stat_time
                ros2_rate = (
                    self.stats["ros2_writes"] - getattr(self, "_last_ros2", 0)
                ) / dt
                cmd_rate = (self.stats["cmd_rx"] - getattr(self, "_last_cmd", 0)) / dt
                state_rate = (
                    self.stats["state_tx"] - getattr(self, "_last_state", 0)
                ) / dt
                self._last_ros2 = self.stats["ros2_writes"]
                self._last_cmd = self.stats["cmd_rx"]
                self._last_state = self.stats["state_tx"]
                print(
                    f"[Bridge] state_tx: {state_rate:.0f}Hz, cmd_rx: {cmd_rate:.0f}Hz, "
                    f"ros2_writes: {ros2_rate:.0f}Hz | totals: state={self.stats['state_tx']}, "
                    f"cmd={self.stats['cmd_rx']}, ros2={self.stats['ros2_writes']}, "
                    f"err={self.stats['errors']}"
                )
                last_stat_time = now

            elapsed = time.time() - loop_start
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)

    def stop(self):
        print("[Bridge] Stopping...")
        self.running = False

    def cleanup(self):
        if self.state_pub:
            self.state_pub.close()
        if self.cmd_sub:
            self.cmd_sub.close()
        if self.context:
            self.context.term()
        if self.node is not None:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _release_motion_switcher():
    """Best-effort: ask Unitree main service to release any active control mode."""
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        # ChannelFactoryInitialize is a no-op if rclpy already initialized CycloneDDS;
        # we still try, to support standalone invocation.
        try:
            ChannelFactoryInitialize(
                0, os.environ.get("UNITREE_NETWORK_INTERFACE", "eth0")
            )
        except Exception:
            pass
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        for _ in range(5):
            status, result = msc.CheckMode()
            mode_name = (result or {}).get("name", "")
            if mode_name:
                print(f"[Bridge] Found active mode: '{mode_name}', releasing...")
                msc.ReleaseMode()
                time.sleep(1)
            else:
                print("[Bridge] No active control mode")
                return
    except Exception as e:
        print(f"[Bridge] Warning: Could not check/release control mode: {e}")


def main():
    parser = argparse.ArgumentParser(description="ZMQ-ROS2 Bridge for Unitree G1 + BrainCo")
    parser.add_argument("--check", action="store_true", help="Check imported dependencies without opening sockets or initializing ROS")
    parser.add_argument("--enable-control", action="store_true", help="Explicitly enable live robot command forwarding")
    parser.add_argument("--state-port", type=int, default=5560)
    parser.add_argument("--cmd-port", type=int, default=5561)
    parser.add_argument("--rate", type=float, default=500.0, help="Main loop rate Hz")
    parser.add_argument("--no-dex3", action="store_true", help="Disable dex3 hand bridging")
    parser.add_argument("--no-brainco", action="store_true", help="Disable brainco hand bridging")
    parser.add_argument("--no-release-mode", action="store_true", help="Skip MotionSwitcher release")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--udp-target", type=str, default="", help="Target IP for UDP state")
    parser.add_argument("--udp-port", type=int, default=5562)
    parser.add_argument("--no-udp", action="store_true")
    parser.add_argument(
        "--no-odometry",
        action="store_true",
        help="Disable /sportmodestate -> rt/odostate forwarding",
    )
    parser.add_argument(
        "--odometry-topic",
        default="/sportmodestate",
        help="ROS2 unitree_go/SportModeState topic",
    )
    args = parser.parse_args()
    if args.check:
        if not HAS_BRAINCO or not HAS_SPORT_MODE_STATE:
            parser.error("BrainCo and Unitree state messages must be built and sourced")
        try:
            CRC().Crc(unitree_hg_msg_dds__LowCmd_())
        except (OSError, TypeError) as exc:
            parser.error(f"Unitree native CRC check failed: {exc}; install the SDK in editable mode")
        print("Bridge dependencies and native CRC verified; no ROS node, network or hardware initialized.")
        return
    if not args.enable_control:
        parser.error("Live robot control requires --enable-control; use --check for installation verification")

    config = BridgeConfig(
        state_port=args.state_port,
        cmd_port=args.cmd_port,
        udp_state_port=args.udp_port,
        loop_rate=args.rate,
        enable_dex3=not args.no_dex3,
        enable_brainco=not args.no_brainco,
        verbose=args.verbose,
        enable_udp_state=not args.no_udp and bool(args.udp_target),
        udp_target_ip=args.udp_target,
        enable_odometry=not args.no_odometry,
        odometry_topic=args.odometry_topic,
    )

    bridge = ZmqRos2Bridge(config)

    def signal_handler(_sig, _frame):
        print("\n[Bridge] Received shutdown signal")
        bridge.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not bridge.init_zmq():
        print("[Bridge] Failed to initialize ZMQ")
        sys.exit(1)
    # Release MotionSwitcher BEFORE init_ros2: rclpy locks CycloneDDS domain 0,
    # after which unitree_sdk2py.ChannelFactoryInitialize fails. When invoked via
    # robot_services.sh, --no-release-mode is set and release is done in a
    # short-lived subprocess instead (so the domain is freed before rclpy starts).
    if not args.no_release_mode:
        print("[Bridge] Checking and releasing control mode...")
        _release_motion_switcher()
    if not bridge.init_ros2():
        print("[Bridge] Failed to initialize ROS2")
        bridge.cleanup()
        sys.exit(1)
    if config.enable_udp_state and config.udp_target_ip:
        if not bridge.init_udp(config.udp_target_ip):
            print("[Bridge] Warning: UDP init failed, continuing without UDP")
            config.enable_udp_state = False
        else:
            print(
                f"[Bridge] UDP state transmission enabled to "
                f"{config.udp_target_ip}:{config.udp_state_port}"
            )

    try:
        bridge.run()
    finally:
        bridge.cleanup()
        print("[Bridge] Shutdown complete")


if __name__ == "__main__":
    main()
