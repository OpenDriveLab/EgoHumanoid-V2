"""ZMQ Message Serializer for Unitree DDS Messages.

This module provides serialization/deserialization for DDS messages using msgpack.
It creates duck-typed compatible classes that can be used interchangeably with
the original DDS message types.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

import msgpack


# =============================================================================
# Duck-typed message classes (compatible with DDS message API)
# =============================================================================


@dataclass
class ZmqMotorState:
    """Duck-typed MotorState_ compatible class."""

    mode: int = 0
    q: float = 0.0
    dq: float = 0.0
    ddq: float = 0.0
    tau_est: float = 0.0
    temperature: List[int] = field(default_factory=lambda: [0, 0])
    vol: float = 0.0
    sensor: List[int] = field(default_factory=lambda: [0, 0])
    motorstate: int = 0
    reserve: List[int] = field(default_factory=lambda: [0, 0, 0, 0])


@dataclass
class ZmqMotorCmd:
    """Duck-typed MotorCmd_ compatible class."""

    mode: int = 0
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    reserve: int = 0


@dataclass
class ZmqIMUState:
    """Duck-typed IMUState_ compatible class."""

    quaternion: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    gyroscope: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    accelerometer: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    temperature: int = 0


@dataclass
class ZmqPressSensorState:
    """Duck-typed PressSensorState_ compatible class."""

    pressure: List[float] = field(default_factory=lambda: [0.0] * 12)
    temperature: List[float] = field(default_factory=lambda: [0.0] * 12)
    lost: int = 0
    reserve: int = 0


@dataclass
class ZmqLowState:
    """Duck-typed LowState_ compatible class."""

    version: List[int] = field(default_factory=lambda: [0, 0])
    mode_pr: int = 0
    mode_machine: int = 0
    tick: int = 0
    imu_state: ZmqIMUState = field(default_factory=ZmqIMUState)
    motor_state: List[ZmqMotorState] = field(
        default_factory=lambda: [ZmqMotorState() for _ in range(35)]
    )
    wireless_remote: List[int] = field(default_factory=lambda: [0] * 40)
    reserve: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    crc: int = 0


@dataclass
class ZmqLowCmd:
    """Duck-typed LowCmd_ compatible class."""

    mode_pr: int = 0
    mode_machine: int = 0
    motor_cmd: List[ZmqMotorCmd] = field(
        default_factory=lambda: [ZmqMotorCmd() for _ in range(35)]
    )
    reserve: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    crc: int = 0


@dataclass
class ZmqHandState:
    """Duck-typed HandState_ compatible class."""

    motor_state: List[ZmqMotorState] = field(
        default_factory=lambda: [ZmqMotorState() for _ in range(7)]
    )
    press_sensor_state: List[ZmqPressSensorState] = field(default_factory=list)
    imu_state: ZmqIMUState = field(default_factory=ZmqIMUState)
    power_v: float = 0.0
    power_a: float = 0.0
    system_v: float = 0.0
    device_v: float = 0.0
    error: List[int] = field(default_factory=lambda: [0, 0])
    reserve: List[int] = field(default_factory=lambda: [0, 0])


@dataclass
class ZmqHandCmd:
    """Duck-typed HandCmd_ compatible class."""

    motor_cmd: List[ZmqMotorCmd] = field(
        default_factory=lambda: [ZmqMotorCmd() for _ in range(7)]
    )
    reserve: List[int] = field(default_factory=lambda: [0, 0, 0, 0])


@dataclass
class ZmqOdoState:
    """Duck-typed OdoState_ (sim base pose/vel from MuJoCo)."""

    version: List[int] = field(default_factory=lambda: [0, 0])
    tick: int = 0
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    linear_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    crc: int = 0


def sport_mode_state_to_odo_payload(msg) -> Dict[str, Any]:
    """Map ROS2 unitree_go/SportModeState into the ZmqOdoState schema."""

    stamp = msg.stamp
    seconds = int(getattr(stamp, "sec", 0))
    nanoseconds = int(getattr(stamp, "nanosec", getattr(stamp, "nsec", 0)))
    tick = seconds * 1_000_000_000 + nanoseconds
    quaternion_wxyz = [float(value) for value in msg.imu_state.quaternion]
    if len(quaternion_wxyz) != 4:
        raise ValueError("SportModeState IMU quaternion must have four values")
    position = [float(value) for value in msg.position]
    velocity = [float(value) for value in msg.velocity]
    if len(position) != 3 or len(velocity) != 3:
        raise ValueError("SportModeState position/velocity must have three values")
    return {
        "version": [1, 0],
        "tick": tick if tick > 0 else time.time_ns(),
        "position": position,
        # Unitree IMU messages are wxyz; OdoState's generated field is xyzw.
        "orientation": quaternion_wxyz[1:] + quaternion_wxyz[:1],
        "linear_velocity": velocity,
        "angular_velocity": [0.0, 0.0, float(msg.yaw_speed)],
        "crc": 0,
    }


@dataclass
class ZmqBraincoMotorStatus:
    """BrainCo Revo2 6-DOF motor status (positions/speeds 0-100, currents mA, states)."""

    positions: List[int] = field(default_factory=lambda: [0] * 6)
    speeds: List[int] = field(default_factory=lambda: [0] * 6)
    currents: List[int] = field(default_factory=lambda: [0] * 6)
    states: List[int] = field(default_factory=lambda: [0] * 6)


@dataclass
class ZmqJointCmd:
    """Generic joint position command (e.g. brainco hand 6-DOF positions in 0~1)."""

    position: List[float] = field(default_factory=lambda: [0.0] * 6)


@dataclass
class ZmqBraincoAppliedCommand:
    """Robot-applied two-hand command with robot clock timestamps."""

    sequence: int = 0
    monotonic_ns: int = 0
    wall_time_ns: int = 0
    left: List[float] = field(default_factory=lambda: [0.0] * 6)
    right: List[float] = field(default_factory=lambda: [0.0] * 6)


# =============================================================================
# Serialization functions (DDS -> dict -> msgpack)
# =============================================================================


def motor_state_to_dict(state) -> Dict[str, Any]:
    """Convert MotorState_ to dict."""
    return {
        "mode": int(state.mode),
        "q": float(state.q),
        "dq": float(state.dq),
        "ddq": float(state.ddq),
        "tau_est": float(state.tau_est),
        "temperature": list(state.temperature),
        "vol": float(state.vol),
        "sensor": list(state.sensor),
        "motorstate": int(state.motorstate),
        "reserve": list(state.reserve),
    }


def motor_cmd_to_dict(cmd) -> Dict[str, Any]:
    """Convert MotorCmd_ to dict."""
    return {
        "mode": int(cmd.mode),
        "q": float(cmd.q),
        "dq": float(cmd.dq),
        "tau": float(cmd.tau),
        "kp": float(cmd.kp),
        "kd": float(cmd.kd),
        "reserve": int(cmd.reserve),
    }


def imu_state_to_dict(state) -> Dict[str, Any]:
    """Convert IMUState_ to dict."""
    return {
        "quaternion": list(state.quaternion),
        "gyroscope": list(state.gyroscope),
        "accelerometer": list(state.accelerometer),
        "rpy": list(state.rpy),
        "temperature": int(state.temperature),
    }


def press_sensor_state_to_dict(state) -> Dict[str, Any]:
    """Convert PressSensorState_ to dict."""
    return {
        "pressure": list(state.pressure),
        "temperature": list(state.temperature),
        "lost": int(state.lost),
        "reserve": int(state.reserve),
    }


def low_state_to_dict(state) -> Dict[str, Any]:
    """Convert LowState_ to dict."""
    return {
        "version": list(state.version),
        "mode_pr": int(state.mode_pr),
        "mode_machine": int(state.mode_machine),
        "tick": int(state.tick),
        "imu_state": imu_state_to_dict(state.imu_state),
        "motor_state": [motor_state_to_dict(m) for m in state.motor_state],
        "wireless_remote": list(state.wireless_remote),
        "reserve": list(state.reserve),
        "crc": int(state.crc),
    }


def low_cmd_to_dict(cmd) -> Dict[str, Any]:
    """Convert LowCmd_ to dict."""
    return {
        "mode_pr": int(cmd.mode_pr),
        "mode_machine": int(cmd.mode_machine),
        "motor_cmd": [motor_cmd_to_dict(m) for m in cmd.motor_cmd],
        "reserve": list(cmd.reserve),
        "crc": int(cmd.crc),
    }


def hand_state_to_dict(state) -> Dict[str, Any]:
    """Convert HandState_ to dict."""
    return {
        "motor_state": [motor_state_to_dict(m) for m in state.motor_state],
        "press_sensor_state": [press_sensor_state_to_dict(p) for p in state.press_sensor_state],
        "imu_state": imu_state_to_dict(state.imu_state),
        "power_v": float(state.power_v),
        "power_a": float(state.power_a),
        "system_v": float(state.system_v),
        "device_v": float(state.device_v),
        "error": list(state.error),
        "reserve": list(state.reserve),
    }


def hand_cmd_to_dict(cmd) -> Dict[str, Any]:
    """Convert HandCmd_ to dict."""
    return {
        "motor_cmd": [motor_cmd_to_dict(m) for m in cmd.motor_cmd],
        "reserve": list(cmd.reserve),
    }


def odo_state_to_dict(msg) -> Dict[str, Any]:
    """Convert OdoState_ (sim base pose/vel) to dict."""
    return {
        "version": [int(v) for v in msg.version],
        "tick": int(msg.tick),
        "position": [float(v) for v in msg.position],
        "orientation": [float(v) for v in msg.orientation],
        "linear_velocity": [float(v) for v in msg.linear_velocity],
        "angular_velocity": [float(v) for v in msg.angular_velocity],
        "crc": int(msg.crc),
    }


def dict_to_odo_state(data: Dict[str, Any]) -> ZmqOdoState:
    return ZmqOdoState(
        version=list(data.get("version", [0, 0])),
        tick=int(data.get("tick", 0)),
        position=list(data.get("position", [0.0, 0.0, 0.0])),
        orientation=list(data.get("orientation", [0.0, 0.0, 0.0, 1.0])),
        linear_velocity=list(data.get("linear_velocity", [0.0, 0.0, 0.0])),
        angular_velocity=list(data.get("angular_velocity", [0.0, 0.0, 0.0])),
        crc=int(data.get("crc", 0)),
    )


def brainco_motor_status_to_dict(msg) -> Dict[str, Any]:
    """Convert ros2_stark_interfaces/MotorStatus (or duck-typed) to dict."""
    return {
        "positions": [int(v) for v in msg.positions],
        "speeds": [int(v) for v in msg.speeds],
        "currents": [int(v) for v in msg.currents],
        "states": [int(v) for v in msg.states],
    }


def joint_cmd_to_dict(cmd) -> Dict[str, Any]:
    """Convert ZmqJointCmd to dict."""
    return {"position": [float(v) for v in cmd.position]}


def brainco_applied_command_to_dict(msg) -> Dict[str, Any]:
    return {
        "sequence": int(msg.sequence),
        "monotonic_ns": int(msg.monotonic_ns),
        "wall_time_ns": int(msg.wall_time_ns),
        "left": [float(value) for value in msg.left],
        "right": [float(value) for value in msg.right],
    }


# =============================================================================
# Deserialization functions (msgpack -> dict -> duck-typed class)
# =============================================================================


def dict_to_motor_state(data: Dict[str, Any]) -> ZmqMotorState:
    """Convert dict to ZmqMotorState."""
    return ZmqMotorState(
        mode=data.get("mode", 0),
        q=data.get("q", 0.0),
        dq=data.get("dq", 0.0),
        ddq=data.get("ddq", 0.0),
        tau_est=data.get("tau_est", 0.0),
        temperature=data.get("temperature", [0, 0]),
        vol=data.get("vol", 0.0),
        sensor=data.get("sensor", [0, 0]),
        motorstate=data.get("motorstate", 0),
        reserve=data.get("reserve", [0, 0, 0, 0]),
    )


def dict_to_motor_cmd(data: Dict[str, Any]) -> ZmqMotorCmd:
    """Convert dict to ZmqMotorCmd."""
    return ZmqMotorCmd(
        mode=data.get("mode", 0),
        q=data.get("q", 0.0),
        dq=data.get("dq", 0.0),
        tau=data.get("tau", 0.0),
        kp=data.get("kp", 0.0),
        kd=data.get("kd", 0.0),
        reserve=data.get("reserve", 0),
    )


def dict_to_imu_state(data: Dict[str, Any]) -> ZmqIMUState:
    """Convert dict to ZmqIMUState."""
    return ZmqIMUState(
        quaternion=data.get("quaternion", [1.0, 0.0, 0.0, 0.0]),
        gyroscope=data.get("gyroscope", [0.0, 0.0, 0.0]),
        accelerometer=data.get("accelerometer", [0.0, 0.0, 0.0]),
        rpy=data.get("rpy", [0.0, 0.0, 0.0]),
        temperature=data.get("temperature", 0),
    )


def dict_to_press_sensor_state(data: Dict[str, Any]) -> ZmqPressSensorState:
    """Convert dict to ZmqPressSensorState."""
    return ZmqPressSensorState(
        pressure=data.get("pressure", [0.0] * 12),
        temperature=data.get("temperature", [0.0] * 12),
        lost=data.get("lost", 0),
        reserve=data.get("reserve", 0),
    )


def dict_to_low_state(data: Dict[str, Any]) -> ZmqLowState:
    """Convert dict to ZmqLowState."""
    motor_states = [dict_to_motor_state(m) for m in data.get("motor_state", [])]
    # Ensure we have 35 motor states
    while len(motor_states) < 35:
        motor_states.append(ZmqMotorState())

    return ZmqLowState(
        version=data.get("version", [0, 0]),
        mode_pr=data.get("mode_pr", 0),
        mode_machine=data.get("mode_machine", 0),
        tick=data.get("tick", 0),
        imu_state=dict_to_imu_state(data.get("imu_state", {})),
        motor_state=motor_states,
        wireless_remote=data.get("wireless_remote", [0] * 40),
        reserve=data.get("reserve", [0, 0, 0, 0]),
        crc=data.get("crc", 0),
    )


def dict_to_low_cmd(data: Dict[str, Any]) -> ZmqLowCmd:
    """Convert dict to ZmqLowCmd."""
    motor_cmds = [dict_to_motor_cmd(m) for m in data.get("motor_cmd", [])]
    # Ensure we have 35 motor commands
    while len(motor_cmds) < 35:
        motor_cmds.append(ZmqMotorCmd())

    return ZmqLowCmd(
        mode_pr=data.get("mode_pr", 0),
        mode_machine=data.get("mode_machine", 0),
        motor_cmd=motor_cmds,
        reserve=data.get("reserve", [0, 0, 0, 0]),
        crc=data.get("crc", 0),
    )


def dict_to_hand_state(data: Dict[str, Any]) -> ZmqHandState:
    """Convert dict to ZmqHandState."""
    motor_states = [dict_to_motor_state(m) for m in data.get("motor_state", [])]
    # Ensure we have at least 7 motor states for hand
    while len(motor_states) < 7:
        motor_states.append(ZmqMotorState())

    press_sensor_states = [
        dict_to_press_sensor_state(p) for p in data.get("press_sensor_state", [])
    ]

    return ZmqHandState(
        motor_state=motor_states,
        press_sensor_state=press_sensor_states,
        imu_state=dict_to_imu_state(data.get("imu_state", {})),
        power_v=data.get("power_v", 0.0),
        power_a=data.get("power_a", 0.0),
        system_v=data.get("system_v", 0.0),
        device_v=data.get("device_v", 0.0),
        error=data.get("error", [0, 0]),
        reserve=data.get("reserve", [0, 0]),
    )


def dict_to_hand_cmd(data: Dict[str, Any]) -> ZmqHandCmd:
    """Convert dict to ZmqHandCmd."""
    motor_cmds = [dict_to_motor_cmd(m) for m in data.get("motor_cmd", [])]
    # Ensure we have at least 7 motor commands for hand
    while len(motor_cmds) < 7:
        motor_cmds.append(ZmqMotorCmd())

    return ZmqHandCmd(
        motor_cmd=motor_cmds,
        reserve=data.get("reserve", [0, 0, 0, 0]),
    )


def dict_to_brainco_motor_status(data: Dict[str, Any]) -> ZmqBraincoMotorStatus:
    """Convert dict to ZmqBraincoMotorStatus."""
    return ZmqBraincoMotorStatus(
        positions=list(data.get("positions", [0] * 6)),
        speeds=list(data.get("speeds", [0] * 6)),
        currents=list(data.get("currents", [0] * 6)),
        states=list(data.get("states", [0] * 6)),
    )


def dict_to_joint_cmd(data: Dict[str, Any]) -> ZmqJointCmd:
    """Convert dict to ZmqJointCmd."""
    return ZmqJointCmd(position=list(data.get("position", [0.0] * 6)))


def dict_to_brainco_applied_command(
    data: Dict[str, Any],
) -> ZmqBraincoAppliedCommand:
    return ZmqBraincoAppliedCommand(
        sequence=int(data.get("sequence", 0)),
        monotonic_ns=int(data.get("monotonic_ns", 0)),
        wall_time_ns=int(data.get("wall_time_ns", 0)),
        left=[float(value) for value in data.get("left", [0.0] * 6)],
        right=[float(value) for value in data.get("right", [0.0] * 6)],
    )


# =============================================================================
# High-level serialization API
# =============================================================================


def _msgpack_default(obj):
    """Convert non-native types (e.g. numpy scalars from rclpy messages) to native."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Cannot serialize object of type {type(obj)!r}")


def serialize_message(topic: str, msg) -> bytes:
    """Serialize a DDS or ROS2 message to msgpack bytes.

    Args:
        topic: The DDS topic name (used to determine message type)
        msg: The message object (unitree_sdk2py DDS class or rclpy ROS2 message)

    Returns:
        msgpack-encoded bytes
    """
    topic_l = topic.lower()
    if topic_l == "rt/brainco/glove/applied":
        data = brainco_applied_command_to_dict(msg)
    elif "brainco" in topic_l and "state" in topic_l:
        data = brainco_motor_status_to_dict(msg)
    elif "brainco" in topic_l and "cmd" in topic_l:
        data = joint_cmd_to_dict(msg)
    elif "lowstate" in topic_l:
        data = low_state_to_dict(msg)
    elif "lowcmd" in topic_l:
        data = low_cmd_to_dict(msg)
    elif "secondary_imu" in topic_l:
        data = imu_state_to_dict(msg)
    elif "odostate" in topic_l:
        data = odo_state_to_dict(msg)
    elif "state" in topic_l and "dex3" in topic_l:
        data = hand_state_to_dict(msg)
    elif "cmd" in topic_l and "dex3" in topic_l:
        data = hand_cmd_to_dict(msg)
    else:
        raise ValueError(f"Unknown topic: {topic}")

    return msgpack.packb(data, use_bin_type=True, default=_msgpack_default)


def deserialize_message(topic: str, data: bytes):
    """Deserialize msgpack bytes to a duck-typed message object.

    Args:
        topic: The DDS topic name (used to determine message type)
        data: msgpack-encoded bytes

    Returns:
        Duck-typed message object compatible with DDS message API
    """
    unpacked = msgpack.unpackb(data, raw=False)

    topic_l = topic.lower()
    if topic_l == "rt/brainco/glove/applied":
        return dict_to_brainco_applied_command(unpacked)
    elif "brainco" in topic_l and "state" in topic_l:
        return dict_to_brainco_motor_status(unpacked)
    elif "brainco" in topic_l and "cmd" in topic_l:
        return dict_to_joint_cmd(unpacked)
    elif "lowstate" in topic_l:
        return dict_to_low_state(unpacked)
    elif "lowcmd" in topic_l:
        return dict_to_low_cmd(unpacked)
    elif "secondary_imu" in topic_l:
        return dict_to_imu_state(unpacked)
    elif "odostate" in topic_l:
        return dict_to_odo_state(unpacked)
    elif "state" in topic_l and "dex3" in topic_l:
        return dict_to_hand_state(unpacked)
    elif "cmd" in topic_l and "dex3" in topic_l:
        return dict_to_hand_cmd(unpacked)
    else:
        raise ValueError(f"Unknown topic: {topic}")


# =============================================================================
# Topic-to-type mapping
# =============================================================================


# Map topic names to their corresponding Zmq message classes
TOPIC_TO_ZMQ_TYPE = {
    "rt/lowstate": ZmqLowState,
    "rt/lowcmd": ZmqLowCmd,
    "rt/secondary_imu": ZmqIMUState,
    "rt/odostate": ZmqOdoState,
    "rt/dex3/left/state": ZmqHandState,
    "rt/dex3/right/state": ZmqHandState,
    "rt/dex3/left/cmd": ZmqHandCmd,
    "rt/dex3/right/cmd": ZmqHandCmd,
    "rt/brainco/left/state": ZmqBraincoMotorStatus,
    "rt/brainco/right/state": ZmqBraincoMotorStatus,
    "rt/brainco/left/cmd": ZmqJointCmd,
    "rt/brainco/right/cmd": ZmqJointCmd,
    "rt/brainco/glove/applied": ZmqBraincoAppliedCommand,
}


def get_zmq_type_for_topic(topic: str):
    """Get the ZMQ message type class for a topic.

    Args:
        topic: DDS topic name

    Returns:
        ZMQ message type class
    """
    if topic in TOPIC_TO_ZMQ_TYPE:
        return TOPIC_TO_ZMQ_TYPE[topic]

    # Fallback matching
    topic_lower = topic.lower()
    if topic_lower == "rt/brainco/glove/applied":
        return ZmqBraincoAppliedCommand
    elif "brainco" in topic_lower and "state" in topic_lower:
        return ZmqBraincoMotorStatus
    elif "brainco" in topic_lower and "cmd" in topic_lower:
        return ZmqJointCmd
    elif "lowstate" in topic_lower:
        return ZmqLowState
    elif "lowcmd" in topic_lower:
        return ZmqLowCmd
    elif "secondary_imu" in topic_lower:
        return ZmqIMUState
    elif "odostate" in topic_lower:
        return ZmqOdoState
    elif "state" in topic_lower and "dex3" in topic_lower:
        return ZmqHandState
    elif "cmd" in topic_lower and "dex3" in topic_lower:
        return ZmqHandCmd

    raise ValueError(f"Unknown topic: {topic}")
