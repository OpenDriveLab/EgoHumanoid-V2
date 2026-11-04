from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
import time
from typing import Any, Mapping

import numpy as np


BODY_POSE_SHAPE = (24, 7)
BODY_VELOCITY_SHAPE = (24, 6)
BODY_JOINT_TIMESTAMP_SHAPE = (24,)
HAND_POSE_SHAPE = (26, 7)
DEVICE_POSE_SHAPE = (7,)
CHEST_JOINT_INDEX = 9
HEAD_JOINT_INDEX = 15


def _identity_pose(shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    out[:, 6] = 1.0
    return out


def _coerce_array(
    value: Any,
    shape: tuple[int, ...],
    *,
    default: np.ndarray | None = None,
    name: str,
) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return default.astype(np.float32, copy=True)

    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr.copy()


def _coerce_int_array(
    value: Any,
    shape: tuple[int, ...],
    *,
    default: np.ndarray | None = None,
    name: str,
) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return default.astype(np.int64, copy=True)

    arr = np.asarray(value, dtype=np.int64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr.copy()


@dataclass(frozen=True)
class PicoSample:
    body_pose: np.ndarray
    body_available: bool
    body_timestamp_ns: int
    body_joint_timestamp_ns: np.ndarray
    body_velocity: np.ndarray
    left_controller_pose: np.ndarray
    right_controller_pose: np.ndarray
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    left_hand_active: bool
    right_hand_active: bool
    local_timestamp_ns: int

    def to_lerobot_frame(self, task: str) -> dict[str, np.ndarray | str]:
        return {
            "observation.pico.body_pose": self.body_pose.astype(np.float32, copy=False),
            "observation.pico.body_available": np.asarray([self.body_available], dtype=np.bool_),
            "observation.pico.body_timestamp_ns": np.asarray(
                [self.body_timestamp_ns], dtype=np.int64
            ),
            "observation.pico.body_joint_timestamp_ns": self.body_joint_timestamp_ns.astype(
                np.int64, copy=False
            ),
            "observation.pico.body_velocity": self.body_velocity.astype(np.float32, copy=False),
            "observation.pico.local_timestamp_ns": np.asarray(
                [self.local_timestamp_ns], dtype=np.int64
            ),
            "observation.pico.left_controller_pose": self.left_controller_pose.astype(
                np.float32, copy=False
            ),
            "observation.pico.right_controller_pose": self.right_controller_pose.astype(
                np.float32, copy=False
            ),
            "observation.pico.left_hand_pose": self.left_hand_pose.astype(np.float32, copy=False),
            "observation.pico.right_hand_pose": self.right_hand_pose.astype(np.float32, copy=False),
            "observation.pico.left_hand_active": np.asarray(
                [self.left_hand_active], dtype=np.bool_
            ),
            "observation.pico.right_hand_active": np.asarray(
                [self.right_hand_active], dtype=np.bool_
            ),
            "task": task,
        }


def coerce_pico_sample(sample: Mapping[str, Any]) -> PicoSample:
    """Validate and normalize raw Pico SDK values into one LeRobot-ready sample."""

    body_pose = _coerce_array(
        sample.get("body_pose"),
        BODY_POSE_SHAPE,
        default=_identity_pose(BODY_POSE_SHAPE),
        name="body_pose",
    )
    body_joint_timestamp_ns = _coerce_int_array(
        sample.get("body_joint_timestamp_ns"),
        BODY_JOINT_TIMESTAMP_SHAPE,
        default=np.zeros(BODY_JOINT_TIMESTAMP_SHAPE, dtype=np.int64),
        name="body_joint_timestamp_ns",
    )
    body_velocity = _coerce_array(
        sample.get("body_velocity"),
        BODY_VELOCITY_SHAPE,
        default=np.zeros(BODY_VELOCITY_SHAPE, dtype=np.float32),
        name="body_velocity",
    )
    zero_device = np.zeros(DEVICE_POSE_SHAPE, dtype=np.float32)
    left_controller = _coerce_array(
        sample.get("left_controller_pose"),
        DEVICE_POSE_SHAPE,
        default=zero_device,
        name="left_controller_pose",
    )
    right_controller = _coerce_array(
        sample.get("right_controller_pose"),
        DEVICE_POSE_SHAPE,
        default=zero_device,
        name="right_controller_pose",
    )
    left_hand = _coerce_array(
        sample.get("left_hand_pose"),
        HAND_POSE_SHAPE,
        default=_identity_pose(HAND_POSE_SHAPE),
        name="left_hand_pose",
    )
    right_hand = _coerce_array(
        sample.get("right_hand_pose"),
        HAND_POSE_SHAPE,
        default=_identity_pose(HAND_POSE_SHAPE),
        name="right_hand_pose",
    )

    return PicoSample(
        body_pose=body_pose,
        body_available=bool(sample.get("body_available", False)),
        body_timestamp_ns=int(sample.get("body_timestamp_ns", 0)),
        body_joint_timestamp_ns=body_joint_timestamp_ns,
        body_velocity=body_velocity,
        left_controller_pose=left_controller,
        right_controller_pose=right_controller,
        left_hand_pose=left_hand,
        right_hand_pose=right_hand,
        left_hand_active=bool(sample.get("left_hand_active", False)),
        right_hand_active=bool(sample.get("right_hand_active", False)),
        local_timestamp_ns=int(sample.get("local_timestamp_ns", time.time_ns())),
    )


class PicoStreamReset(RuntimeError):
    """The current episode must not span an SDK stream reset."""


class XRoboToolkitPicoClient:
    """Thin lazy wrapper around xrobotoolkit_sdk.

    The import is intentionally delayed so tests and metadata tools can run
    without a Pico SDK installation.
    """

    def __init__(self):
        self._xrt = None
        self._last_tick = None
        self._last_update = None
        self._missing_since = None
        self._seen_body = False

    def _reset_watchdog(self):
        self._last_tick = None
        self._last_update = None
        self._missing_since = None
        self._seen_body = False

    def open(self) -> None:
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as exc:
            raise RuntimeError(
                "Pico SDK Python binding xrobotoolkit_sdk is not installed. "
                "Install the XRoboToolkit native Python binding in the collection "
                "environment before using real PICO hardware."
            ) from exc

        self._xrt = xrt
        self._reset_watchdog()
        with _suppress_native_output():
            self._xrt.init()
            time.sleep(0.3)

    def close(self) -> None:
        if self._xrt is not None:
            with _suppress_native_output():
                self._xrt.close()
                time.sleep(0.1)
            self._xrt = None
        self._reset_watchdog()

    def read_sample(self) -> PicoSample:
        sample = self._read_sample_once()
        now = time.monotonic()
        tick = int(np.max(sample.body_joint_timestamp_ns)) or sample.body_timestamp_ns
        regressed = self._last_tick is not None and 0 < tick < self._last_tick
        if sample.body_available and tick > 0:
            self._seen_body = True
            self._missing_since = None
            if tick != self._last_tick:
                self._last_tick, self._last_update = tick, now
            stale = self._last_update is not None and now - self._last_update >= 2.0
        elif self._seen_body:
            if self._missing_since is None:
                self._missing_since = now
            stale = now - self._missing_since >= 2.0
        else:
            stale = False
        if stale or regressed:
            try:
                with _suppress_native_output():
                    self._xrt.close()
                    time.sleep(0.2)
                    self._xrt.init()
                    time.sleep(0.5)
            except Exception as exc:
                raise RuntimeError("PICO stream reset failed; restart collection.") from exc
            finally:
                self._reset_watchdog()
            raise PicoStreamReset("PICO stream reset after tracking stopped; discard this episode.")
        return sample

    def _read_sample_once(self) -> PicoSample:
        if self._xrt is None:
            raise RuntimeError("Pico client is not open")
        xrt = self._xrt

        body_available = bool(_safe_call(xrt.is_body_data_available, False))
        raw = {
            "body_pose": _safe_call(xrt.get_body_joints_pose, None),
            "body_available": body_available,
            "body_timestamp_ns": int(
                _safe_call(
                    getattr(xrt, "get_body_timestamp_ns", None),
                    _safe_call(getattr(xrt, "get_time_stamp_ns", None), 0),
                )
            ),
            "body_joint_timestamp_ns": _safe_call(
                getattr(xrt, "get_body_joints_timestamp", None), None
            ),
            "body_velocity": _safe_call(getattr(xrt, "get_body_joints_velocity", None), None),
            "left_controller_pose": _safe_call(xrt.get_left_controller_pose, None),
            "right_controller_pose": _safe_call(xrt.get_right_controller_pose, None),
            "left_hand_pose": _safe_call(xrt.get_left_hand_tracking_state, None),
            "right_hand_pose": _safe_call(xrt.get_right_hand_tracking_state, None),
            "left_hand_active": bool(_safe_call(xrt.get_left_hand_is_active, False)),
            "right_hand_active": bool(_safe_call(xrt.get_right_hand_is_active, False)),
            "local_timestamp_ns": time.time_ns(),
        }
        return coerce_pico_sample(raw)


def _safe_call(fn, default):
    if fn is None:
        return default
    try:
        with _suppress_native_output():
            return fn()
    except Exception:
        return default


@contextmanager
def _suppress_native_output():
    stdout_fd = stderr_fd = devnull_fd = None
    saved_stdout_fd = saved_stderr_fd = None
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = sys.__stdout__.fileno()
        stderr_fd = sys.__stderr__.fileno()
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        if saved_stdout_fd is not None and stdout_fd is not None:
            os.dup2(saved_stdout_fd, stdout_fd)
        if saved_stderr_fd is not None and stderr_fd is not None:
            os.dup2(saved_stderr_fd, stderr_fd)
        for fd in (saved_stdout_fd, saved_stderr_fd, devnull_fd):
            if fd is not None:
                os.close(fd)


class MockPicoClient:
    """Deterministic Pico source for smoke tests and offline dataset checks."""

    def __init__(self):
        self.frame = 0
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def read_sample(self) -> PicoSample:
        if not self.opened:
            raise RuntimeError("mock Pico client is not open")
        body = _identity_pose(BODY_POSE_SHAPE)
        body[0, 0] = float(self.frame) * 0.01
        body_velocity = np.zeros(BODY_VELOCITY_SHAPE, dtype=np.float32)
        body_joint_timestamp_ns = np.full(
            BODY_JOINT_TIMESTAMP_SHAPE,
            time.monotonic_ns(),
            dtype=np.int64,
        )
        hand = _identity_pose(HAND_POSE_SHAPE)
        sample = coerce_pico_sample(
            {
                "body_pose": body,
                "body_available": True,
                "body_timestamp_ns": self.frame,
                "body_joint_timestamp_ns": body_joint_timestamp_ns,
                "body_velocity": body_velocity,
                "left_hand_pose": hand,
                "right_hand_pose": hand,
                "left_hand_active": True,
                "right_hand_active": True,
                "local_timestamp_ns": time.time_ns(),
            }
        )
        self.frame += 1
        return sample
