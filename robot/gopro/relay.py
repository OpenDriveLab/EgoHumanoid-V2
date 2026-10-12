#!/usr/bin/env python3
"""Forward a wired GoPro MPEG-TS stream from the robot to a host.

Wired Open GoPro cameras expose an HTTP server over a USB CDC-NCM network
interface. The camera sends its preview/webcam stream as UDP datagrams to the
HTTP client. This process runs on the robot, starts that stream, and forwards
each UDP datagram unchanged to the inference host.

The module intentionally uses only the Python standard library so it runs on
the robot's stock Python 3.8 environment.
"""

import argparse
import ipaddress
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener


GOPRO_MDNS_SERVICE = "_gopro-web._tcp"
DEFAULT_CONTROL_PORT = 8080
DEFAULT_STREAM_PORT = 8554

# Default 1080p/30 fps profile using public Open GoPro settings.
VIDEO_1080P30_SETTINGS = (
    (175, 1),  # Pro control mode
    (232, 1),  # 16:9 video framing on newer cameras
    (2, 9),  # 1080p recording resolution
    (234, 8),  # 30 fps on newer cameras
    (121, 4),  # Linear lens
    (135, 100),  # HyperSmooth Standard
    (150, 0),  # Horizon leveling off
    (134, 0),  # NTSC / 29.97 fps timing
    (173, 0),  # Maximum video performance
    (182, 0),  # Standard bitrate
    (183, 0),  # 8-bit
    (184, 0),  # Standard, non-HDR profile
)


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def camera_ip_from_serial(serial: str) -> str:
    """Return the official wired-GoPro address for a serial-number suffix."""
    match = re.search(r"(\d{3})$", str(serial).strip())
    if match is None:
        raise ValueError("GoPro serial must end with at least three digits")
    first, second, third = match.group(1)
    return "172.2{}.1{}{}.51".format(first, second, third)


def parse_avahi_output(output: str) -> List[str]:
    """Parse wired IPv4 ``avahi-browse -p`` GoPro service records.

    A camera provisioned onto the robot LAN advertises the same service over
    WiFi/COHN. The USB relay must ignore that record and only consider the
    official 172.2x.1xx.51 wired address range.
    """
    addresses = []
    for raw_line in output.splitlines():
        fields = raw_line.strip().split(";")
        if len(fields) < 9:
            continue
        if fields[0] != "=" or fields[2].upper() != "IPV4":
            continue
        if fields[4] != GOPRO_MDNS_SERVICE:
            continue
        try:
            address = str(ipaddress.IPv4Address(fields[7]))
        except ipaddress.AddressValueError:
            continue
        octets = [int(part) for part in address.split(".")]
        if not (
            octets[0] == 172
            and 20 <= octets[1] <= 29
            and 100 <= octets[2] <= 199
            and octets[3] == 51
        ):
            continue
        addresses.append(address)
    return _unique(addresses)


def camera_ips_from_address_output(output: str) -> List[str]:
    """Derive camera ``.51`` candidates from host-side GoPro NCM addresses."""
    candidates = []
    for match in re.finditer(r"\binet\s+(\d+(?:\.\d+){3})(?:/\d+)?", output):
        try:
            host_ip = ipaddress.IPv4Address(match.group(1))
        except ipaddress.AddressValueError:
            continue
        octets = [int(part) for part in str(host_ip).split(".")]
        # Open GoPro USB addressing is 172.2x.1xx.51. Restricting the scan to
        # that shape avoids treating Docker and other RFC1918 links as cameras.
        if octets[0] != 172 or not 20 <= octets[1] <= 29:
            continue
        if not 100 <= octets[2] <= 199:
            continue
        candidates.append("{}.{}.{}.51".format(octets[0], octets[1], octets[2]))
    return _unique(candidates)


def _run_text_command(command: List[str], timeout_s: float) -> str:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            timeout=max(0.1, timeout_s),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout or ""


def discover_camera_candidates(
    camera_ip: str = "",
    serial: str = "",
    discovery_timeout_s: float = 3.0,
) -> List[str]:
    """Return explicit, serial-derived, mDNS, and NCM-derived candidates."""
    candidates = []
    if camera_ip:
        candidates.append(str(ipaddress.IPv4Address(camera_ip.strip())))
    if serial:
        candidates.append(camera_ip_from_serial(serial))

    avahi_output = _run_text_command(
        ["avahi-browse", "-rtp", GOPRO_MDNS_SERVICE], discovery_timeout_s
    )
    candidates.extend(parse_avahi_output(avahi_output))

    address_output = _run_text_command(
        ["ip", "-o", "-4", "address", "show"], discovery_timeout_s
    )
    candidates.extend(camera_ips_from_address_output(address_output))
    return _unique(candidates)


RequestJson = Callable[[str, Optional[Dict[str, object]], bool], Dict[str, object]]


class GoProUsbController:
    """Minimal Open GoPro USB HTTP controller."""

    def __init__(
        self,
        camera_ip: str,
        control_port: int = DEFAULT_CONTROL_PORT,
        timeout_s: float = 2.0,
        request_json: Optional[RequestJson] = None,
    ) -> None:
        self.camera_ip = str(ipaddress.IPv4Address(camera_ip))
        self.control_port = int(control_port)
        self.timeout_s = max(0.1, float(timeout_s))
        self.base_url = "http://{}:{}".format(self.camera_ip, self.control_port)
        self._request_override = request_json
        self.last_profile_result = {"profile": "none", "applied": 0, "errors": []}
        # Wired camera traffic must never be sent through a host HTTP proxy.
        self._opener = build_opener(ProxyHandler({}))

    def request_json(
        self,
        path: str,
        params: Optional[Dict[str, object]] = None,
        required: bool = True,
    ) -> Dict[str, object]:
        if self._request_override is not None:
            return self._request_override(path, params, required)

        url = self.base_url + path
        if params:
            url += "?" + urlencode(params)
        request = Request(url, headers={"User-Agent": "gopro-robot-relay/1"}, method="GET")
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                payload = response.read()
        except (HTTPError, URLError, OSError, socket.timeout) as exc:
            if required:
                raise RuntimeError("GoPro request failed for {}: {}".format(url, exc))
            return {}

        if not payload:
            return {}
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            if required:
                raise RuntimeError("Invalid GoPro JSON from {}: {}".format(url, exc))
            return {}
        return decoded if isinstance(decoded, dict) else {"value": decoded}

    @staticmethod
    def _raise_on_camera_error(response: Dict[str, object], operation: str) -> None:
        error = response.get("error")
        if error not in (None, 0, "0"):
            raise RuntimeError("GoPro {} failed with error {}".format(operation, error))

    def probe(self) -> Dict[str, object]:
        return self.request_json("/gopro/camera/info", required=True)

    def apply_camera_profile(self, profile: str) -> Dict[str, object]:
        if profile == "none":
            self.last_profile_result = {"profile": profile, "applied": 0, "errors": []}
            return self.last_profile_result
        if profile != "1080p30":
            raise ValueError("unsupported GoPro camera profile: {}".format(profile))

        self.request_json("/gopro/camera/presets/set_group", {"id": 1000}, required=False)
        applied = 0
        errors = []
        for setting_id, option_id in VIDEO_1080P30_SETTINGS:
            try:
                response = self.request_json(
                    "/gopro/camera/setting",
                    {"setting": setting_id, "option": option_id},
                    required=True,
                )
                self._raise_on_camera_error(response, "setting {}".format(setting_id))
                applied += 1
            except RuntimeError as exc:
                errors.append(str(exc))
        self.last_profile_result = {
            "profile": profile,
            "applied": applied,
            "errors": errors,
        }
        return self.last_profile_result

    def start_stream(
        self,
        mode: str = "preview",
        port: int = DEFAULT_STREAM_PORT,
        resolution: int = 12,
        fov: int = 4,
        camera_profile: str = "none",
    ) -> None:
        mode = mode.lower()
        if mode not in ("preview", "webcam"):
            raise ValueError("stream mode must be preview or webcam")

        # ViewFinder Preview owns wired camera control. Full Webcam requires
        # external control (p=0) in the official Open GoPro sequence.
        self.request_json(
            "/gopro/camera/control/wired_usb",
            {"p": 0 if mode == "webcam" else 1},
            required=True,
        )
        self.request_json(
            "/gopro/camera/analytics/set_client_info", required=False
        )
        if mode == "webcam" and camera_profile != "none":
            # HERO13 can report an incorrect post-USB webcam state. The
            # documented Start -> Stop handshake places it in real IDLE so
            # settings and the final stream start take effect. Only do this
            # when changing a camera profile: with no setting changes, a
            # forced stop can hang HERO13 even though direct start delivers UDP.
            self.request_json(
                "/gopro/webcam/start",
                {
                    "port": int(port),
                    "res": int(resolution),
                    "fov": int(fov),
                    "protocol": "TS",
                },
                required=True,
            )
            self.request_json("/gopro/webcam/stop", required=True)
        elif mode == "preview":
            self.request_json("/gopro/camera/stream/stop", required=False)
            self.request_json("/gopro/camera/shutter/stop", required=False)
        self.apply_camera_profile(camera_profile)

        if mode == "preview":
            response = self.request_json(
                "/gopro/camera/stream/start", {"port": int(port)}, required=True
            )
            self._raise_on_camera_error(response, "preview start")
            return

        response = self.request_json(
            "/gopro/webcam/start",
            {
                "port": int(port),
                "res": int(resolution),
                "fov": int(fov),
                "protocol": "TS",
            },
            required=True,
        )
        self._raise_on_camera_error(response, "webcam start")

    def stop_stream(self, mode: str) -> None:
        if mode.lower() == "webcam":
            self.request_json("/gopro/webcam/stop", required=False)
            self.request_json("/gopro/webcam/exit", required=False)
        else:
            self.request_json("/gopro/camera/stream/stop", required=False)

    def keep_alive(self) -> None:
        self.request_json("/gopro/camera/keep_alive", required=True)


def discover_camera_ip(
    camera_ip: str = "",
    serial: str = "",
    control_port: int = DEFAULT_CONTROL_PORT,
    discovery_timeout_s: float = 3.0,
    http_timeout_s: float = 2.0,
) -> Tuple[str, Dict[str, object]]:
    """Find and probe the first reachable wired GoPro."""
    candidates = discover_camera_candidates(camera_ip, serial, discovery_timeout_s)
    if not candidates:
        raise RuntimeError("No wired GoPro candidate found")
    errors = []
    for candidate in candidates:
        controller = GoProUsbController(candidate, control_port, http_timeout_s)
        try:
            return candidate, controller.probe()
        except RuntimeError as exc:
            errors.append("{}: {}".format(candidate, exc))
    raise RuntimeError("No reachable wired GoPro: {}".format("; ".join(errors)))


class UdpForwarder:
    """Forward UDP datagrams without decoding or changing their boundaries."""

    def __init__(
        self,
        listen_host: str,
        listen_port: int,
        forward_host: str,
        forward_port: int,
        socket_buffer_bytes: int = 4 * 1024 * 1024,
        forwarding_enabled: bool = True,
    ) -> None:
        self._receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, socket_buffer_bytes)
        self._receiver.bind((listen_host, int(listen_port)))
        self.listen_port = int(self._receiver.getsockname()[1])

        self._sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, socket_buffer_bytes)
        self._target = (socket.gethostbyname(forward_host), int(forward_port))
        self.datagrams_forwarded = 0
        self.bytes_forwarded = 0
        self.last_packet_monotonic = 0.0
        self.last_source = None  # type: Optional[Tuple[str, int]]
        self.forwarding_enabled = forwarding_enabled
        self.datagrams_dropped_while_paused = 0
        self._closed = False

    def set_forwarding_enabled(self, enabled: bool) -> None:
        self.forwarding_enabled = enabled
        if not enabled:
            self.datagrams_dropped_while_paused = 0

    def forward_one(self, timeout_s: Optional[float] = None) -> int:
        self._receiver.settimeout(timeout_s)
        payload, source = self._receiver.recvfrom(65535)
        self.last_packet_monotonic = time.monotonic()
        self.last_source = source
        if not self.forwarding_enabled:
            self.datagrams_dropped_while_paused += 1
            return 0
        sent = self._sender.sendto(payload, self._target)
        if sent != len(payload):
            raise RuntimeError("short UDP send: {} of {} bytes".format(sent, len(payload)))
        self.datagrams_forwarded += 1
        self.bytes_forwarded += sent
        return sent

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._receiver.close()
        self._sender.close()


class GoProUsbRelay:
    """Supervise camera discovery, HTTP control, forwarding, and recovery."""

    def __init__(self, args: argparse.Namespace, packet_receiver=None) -> None:
        self.args = args
        self.stop_event = threading.Event()
        self.controller = None  # type: Optional[GoProUsbController]
        self.restart_count = 0
        self.stream_started_monotonic = 0.0
        self.forwarder = packet_receiver if packet_receiver is not None else UdpForwarder(
            args.listen_host,
            args.stream_port,
            args.forward_host,
            args.forward_port,
            args.socket_buffer_bytes,
            forwarding_enabled=args.stream_mode != "webcam",
        )

    def _log(self, message: str) -> None:
        print("[gopro-usb-relay] {}".format(message), flush=True)

    def _connect_and_start(self) -> None:
        self.forwarder.set_forwarding_enabled(False)
        camera_ip, state = discover_camera_ip(
            camera_ip=self.args.camera_ip,
            serial=self.args.serial,
            control_port=self.args.control_port,
            discovery_timeout_s=self.args.discovery_timeout_s,
            http_timeout_s=self.args.http_timeout_s,
        )
        controller = GoProUsbController(
            camera_ip, self.args.control_port, self.args.http_timeout_s
        )
        controller.start_stream(
            mode=self.args.stream_mode,
            port=self.args.stream_port,
            resolution=self.args.webcam_resolution,
            fov=self.args.webcam_fov,
            camera_profile=self.args.camera_profile,
        )
        self.stream_started_monotonic = time.monotonic()
        self.controller = controller
        self.forwarder.set_forwarding_enabled(True)
        self._log(
            "camera={} mode={} profile={} settings={}/{} state_keys={} UDP :{} -> {}:{}".format(
                camera_ip,
                self.args.stream_mode,
                self.args.camera_profile,
                controller.last_profile_result.get("applied", 0),
                len(VIDEO_1080P30_SETTINGS) if self.args.camera_profile != "none" else 0,
                len(state),
                self.args.stream_port,
                self.args.forward_host,
                self.args.forward_port,
            )
        )
        profile_errors = controller.last_profile_result.get("errors", [])
        if profile_errors:
            self._log(
                "profile warning: {} setting request(s) failed; first={}".format(
                    len(profile_errors), profile_errors[0]
                )
            )

    def _control_loop(self) -> None:
        """Own camera HTTP calls without ever blocking the UDP forwarding loop."""
        next_keep_alive = time.monotonic()
        try:
            while not self.stop_event.is_set():
                if self.controller is None:
                    try:
                        self._connect_and_start()
                        next_keep_alive = time.monotonic() + self.args.keep_alive_interval_s
                    except Exception as exc:  # Keep supervising USB hot-plug/reconnect.
                        self._log("camera unavailable: {}; retrying".format(exc))
                        self.stop_event.wait(self.args.discovery_retry_s)
                        continue

                now = time.monotonic()
                if (
                    self.args.stream_mode == "preview"
                    and self.args.preview_keep_alive
                    and now >= next_keep_alive
                    and self.controller is not None
                ):
                    try:
                        self.controller.keep_alive()
                    except Exception as exc:
                        self._log("keep-alive failed: {}; rediscovering".format(exc))
                        self.controller = None
                    next_keep_alive = now + self.args.keep_alive_interval_s

                if (
                    self.controller is not None
                    and now
                    - max(
                        self.forwarder.last_packet_monotonic,
                        self.stream_started_monotonic,
                    )
                    >= self.args.watchdog_timeout_s
                ):
                    self.restart_count += 1
                    self._log(
                        "no UDP for {:.1f}s; restarting stream ({})".format(
                            now
                            - max(
                                self.forwarder.last_packet_monotonic,
                                self.stream_started_monotonic,
                            ),
                            self.restart_count,
                        )
                    )
                    try:
                        self.forwarder.set_forwarding_enabled(False)
                        self.controller.start_stream(
                            mode=self.args.stream_mode,
                            port=self.args.stream_port,
                            resolution=self.args.webcam_resolution,
                            fov=self.args.webcam_fov,
                            camera_profile=self.args.camera_profile,
                        )
                        self.stream_started_monotonic = now
                        self.forwarder.set_forwarding_enabled(True)
                    except Exception as exc:
                        self._log("stream restart failed: {}; rediscovering".format(exc))
                        self.controller = None
                self.stop_event.wait(0.05)
        finally:
            if self.controller is not None:
                self.controller.stop_stream(self.args.stream_mode)

    def run(self) -> None:
        control_thread = threading.Thread(
            target=self._control_loop,
            name="gopro-control",
            daemon=True,
        )
        control_thread.start()
        next_stats = time.monotonic() + self.args.stats_interval_s
        try:
            while not self.stop_event.is_set():
                try:
                    self.forwarder.forward_one(timeout_s=0.1)
                except socket.timeout:
                    pass
                except OSError as exc:
                    if self.stop_event.is_set():
                        break
                    self._log("UDP receive error: {}".format(exc))

                now = time.monotonic()
                if self.args.verbose and now >= next_stats:
                    age = (
                        now - self.forwarder.last_packet_monotonic
                        if self.forwarder.last_packet_monotonic
                        else float("inf")
                    )
                    self._log(
                        "datagrams={} bytes={} startup_drops={} last_packet_age={:.3f}s restarts={}".format(
                            self.forwarder.datagrams_forwarded,
                            self.forwarder.bytes_forwarded,
                            getattr(self.forwarder, "datagrams_dropped_while_paused", 0),
                            age,
                            self.restart_count,
                        )
                    )
                    next_stats = now + self.args.stats_interval_s
        finally:
            self.stop_event.set()
            control_thread.join(
                timeout=max(
                    1.0,
                    getattr(self.args, "discovery_timeout_s", 0.0)
                    + getattr(self.args, "http_timeout_s", 0.0)
                    + 0.5,
                )
            )
            self.forwarder.close()
            self._log("stopped")

    def stop(self) -> None:
        self.stop_event.set()


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-ip", default=os.environ.get("GOPRO_CAMERA_IP", ""))
    parser.add_argument("--serial", default=os.environ.get("GOPRO_SERIAL", ""))
    parser.add_argument(
        "--control-port",
        type=int,
        default=_env_int("GOPRO_CONTROL_PORT", DEFAULT_CONTROL_PORT),
    )
    parser.add_argument(
        "--stream-mode",
        choices=("preview", "webcam"),
        default=os.environ.get("GOPRO_STREAM_MODE", "webcam"),
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=_env_int("GOPRO_STREAM_PORT", DEFAULT_STREAM_PORT),
    )
    parser.add_argument(
        "--webcam-resolution",
        type=int,
        choices=(4, 7, 12),
        default=_env_int("GOPRO_WEBCAM_RESOLUTION", 12),
    )
    parser.add_argument(
        "--webcam-fov",
        type=int,
        choices=(0, 2, 3, 4),
        default=_env_int("GOPRO_WEBCAM_FOV", 4),
    )
    parser.add_argument(
        "--camera-profile",
        choices=("none", "1080p30"),
        default=os.environ.get("GOPRO_CAMERA_PROFILE", "1080p30"),
    )
    parser.add_argument("--listen-host", default=os.environ.get("GOPRO_LISTEN_HOST", "0.0.0.0"))
    parser.add_argument("--forward-host", default=os.environ.get("GOPRO_FORWARD_HOST", ""))
    parser.add_argument(
        "--forward-port",
        type=int,
        default=_env_int("GOPRO_FORWARD_PORT", DEFAULT_STREAM_PORT),
    )
    parser.add_argument(
        "--http-timeout-s",
        type=float,
        default=_env_float("GOPRO_HTTP_TIMEOUT_S", 2.0),
    )
    parser.add_argument(
        "--discovery-timeout-s",
        type=float,
        default=_env_float("GOPRO_DISCOVERY_TIMEOUT_S", 3.0),
    )
    parser.add_argument(
        "--discovery-retry-s",
        type=float,
        default=_env_float("GOPRO_DISCOVERY_RETRY_S", 2.0),
    )
    parser.add_argument(
        "--keep-alive-interval-s",
        type=float,
        default=_env_float("GOPRO_KEEP_ALIVE_INTERVAL_S", 3.0),
    )
    parser.add_argument(
        "--preview-keep-alive",
        action="store_true",
        default=os.environ.get("GOPRO_PREVIEW_KEEP_ALIVE", "false").lower()
        in ("1", "true", "yes", "y"),
        help="Send the HTTP keep-alive request during Preview streaming.",
    )
    parser.add_argument(
        "--watchdog-timeout-s",
        type=float,
        default=_env_float("GOPRO_WATCHDOG_TIMEOUT_S", 15.0),
    )
    parser.add_argument(
        "--socket-buffer-bytes",
        type=int,
        default=_env_int("GOPRO_SOCKET_BUFFER_BYTES", 1024 * 1024),
    )
    parser.add_argument(
        "--stats-interval-s",
        type=float,
        default=_env_float("GOPRO_STATS_INTERVAL_S", 10.0),
    )
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.probe_only and not args.forward_host:
        parser.error("--forward-host or GOPRO_FORWARD_HOST is required")
    for name in (
        "http_timeout_s",
        "discovery_timeout_s",
        "discovery_retry_s",
        "keep_alive_interval_s",
        "watchdog_timeout_s",
        "stats_interval_s",
    ):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.probe_only:
        camera_ip, state = discover_camera_ip(
            camera_ip=args.camera_ip,
            serial=args.serial,
            control_port=args.control_port,
            discovery_timeout_s=args.discovery_timeout_s,
            http_timeout_s=args.http_timeout_s,
        )
        print(
            json.dumps(
                {"camera_ip": camera_ip, "state": state},
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 0

    relay = GoProUsbRelay(args)

    def handle_signal(_signum: int, _frame: object) -> None:
        relay.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    relay.run()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (RuntimeError, ValueError) as exc:
        print("[gopro-usb-relay] ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)
