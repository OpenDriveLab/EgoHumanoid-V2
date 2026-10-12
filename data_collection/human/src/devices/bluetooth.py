from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Callable


CMD_REQ_UUID = "b5f90072-aa8d-11e3-9046-0002a5d5c51b"
CMD_RESP_UUID = "b5f90073-aa8d-11e3-9046-0002a5d5c51b"
SETTINGS_UUID = "b5f90074-aa8d-11e3-9046-0002a5d5c51b"
QUERY_REQ_UUID = "b5f90076-aa8d-11e3-9046-0002a5d5c51b"
QUERY_RESP_UUID = "b5f90077-aa8d-11e3-9046-0002a5d5c51b"
NETWORK_REQ_UUID = "b5f90091-aa8d-11e3-9046-0002a5d5c51b"
NETWORK_RESP_UUID = "b5f90092-aa8d-11e3-9046-0002a5d5c51b"

SET_SHUTTER_ON = bytes([0x03, 0x01, 0x01, 0x01])
SET_SHUTTER_OFF = bytes([0x03, 0x01, 0x01, 0x00])
KEEPALIVE = bytes([0x03, 0x5B, 0x01, 0x42])
PROVISION_SUCCESS = {5, 6}
CONFIGURED_AP_FLAG = 2


@dataclass(frozen=True)
class HostWifi:
    ssid: str
    password: str | None = None


def detect_host_wifi(
    *,
    ssid: str | None = None,
    password: str | None = None,
    cached_ssid: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> HostWifi:
    resolved_ssid = ssid or _active_wifi_ssid(runner) or cached_ssid
    if not resolved_ssid:
        raise RuntimeError(
            "Cannot determine host WiFi SSID. Pass --gopro-wifi-ssid explicitly."
        )
    resolved_password = password if password is not None else _wifi_password(resolved_ssid, runner)
    return HostWifi(ssid=resolved_ssid, password=resolved_password)


def _active_wifi_ssid(runner: Callable[..., subprocess.CompletedProcess]) -> str | None:
    result = _run_nmcli(["-t", "-f", "ACTIVE,SSID", "dev", "wifi"], runner)
    for raw_line in result.stdout.splitlines():
        active, _, ssid = raw_line.partition(":")
        if active == "yes" and ssid:
            return ssid.replace(r"\:", ":")
    return None


def _wifi_password(
    ssid: str,
    runner: Callable[..., subprocess.CompletedProcess],
) -> str | None:
    candidates = [ssid, *_active_wifi_connection_names(runner)]
    seen: set[str] = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        result = _run_nmcli(
            ["-s", "-g", "802-11-wireless-security.psk", "connection", "show", name],
            runner,
            check=False,
        )
        password = result.stdout.strip()
        if result.returncode == 0 and password:
            return password
    return None


def _active_wifi_connection_names(
    runner: Callable[..., subprocess.CompletedProcess],
) -> list[str]:
    result = _run_nmcli(["-t", "-f", "NAME,TYPE", "connection", "show", "--active"], runner)
    names: list[str] = []
    for raw_line in result.stdout.splitlines():
        name, _, connection_type = raw_line.partition(":")
        if connection_type in {"802-11-wireless", "wifi"}:
            names.append(name.replace(r"\:", ":"))
    return names


def _run_nmcli(
    args: list[str],
    runner: Callable[..., subprocess.CompletedProcess],
    *,
    check: bool = False,
) -> subprocess.CompletedProcess:
    try:
        result = runner(
            ["nmcli", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "nmcli is required to auto-detect host WiFi. Pass --gopro-wifi-ssid "
            "and --gopro-wifi-password explicitly, or install NetworkManager tools."
        ) from exc
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nmcli failed")
    return result


def _fragment(data: bytes):
    max_len = 20
    data_len = len(data)
    if data_len >= (2**13 - 1):
        raise ValueError("Open GoPro BLE payload is too long")
    header = bytes([0x20 | ((data_len >> 8) & 0x1F), data_len & 0xFF])
    remaining = bytearray(data)
    first = True
    while remaining:
        packet = bytearray(header) if first else bytearray([0x80])
        first = False
        room = max_len - len(packet)
        packet += remaining[:room]
        del remaining[:room]
        yield bytes(packet)


class _Accumulator:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.buf = bytearray()
        self.rem = 0
        self.done = False

    def feed(self, data: bytearray | bytes) -> None:
        packet = bytearray(data)
        if not packet:
            return
        if packet[0] & 0x80:
            packet.pop(0)
        else:
            self.buf = bytearray()
            header_type = (packet[0] & 0x60) >> 5
            if header_type == 0:
                self.rem = packet[0] & 0x1F
                packet = packet[1:]
            elif header_type == 1:
                self.rem = ((packet[0] & 0x1F) << 8) + packet[1]
                packet = packet[2:]
            elif header_type == 2:
                self.rem = (packet[1] << 8) + packet[2]
                packet = packet[3:]
            self.done = False
        self.buf.extend(packet)
        self.rem -= len(packet)
        if self.rem <= 0:
            self.done = True


class AsyncGoProBle:
    """Async Open GoPro BLE controller for wake, WiFi provisioning, and keepalive."""

    def __init__(self, *, name_prefix: str = "GoPro"):
        self.name_prefix = name_prefix
        self._client: Any | None = None
        self._ack_event: asyncio.Event | None = None
        self._last_ack: bool | None = None
        self._queue: asyncio.Queue | None = None
        self._parsers: dict[int, Any] = {}
        self._accumulators: dict[str, _Accumulator] = {}
        self._keepalive_stop: asyncio.Event | None = None
        self._keepalive_task: asyncio.Task | None = None

    async def connect(
        self,
        *,
        address: str | None = None,
        timeout: float = 15.0,
        retries: int = 4,
    ) -> str:
        try:
            from bleak import BleakClient, BleakScanner
            from bleak.exc import BleakError
        except ImportError as exc:
            raise RuntimeError(
                "GoPro BLE control requires bleak. Install project dependencies again."
            ) from exc

        target = address
        name = _bluez_device_name(address) if address else self.name_prefix
        if target is None:
            device = await BleakScanner.find_device_by_filter(
                lambda dev, _adv: bool(dev.name) and dev.name.startswith(self.name_prefix),
                timeout=timeout,
            )
            if device is not None:
                target = device.address
                name = device.name or device.address
            else:
                paired = _paired_bluez_device(self.name_prefix)
                if paired is None:
                    raise RuntimeError(
                        f"no paired or advertising GoPro BLE device with name prefix "
                        f"{self.name_prefix!r} was found"
                    )
                target, name = paired

        self._ack_event = asyncio.Event()
        self._queue = asyncio.Queue()
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            client = BleakClient(target, timeout=timeout)
            try:
                print(
                    f"🔵 正在连接 GoPro BLE ({target})，第 {attempt}/{retries} 次...",
                    flush=True,
                )
                await asyncio.wait_for(client.connect(), timeout=timeout)
                self._client = client
                await asyncio.wait_for(
                    client.start_notify(CMD_RESP_UUID, self._on_command_response),
                    timeout=timeout,
                )
                return name or target
            except (BleakError, asyncio.TimeoutError, EOFError) as exc:
                last_error = exc
                try:
                    await client.disconnect()
                except Exception:
                    pass
                if attempt < retries:
                    await asyncio.sleep(2.0)

        raise RuntimeError(
            f"GoPro BLE connection failed after {retries} retries: {last_error}. "
            "Confirm the GoPro is powered on or asleep, not fully off, and pair it again "
            "from the GoPro Bluetooth pairing screen if needed."
        )

    def _on_command_response(self, _characteristic, data: bytearray) -> None:
        self._last_ack = len(data) >= 3 and data[2] == 0x00
        if self._ack_event is not None:
            self._ack_event.set()
        self._feed_proto(CMD_RESP_UUID, data)

    def _on_proto_response(self, uuid: str):
        def handler(_characteristic, data: bytearray) -> None:
            self._feed_proto(uuid, data)

        return handler

    def _feed_proto(self, uuid: str, data: bytearray | bytes) -> None:
        if self._queue is None:
            return
        accumulator = self._accumulators.setdefault(uuid, _Accumulator())
        accumulator.feed(data)
        if not accumulator.done:
            return
        full = bytes(accumulator.buf)
        accumulator.reset()
        if len(full) < 2:
            return
        action_id = full[1]
        message = None
        parser = self._parsers.get(action_id)
        if parser is not None:
            message = parser()
            try:
                message.ParseFromString(full[2:])
            except Exception:
                message = None
        self._queue.put_nowait((action_id, message))

    async def _command(self, payload: bytes, *, timeout: float = 5.0) -> bool:
        if self._client is None or self._ack_event is None:
            raise RuntimeError("GoPro BLE controller is not connected")
        self._ack_event.clear()
        self._last_ack = None
        await self._client.write_gatt_char(CMD_REQ_UUID, payload, response=True)
        try:
            await asyncio.wait_for(self._ack_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return bool(self._last_ack)

    async def start_record(self) -> bool:
        return await self._command(SET_SHUTTER_ON)

    async def stop_record(self) -> bool:
        return await self._command(SET_SHUTTER_OFF)

    async def ensure_wifi_and_cohn(
        self,
        *,
        wifi: HostWifi,
        creds_path: Path,
        cert_path: Path,
        ready_timeout: float = 60.0,
        poll_sec: float = 1.0,
    ) -> None:
        if self._client is None or self._queue is None:
            raise RuntimeError("GoPro BLE controller is not connected")
        try:
            from open_gopro import proto
            from open_gopro.constants import ActionId, FeatureId
        except ImportError as exc:
            raise RuntimeError(
                "GoPro WiFi provisioning requires open-gopro. Install project "
                "dependencies again with `pip install -e .`."
            ) from exc

        self._parsers.update(
            {
                ActionId.RESPONSE_CREATE_COHN_CERT.value: proto.ResponseGeneric,
                ActionId.RESPONSE_GET_COHN_STATUS.value: proto.NotifyCOHNStatus,
                ActionId.RESPONSE_GET_COHN_CERT.value: proto.ResponseCOHNCert,
                ActionId.SCAN_WIFI_NETWORKS.value | 0x80: proto.ResponseStartScanning,
                ActionId.GET_AP_ENTRIES.value | 0x80: proto.ResponseGetApEntries,
                ActionId.REQUEST_WIFI_CONNECT.value | 0x80: proto.ResponseConnect,
                ActionId.REQUEST_WIFI_CONNECT_NEW.value | 0x80: proto.ResponseConnectNew,
                ActionId.NOTIF_START_SCAN.value: proto.NotifStartScanning,
                ActionId.NOTIF_PROVIS_STATE.value: proto.NotifProvisioningState,
            }
        )
        await self._client.start_notify(NETWORK_RESP_UUID, self._on_proto_response(NETWORK_RESP_UUID))
        await self._client.start_notify(QUERY_RESP_UUID, self._on_proto_response(QUERY_RESP_UUID))
        await self.start_keepalive()

        await self._connect_wifi(proto, ActionId, FeatureId, wifi)
        ip, password = await self._provision_cohn(
            proto,
            ActionId,
            FeatureId,
            timeout=float(ready_timeout),
            poll_sec=float(poll_sec),
        )
        await self._write_cohn_files(
            proto,
            ActionId,
            FeatureId,
            cert_path=cert_path,
            creds_path=creds_path,
            ip=ip,
            password=password,
            wifi_ssid=wifi.ssid,
        )

    async def _connect_wifi(self, proto, ActionId, FeatureId, wifi: HostWifi) -> None:
        print("📡 GoPro 正在扫描可见 WiFi...", flush=True)
        await self._send_proto(
            NETWORK_REQ_UUID,
            FeatureId.NETWORK_MANAGEMENT.value,
            ActionId.SCAN_WIFI_NETWORKS.value,
            proto.RequestStartScan(),
        )
        scan = None
        try:
            while True:
                message = await self._wait_for(ActionId.NOTIF_START_SCAN.value, timeout=45.0)
                if getattr(message, "scanning_state", None) == 5:
                    scan = message
                    break
        except asyncio.TimeoutError as exc:
            raise RuntimeError("GoPro WiFi scan timed out over BLE") from exc
        await self._send_proto(
            NETWORK_REQ_UUID,
            FeatureId.NETWORK_MANAGEMENT.value,
            ActionId.GET_AP_ENTRIES.value,
            proto.RequestGetApEntries(
                scan_id=scan.scan_id,
                start_index=0,
                max_entries=getattr(scan, "total_entries", 0) or 100,
            ),
        )
        try:
            response = await self._wait_for(ActionId.GET_AP_ENTRIES.value | 0x80, timeout=30.0)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("GoPro WiFi AP list timed out over BLE") from exc
        target = next((entry for entry in response.entries if entry.ssid == wifi.ssid), None)
        if target is None:
            raise RuntimeError(f"GoPro cannot see host WiFi SSID {wifi.ssid!r}")
        print(f"📶 GoPro 已看到目标局域网: {wifi.ssid}", flush=True)
        if target.scan_entry_flags & CONFIGURED_AP_FLAG:
            request = proto.RequestConnect(ssid=wifi.ssid)
            action = ActionId.REQUEST_WIFI_CONNECT.value
            print(f"🔁 GoPro 使用已保存的 WiFi 配置连接: {wifi.ssid}", flush=True)
        else:
            if not wifi.password:
                raise RuntimeError(
                    f"GoPro has not stored WiFi {wifi.ssid!r}; pass --gopro-wifi-password."
                )
            request = proto.RequestConnectNew(ssid=wifi.ssid, password=wifi.password)
            action = ActionId.REQUEST_WIFI_CONNECT_NEW.value
            print(f"🔐 GoPro 写入 WiFi 密码并连接: {wifi.ssid}", flush=True)
        await self._send_proto(NETWORK_REQ_UUID, FeatureId.NETWORK_MANAGEMENT.value, action, request)
        try:
            while True:
                message = await self._wait_for(ActionId.NOTIF_PROVIS_STATE.value, timeout=12.0)
                state = getattr(message, "provisioning_state", None)
                if state is not None:
                    print(f"📶 GoPro WiFi provisioning state: {state}", flush=True)
                if state in PROVISION_SUCCESS:
                    return
                if state is not None and (state >= 7 or state in {3, 4}):
                    raise RuntimeError(f"GoPro WiFi provisioning failed with state {state}")
        except asyncio.TimeoutError:
            print("⚠️  GoPro 未回传 WiFi 完成通知，继续检查 COHN 状态", flush=True)
            return

    async def _provision_cohn(
        self,
        proto,
        ActionId,
        FeatureId,
        *,
        timeout: float,
        poll_sec: float,
    ) -> tuple[str, str]:
        await self._send_proto(
            CMD_REQ_UUID,
            FeatureId.COMMAND.value,
            ActionId.REQUEST_CREATE_COHN_CERT.value,
            proto.RequestCreateCOHNCert(override=True),
        )
        try:
            await self._wait_for(ActionId.RESPONSE_CREATE_COHN_CERT.value, timeout=30.0)
        except asyncio.TimeoutError:
            pass

        await self._send_proto(
            QUERY_REQ_UUID,
            FeatureId.QUERY.value,
            ActionId.REQUEST_GET_COHN_STATUS.value,
            proto.RequestGetCOHNStatus(register_cohn_status=True),
        )
        deadline = asyncio.get_running_loop().time() + max(1.0, timeout)
        last_message = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                message = await self._wait_for(
                    ActionId.RESPONSE_GET_COHN_STATUS.value,
                    timeout=max(0.1, poll_sec),
                )
                last_message = message
            except asyncio.TimeoutError:
                continue
            if getattr(message, "enabled", False) and getattr(message, "ipaddress", ""):
                return str(message.ipaddress), str(message.password)
        raise RuntimeError(f"GoPro COHN did not become ready: {last_message}")

    async def _write_cohn_files(
        self,
        proto,
        ActionId,
        FeatureId,
        *,
        cert_path: Path,
        creds_path: Path,
        ip: str,
        password: str,
        wifi_ssid: str,
    ) -> None:
        await self._send_proto(
            QUERY_REQ_UUID,
            FeatureId.QUERY.value,
            ActionId.REQUEST_GET_COHN_CERT.value,
            proto.RequestCOHNCert(),
        )
        response = await self._wait_for(ActionId.RESPONSE_GET_COHN_CERT.value, timeout=30.0)
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        creds_path.parent.mkdir(parents=True, exist_ok=True)
        cert_path.write_text(response.cert, encoding="utf-8")
        creds_path.write_text(
            json.dumps(
                {
                    "ip": ip,
                    "user": "gopro",
                    "password": password,
                    "wifi_ssid": wifi_ssid,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        creds_path.chmod(0o600)

    async def _send_proto(self, uuid: str, feature: int, action: int, message: Any) -> None:
        if self._client is None:
            raise RuntimeError("GoPro BLE controller is not connected")
        payload = bytes([feature, action]) + message.SerializeToString()
        for packet in _fragment(payload):
            await self._client.write_gatt_char(uuid, packet, response=True)

    async def _wait_for(self, action_id: int, *, timeout: float) -> Any:
        if self._queue is None:
            raise RuntimeError("GoPro BLE controller is not connected")
        while True:
            received_action, message = await asyncio.wait_for(self._queue.get(), timeout)
            if received_action == action_id:
                return message

    async def start_keepalive(self) -> None:
        if self._client is None:
            raise RuntimeError("GoPro BLE controller is not connected")
        if self._keepalive_task is not None:
            return
        self._keepalive_stop = asyncio.Event()
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self) -> None:
        assert self._keepalive_stop is not None
        while not self._keepalive_stop.is_set():
            try:
                if self._client is not None:
                    await self._client.write_gatt_char(SETTINGS_UUID, KEEPALIVE, response=True)
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._keepalive_stop.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                pass

    async def disconnect(self) -> None:
        if self._keepalive_stop is not None:
            self._keepalive_stop.set()
        if self._keepalive_task is not None:
            await self._keepalive_task
            self._keepalive_task = None
        if self._client is None:
            return
        for uuid in (NETWORK_RESP_UUID, QUERY_RESP_UUID, CMD_RESP_UUID):
            try:
                await self._client.stop_notify(uuid)
            except Exception:
                pass
        await self._client.disconnect()
        self._client = None


class BleGoProSetup:
    """Synchronous BLE setup used by the collector before COHN HTTPS recording."""

    def __init__(
        self,
        *,
        address: str | None,
        name_prefix: str,
        timeout: float,
        retries: int,
        wifi_ssid: str | None,
        wifi_password: str | None,
        cohn_creds: str | Path,
        cohn_cert: str | Path,
        cohn_ready_timeout: float,
        cohn_poll_sec: float,
    ):
        self.address = address
        self.name_prefix = name_prefix
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.wifi_ssid = wifi_ssid
        self.wifi_password = wifi_password
        self.cohn_creds = Path(cohn_creds).expanduser()
        self.cohn_cert = Path(cohn_cert).expanduser()
        self.cohn_ready_timeout = float(cohn_ready_timeout)
        self.cohn_poll_sec = float(cohn_poll_sec)
        self._loop = asyncio.new_event_loop()
        self._controller: AsyncGoProBle | None = None
        self._prepared = False

    def prepare(self) -> None:
        if self._prepared:
            return
        cached_ssid = _cached_wifi_ssid(self.cohn_creds)
        try:
            wifi = detect_host_wifi(
                ssid=self.wifi_ssid,
                password=self.wifi_password,
                cached_ssid=cached_ssid,
            )
        except RuntimeError as wifi_error:
            name = self.open()
            print(f"🔵 GoPro BLE: {name}", flush=True)
            if _existing_cohn_reachable(self.cohn_creds, self.cohn_cert):
                print("📶 未检测到本机 WiFi；复用现有 GoPro COHN 连接", flush=True)
                self._prepared = True
                return
            raise RuntimeError(
                f"{wifi_error} The host appears to be on Ethernet and the existing "
                "GoPro COHN credentials are not reachable; set --gopro-wifi-ssid."
            ) from wifi_error
        print(f"📶 本机局域网: {wifi.ssid}", flush=True)
        name = self.open()
        print(f"🔵 GoPro BLE: {name}", flush=True)
        assert self._controller is not None
        self._loop.run_until_complete(
            self._controller.ensure_wifi_and_cohn(
                wifi=wifi,
                creds_path=self.cohn_creds,
                cert_path=self.cohn_cert,
                ready_timeout=self.cohn_ready_timeout,
                poll_sec=self.cohn_poll_sec,
            )
        )
        print(f"✅ GoPro 已连接局域网并准备好 COHN: {wifi.ssid}", flush=True)
        self._prepared = True

    def open(self) -> str:
        if self._controller is not None:
            return self.name_prefix

        async def connect() -> str:
            controller = AsyncGoProBle(name_prefix=self.name_prefix)
            name = await controller.connect(
                address=self.address,
                timeout=self.timeout,
                retries=self.retries,
            )
            self._controller = controller
            return name

        return self._loop.run_until_complete(connect())

    def close(self) -> None:
        if self._controller is not None:
            self._loop.run_until_complete(self._controller.disconnect())
            self._controller = None
        if not self._loop.is_closed():
            self._loop.close()


def _paired_bluez_device(name_prefix: str) -> tuple[str, str] | None:
    try:
        result = subprocess.run(
            ["bluetoothctl", "devices"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    for raw_line in result.stdout.splitlines():
        parts = raw_line.split(maxsplit=2)
        if len(parts) >= 3 and parts[2].startswith(name_prefix):
            return parts[1], parts[2]
    return None


def _cached_wifi_ssid(creds_path: Path) -> str | None:
    try:
        creds = json.loads(creds_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    ssid = creds.get("wifi_ssid") or creds.get("ssid")
    if isinstance(ssid, str) and ssid.strip():
        return ssid.strip()
    return None


def _existing_cohn_reachable(creds_path: Path, cert_path: Path) -> bool:
    try:
        from human.devices.gopro import CohnClient

        client = CohnClient(creds_path=creds_path, cert_path=cert_path)
        client.get_json("/gopro/camera/state", timeout=5.0)
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return True


def _bluez_device_name(address: str | None) -> str | None:
    if not address:
        return None
    try:
        result = subprocess.run(
            ["bluetoothctl", "info", address],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Alias: "):
            return line.removeprefix("Alias: ").strip() or None
        if line.startswith("Name: "):
            return line.removeprefix("Name: ").strip() or None
    return None
