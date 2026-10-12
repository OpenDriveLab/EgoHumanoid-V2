from __future__ import annotations

import base64
from dataclasses import dataclass
import http.client
import json
from json import JSONDecodeError
import math
from pathlib import Path
import ssl
import subprocess
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request


GOPRO_VIDEO_KEY = "observation.images.gopro"
GOPRO_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/observation.images.gopro/"
    "episode_{episode_index:06d}.mp4"
)
DEFAULT_GOPRO_WIFI_SSID: str | None = None


@dataclass(frozen=True)
class GoProConfig:
    fps: float = 29.97
    width: int = 1920
    height: int = 1080
    codec: str = "h264"
    cohn_creds: str | Path = "cohn_creds.json"
    cohn_cert: str | Path = "cohn.crt"
    finalize_sec: float = 2.5
    download_timeout: float = 180.0
    wifi_ssid: str | None = DEFAULT_GOPRO_WIFI_SSID
    wifi_password: str | None = None
    cohn_ready_timeout: float = 60.0
    cohn_poll_sec: float = 1.0
    ble_address: str | None = None
    ble_name_prefix: str = "GoPro"
    ble_timeout: float = 15.0
    ble_retries: int = 4


@dataclass(frozen=True)
class GoProMediaFile:
    directory: str
    name: str
    size: int = 0
    modified: int = 0

    @property
    def identity(self) -> tuple[str, str]:
        return self.directory, self.name

    @property
    def camera_path(self) -> str:
        return f"{self.directory}/{self.name}"


class GoProHttpError(RuntimeError):
    def __init__(
        self,
        *,
        url: str,
        status_code: int,
        reason: str,
        body: str = "",
    ):
        self.url = url
        self.status_code = int(status_code)
        self.reason = reason
        self.body = body
        detail = f"GoPro COHN returned HTTP {self.status_code} {self.reason} for {self.url}"
        if body:
            detail += f": {body[:500]}"
        super().__init__(detail)


class GoProConnectionError(RuntimeError):
    pass


class DisabledGoProRecorder:
    enabled = False

    def start_episode(self, episode_index: int) -> None:
        return None

    def stop_episode(self, *, discard: bool = False) -> None:
        return None

    def close(self) -> None:
        return None


class CohnClient:
    """GoPro COHN HTTPS control and media download client."""

    def __init__(
        self,
        *,
        creds_path: str | Path,
        cert_path: str | Path,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ):
        self.creds_path = Path(creds_path).expanduser()
        self.cert_path = Path(cert_path).expanduser()
        self._opener = opener
        creds = _read_cohn_credentials(self.creds_path)
        self.ip = str(creds["ip"])
        password = str(creds["password"])
        cert = _read_cohn_cert(self.cert_path)
        self._ssl_context = ssl.create_default_context(cadata=cert)
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE
        token = base64.b64encode(f"gopro:{password}".encode("utf-8")).decode("ascii")
        self._auth_header = f"Basic {token}"

    def get_bytes(self, path: str, *, timeout: float = 20.0) -> bytes:
        url = f"https://{self.ip}{path}"
        request = urllib.request.Request(url)
        request.add_header("Authorization", self._auth_header)
        try:
            with self._opener(request, context=self._ssl_context, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                body = ""
            raise GoProHttpError(
                url=url,
                status_code=exc.code,
                reason=str(exc.reason),
                body=body,
            ) from exc
        except http.client.RemoteDisconnected as exc:
            raise GoProConnectionError(
                f"GoPro COHN closed the connection without a response for {url}: {exc}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise GoProConnectionError(
                f"Cannot reach GoPro COHN at {url}: {reason}. "
                "Check that the GoPro is powered on, awake, on the same WiFi, "
                "and that cohn_creds.json matches this camera. "
                "If the IP changed, re-run BLE setup or update --gopro-cohn-creds."
            ) from exc

    def get_json(self, path: str, *, timeout: float = 20.0) -> dict:
        return json.loads(self.get_bytes(path, timeout=timeout))

    def status(self) -> dict:
        return self.get_json("/gopro/camera/state").get("status", {})

    def ensure_video_ready(self) -> None:
        status = self.status()
        if status.get("10") == 1:
            self.stop_shutter()
            time.sleep(1.0)
        try:
            self.get_bytes("/gopro/camera/presets/set_group?id=1000")
        except GoProHttpError as exc:
            if exc.status_code not in {400, 403, 404}:
                raise
            print(
                "⚠️  GoPro 不接受自动切到视频 preset，继续使用当前模式。"
                "请确认 GoPro 已在视频模式，否则保存时可能找不到 MP4。"
                f"({exc.status_code} {exc.reason})",
                flush=True,
            )
        except GoProConnectionError as exc:
            print(
                "⚠️  GoPro 自动切到视频 preset 时断开连接，继续使用当前模式。"
                "请确认 GoPro 已在视频模式，否则保存时可能找不到 MP4。"
                f"({exc})",
                flush=True,
            )
        time.sleep(1.2)

    def start_shutter(self) -> bytes:
        return self.get_bytes("/gopro/camera/shutter/start")

    def stop_shutter(self) -> bytes:
        return self.get_bytes("/gopro/camera/shutter/stop")

    def media_files(self) -> list[GoProMediaFile]:
        media_list = self.get_json("/gopro/media/list")
        files: list[GoProMediaFile] = []
        for directory in media_list.get("media", []):
            directory_name = str(directory.get("d", ""))
            for item in directory.get("fs", []):
                files.append(
                    GoProMediaFile(
                        directory=directory_name,
                        name=str(item.get("n", "")),
                        size=int(item.get("s", 0) or 0),
                        modified=int(item.get("mod", 0) or 0),
                    )
                )
        return sorted(files, key=lambda item: (item.modified, item.directory, item.name))

    def media_info(self, media: GoProMediaFile) -> dict:
        quoted = urllib.parse.quote(media.camera_path)
        return self.get_json(f"/gopro/media/info?path={quoted}")

    def last_captured_media(self) -> GoProMediaFile:
        media = self.get_json("/gopro/media/last_captured")
        directory = str(media.get("folder") or media.get("d") or "")
        name = str(media.get("file") or media.get("n") or "")
        if not directory or not name:
            raise RuntimeError("GoPro last_captured response has no media path")
        return GoProMediaFile(directory=directory, name=name)

    def download_media(
        self,
        media: GoProMediaFile,
        output_path: Path,
        *,
        timeout: float | None = None,
    ) -> None:
        if output_path.exists():
            raise FileExistsError(f"Video already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        directory = urllib.parse.quote(media.directory, safe="")
        name = urllib.parse.quote(media.name, safe="")
        request = urllib.request.Request(f"https://{self.ip}/videos/DCIM/{directory}/{name}")
        request.add_header("Authorization", self._auth_header)
        temporary = None
        try:
            with self._opener(
                request, context=self._ssl_context,
                timeout=float(timeout if timeout is not None else 180.0),
            ) as response, tempfile.NamedTemporaryFile(
                dir=output_path.parent, prefix=output_path.name + ".", suffix=".part", delete=False,
            ) as output:
                temporary = Path(output.name)
                downloaded = 0
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                expected = int(response.headers.get("Content-Length", 0)) or media.size
                if downloaded == 0 or (expected > 0 and downloaded != expected):
                    raise GoProConnectionError(
                        f"Incomplete GoPro video: received {downloaded}, expected {expected} bytes"
                    )
            temporary.replace(output_path)
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError) as exc:
            raise GoProConnectionError(f"GoPro video download failed: {media.camera_path}") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _read_cohn_credentials(path: Path) -> dict:
    try:
        creds = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"GoPro COHN credentials file not found: {path}. "
            "Set --gopro-cohn-creds, put cohn_creds.json in the current directory, "
            "or run with --no-gopro if you want to skip camera recording."
        ) from exc
    except JSONDecodeError as exc:
        raise RuntimeError(f"GoPro COHN credentials JSON is invalid: {path}") from exc
    try:
        creds["ip"]
        creds["password"]
    except KeyError as exc:
        raise RuntimeError(f"GoPro COHN credentials must contain ip and password: {path}") from exc
    return creds


def _read_cohn_cert(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"GoPro COHN certificate file not found: {path}. "
            "Set --gopro-cohn-cert or put cohn.crt in the current directory."
        ) from exc


class BoardGoProRecorder:
    """Trigger GoPro board recording and pull the resulting MP4 over COHN WiFi."""

    enabled = True

    def __init__(
        self,
        *,
        root: Path,
        config: GoProConfig,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        shutter: Any | None = None,
        setup: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.root = Path(root)
        self.config = config
        self.client = client
        self._client_factory = client_factory or (
            lambda: CohnClient(
                creds_path=config.cohn_creds,
                cert_path=config.cohn_cert,
            )
        )
        self.shutter = shutter
        self.setup = setup
        self._sleep = sleep
        self._episode_index: int | None = None
        self._before_media: set[tuple[str, str]] = set()
        self._capture_timing: dict[str, Any] = {}

    def start_episode(self, episode_index: int) -> None:
        if self._episode_index is not None:
            raise RuntimeError("GoPro recorder is already running")
        if self.video_path(episode_index).exists():
            raise FileExistsError(f"Video already exists: {self.video_path(episode_index)}")

        self._prepare_client()
        assert self.client is not None
        if self.shutter is None:
            self.shutter = self.client
        self.client.ensure_video_ready()
        self._before_media = {media.identity for media in self.client.media_files()}
        previous = self._last_captured()
        if previous is not None:
            self._before_media.add(previous.identity)
        start_request_host_ns = time.time_ns()
        result = _call_shutter(self.shutter, "start")
        start_ack_host_ns = time.time_ns()
        if result is False:
            raise RuntimeError("GoPro did not acknowledge start recording")
        self._episode_index = int(episode_index)
        self._capture_timing = {
            "episode_index": int(episode_index),
            "start_request_host_ns": start_request_host_ns,
            "start_ack_host_ns": start_ack_host_ns,
        }

    def stop_episode(self, *, discard: bool = False) -> None:
        if self._episode_index is None:
            return

        episode_index = self._episode_index
        try:
            assert self.client is not None
            stop_request_host_ns = time.time_ns()
            result = _call_shutter(self.shutter, "stop")
            stop_ack_host_ns = time.time_ns()
            if result is False:
                raise RuntimeError("GoPro did not acknowledge stop recording")
            if discard:
                return
            self._sleep(max(0.0, float(self.config.finalize_sec)))
            media = self._newest_new_mp4()
            video_path = self.video_path(episode_index)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            self._capture_timing.update(
                {
                    "stop_request_host_ns": stop_request_host_ns,
                    "stop_ack_host_ns": stop_ack_host_ns,
                    "video_path": str(video_path),
                    "media": {
                        "directory": media.directory,
                        "name": media.name,
                        "size": media.size,
                        "modified": media.modified,
                    },
                }
            )
            self._write_capture_timing(episode_index)
            for attempt in range(3):
                try:
                    self.client.download_media(
                        media, video_path, timeout=float(self.config.download_timeout),
                    )
                    break
                except GoProConnectionError:
                    if attempt == 2:
                        raise
                    self._sleep(1.0)
            try:
                video = _probe_video(video_path)
            except RuntimeError:
                video_path.unlink(missing_ok=True)
                raise
            self._capture_timing["download_done_host_ns"] = time.time_ns()
            self._capture_timing["video"] = video
            self._write_capture_timing(episode_index)
            if video["codec"] != self.config.codec or (video["width"], video["height"]) != (self.config.width, self.config.height) or not math.isclose(
                video["fps"], self.config.fps, rel_tol=0, abs_tol=0.02,
            ):
                raise RuntimeError(
                    f"GoPro video settings differ from collection metadata: {video}. "
                    f"Original retained at {video_path}; match camera and --gopro-* settings."
                )
        finally:
            self._episode_index = None
            self._before_media = set()
            self._capture_timing = {}

    def close(self) -> None:
        closed_ids: set[int] = set()
        for target in (self.shutter, self.setup):
            if target is None or id(target) in closed_ids:
                continue
            closed_ids.add(id(target))
            close = getattr(target, "close", None)
            if close is not None:
                close()

    def video_path(self, episode_index: int) -> Path:
        return _episode_video_path(self.root, episode_index)

    def _prepare_client(self) -> None:
        if self.setup is not None:
            self.setup.prepare()
        if self.client is None:
            self.client = self._client_factory()

    def _newest_new_mp4(self) -> GoProMediaFile:
        for attempt in range(5):
            latest = self._last_captured()
            if latest is not None and latest.identity not in self._before_media:
                return latest
            after = self.client.media_files()
            new_media = [
                media for media in after
                if media.identity not in self._before_media and media.name.upper().endswith(".MP4")
            ]
            if new_media:
                return max(new_media, key=lambda media: (media.modified, media.directory, media.name))
            if attempt < 4:
                self._sleep(0.5)
        raise RuntimeError("GoPro recording stopped but no new MP4 was found; confirm Video mode.")

    def _last_captured(self) -> GoProMediaFile | None:
        try:
            media = self.client.last_captured_media()
        except (AttributeError, RuntimeError, ValueError):
            return None
        return media if media.name.upper().endswith(".MP4") else None

    def _write_capture_timing(self, episode_index: int) -> None:
        path = self.root / "sync" / f"episode_{episode_index:06d}.gopro_capture.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._capture_timing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def create_gopro_recorder(*, root: Path, config: GoProConfig):
    from human.devices.bluetooth import BleGoProSetup

    setup = BleGoProSetup(
        address=config.ble_address,
        name_prefix=config.ble_name_prefix,
        timeout=config.ble_timeout,
        retries=config.ble_retries,
        wifi_ssid=config.wifi_ssid,
        wifi_password=config.wifi_password,
        cohn_creds=config.cohn_creds,
        cohn_cert=config.cohn_cert,
        cohn_ready_timeout=config.cohn_ready_timeout,
        cohn_poll_sec=config.cohn_poll_sec,
    )
    return BoardGoProRecorder(
        root=root,
        config=config,
        client_factory=lambda: CohnClient(
            creds_path=config.cohn_creds,
            cert_path=config.cohn_cert,
        ),
        setup=setup,
    )


def _episode_video_path(root: Path, episode_index: int) -> Path:
    return Path(root) / GOPRO_VIDEO_PATH.format(
        episode_chunk=episode_index // 1000,
        episode_index=episode_index,
    )


def _probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,codec_name,avg_frame_rate,duration:format=duration", "-of", "json", str(path),
    ], capture_output=True, text=True, timeout=30)
    try:
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        numerator, denominator = stream["avg_frame_rate"].split("/")
        fps = float(numerator) / float(denominator)
        duration = float(stream.get("duration") or info["format"]["duration"])
        width, height = int(stream["width"]), int(stream["height"])
        if result.returncode or not all(math.isfinite(v) and v > 0 for v in (fps, duration, width, height)):
            raise ValueError("invalid stream")
        return {"width": width, "height": height, "fps": fps, "duration": duration, "codec": stream["codec_name"]}
    except (ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
        raise RuntimeError(f"Invalid GoPro video: {path}") from exc


def _call_shutter(shutter: Any, action: str):
    if action == "start":
        method = getattr(shutter, "start_shutter", None) or getattr(shutter, "start_record")
    else:
        method = getattr(shutter, "stop_shutter", None) or getattr(shutter, "stop_record")
    return method()
