from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty
from typing import Callable

from human.devices.gopro import (
    DEFAULT_GOPRO_WIFI_SSID,
    DisabledGoProRecorder,
    GoProConfig,
    create_gopro_recorder,
)
from human.storage.lerobot import GoProVideoInfo, create_dataset, save_samples_as_episode
from human.devices.pico import MockPicoClient, PicoSample, PicoStreamReset, XRoboToolkitPicoClient


STATUS_WAITING = "⏳ 等待录制"
STATUS_RECORDING = "🔴 正在录制"
STATUS_SAVED = "✅ 已保存"
STATUS_DISCARDED = "🗑️ 已丢弃"
ACTION_SAVE = "save"
ACTION_DISCARD = "discard"
GOPRO_CONFIG_DIR = Path(
    os.environ.get("GOPRO_CONFIG_DIR", Path(__file__).resolve().parents[1])
).expanduser()


@dataclass(frozen=True)
class ConfigField:
    flag: str
    attr: str
    parser: Callable[[str], object]
    description: str
    is_bool: bool = False
    display: Callable[[object], object] | None = None


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError("expected yes/no")


def _parse_optional_float(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", "-"}:
        return None
    return float(value)


def _parse_enable_gopro(value: str) -> bool:
    return not _parse_bool(value)


def _default_gopro_file(filename: str, *, search_dirs: list[Path] | None = None) -> str:
    dirs = search_dirs if search_dirs is not None else [GOPRO_CONFIG_DIR, Path.cwd()]
    for directory in dirs:
        candidate = directory / filename
        if candidate.exists():
            return str(candidate)
    return str(dirs[0] / filename) if dirs else filename


CONFIG_FIELDS = [
    ConfigField("--data-dir", "data_dir", str, "保存根目录"),
    ConfigField("--task-name", "task_name", str, "任务目录名"),
    ConfigField("--repo-id", "repo_id", str, "LeRobot repo_id 元数据"),
    ConfigField("--fps", "fps", int, "数据集 FPS"),
    ConfigField("--interval", "interval", float, "Pico 轮询间隔，单位秒"),
    ConfigField("--task", "task", str, "写入 LeRobot 的 task 标签"),
    ConfigField("--duration-sec", "duration_sec", _parse_optional_float, "每条 episode 秒数"),
    ConfigField("--episodes", "episodes", int, "采集条数；0 表示持续采集"),
    ConfigField("--mock-pico", "mock_pico", _parse_bool, "使用模拟 Pico 数据", True),
    ConfigField("--gopro-only", "gopro_only", _parse_bool, "只录 GoPro，跳过 Pico 和 parquet", True),
    ConfigField(
        "--body-timeout-sec",
        "body_timeout_sec",
        float,
        "等待 Pico body tracking 的秒数",
    ),
    ConfigField(
        "--allow-missing-body",
        "allow_missing_body",
        _parse_bool,
        "body tracking 不可用时仍录制",
        True,
    ),
    ConfigField(
        "--enable-gopro",
        "no_gopro",
        _parse_enable_gopro,
        "启用 GoPro 录制",
        True,
        display=lambda value: not bool(value),
    ),
    ConfigField("--gopro-width", "gopro_width", int, "GoPro 宽度"),
    ConfigField("--gopro-height", "gopro_height", int, "GoPro 高度"),
    ConfigField("--gopro-fps", "gopro_fps", _parse_optional_float, "GoPro 录制 FPS"),
    ConfigField("--gopro-cohn-creds", "gopro_cohn_creds", str, "COHN 凭证 JSON"),
    ConfigField("--gopro-cohn-cert", "gopro_cohn_cert", str, "COHN 证书"),
    ConfigField("--gopro-finalize-sec", "gopro_finalize_sec", float, "停录后等待 MP4 finalize 秒数"),
    ConfigField("--gopro-download-timeout", "gopro_download_timeout", float, "GoPro WiFi 下载超时秒数"),
    ConfigField(
        "--gopro-wifi-ssid",
        "gopro_wifi_ssid",
        str,
        "目标局域网 SSID；未指定时自动检测主机 WiFi",
    ),
    ConfigField(
        "--gopro-wifi-password",
        "gopro_wifi_password",
        str,
        "目标局域网密码；空则自动检测或使用 GoPro 已保存配置",
        display=lambda value: "(set)" if value else "auto",
    ),
    ConfigField("--gopro-cohn-ready-timeout", "gopro_cohn_ready_timeout", float, "等待 COHN 就绪秒数"),
    ConfigField("--gopro-cohn-poll-sec", "gopro_cohn_poll_sec", float, "COHN 状态轮询间隔秒数"),
    ConfigField("--gopro-ble-address", "gopro_ble_address", str, "GoPro BLE MAC；空则按名称扫描"),
    ConfigField("--gopro-ble-name-prefix", "gopro_ble_name_prefix", str, "GoPro BLE 名称前缀"),
    ConfigField("--gopro-ble-timeout", "gopro_ble_timeout", float, "GoPro BLE 连接超时秒数"),
    ConfigField("--gopro-ble-retries", "gopro_ble_retries", int, "GoPro BLE 连接重试次数"),
]


@contextmanager
def _cbreak_stdin(enabled: bool):
    if not enabled or not sys.stdin.isatty():
        yield
        return
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def _recording_action_pressed() -> str | None:
    if not sys.stdin.isatty():
        return None
    if select.select([sys.stdin], [], [], 0.0)[0]:
        char = sys.stdin.read(1)
        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.read(1)
        normalized = char.lower()
        if normalized == "s":
            return ACTION_SAVE
        if normalized == "d":
            return ACTION_DISCARD
    return None


def collect_episode(
    pico_client,
    *,
    duration_sec: float | None,
    interval_sec: float,
    episode_index: int,
    display: "StatusDisplay",
    action_reader: Callable[[], str | None] | None = None,
) -> tuple[list[PicoSample], str]:
    samples: list[PicoSample] = []
    started = time.monotonic()
    next_display = started
    action = ACTION_SAVE
    display.update(episode_index, 0, STATUS_RECORDING, force=True)
    while True:
        now = time.monotonic()
        if duration_sec is not None and now - started >= duration_sec:
            action = ACTION_SAVE
            break
        action = (action_reader or _recording_action_pressed)() if duration_sec is None else None
        if action is not None:
            break

        samples.append(pico_client.read_sample())
        if now >= next_display:
            display.update(episode_index, len(samples), STATUS_RECORDING)
            next_display = now + 0.1
        time.sleep(interval_sec)
    display.update(episode_index, len(samples), STATUS_RECORDING, force=True)
    return samples, action


def wait_for_body_tracking(pico_client, *, timeout_sec: float, poll_sec: float = 0.2) -> bool:
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while True:
        try:
            if pico_client.read_sample().body_available:
                return True
        except PicoStreamReset as exc:
            print(str(exc), flush=True)
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.0, float(poll_sec)))


class StatusDisplay:
    def __init__(self):
        self._last_line = ""

    def update(
        self,
        episode_index: int,
        frame_count: int,
        state: str,
        *,
        force: bool = False,
    ) -> None:
        line = f"📍 当前index: {episode_index} | 🎞️ 当前帧数: {frame_count} | 状态: {state}"
        if force or line != self._last_line:
            print(f"\r\033[K{line}", end="", flush=True)

        self._last_line = line

    def newline(self) -> None:
        if self._last_line:
            print(flush=True)


def prompt_for_config(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> argparse.Namespace | None:
    while True:
        _clear_screen()
        print("📋 采集参数")
        print("▶️  按 Enter 使用当前值开始；输入编号修改；q 退出。")
        print()
        for index, field in enumerate(CONFIG_FIELDS, start=1):
            current = _format_field_value(field, getattr(args, field.attr))
            default = _format_field_value(field, parser.get_default(field.attr))
            print(
                f"{index:2d}. {field.flag:<20} current: {current:<18} "
                f"default: {default:<18} {field.description}"
            )
        print()

        choice = input("选择: ").strip().lower()
        if choice == "":
            return args
        if choice in {"q", "quit", "exit"}:
            return None
        if not choice.isdigit() or not 1 <= int(choice) <= len(CONFIG_FIELDS):
            _wait_for_menu_error("无效编号")
            continue

        field = CONFIG_FIELDS[int(choice) - 1]
        if not _edit_config_field(args, field):
            _wait_for_menu_error("输入格式无效")


def _edit_config_field(args: argparse.Namespace, field: ConfigField) -> bool:
    current_text = _format_field_value(field, getattr(args, field.attr))
    if field.is_bool:
        raw = input(f"{field.flag} 当前 {current_text}; 输入 yes/no，空回车不改: ").strip()
    else:
        raw = input(
            f"{field.flag} 当前 {current_text}; 输入新值，空回车不改"
            "，none 清空可选数值: "
        ).strip()
    if raw == "":
        return True
    try:
        setattr(args, field.attr, field.parser(raw))
    except ValueError:
        return False
    return True


def _format_value(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _format_field_value(field: ConfigField, value) -> str:
    if field.display is not None:
        value = field.display(value)
    return _format_value(value)


def _clear_screen() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def _wait_for_menu_error(message: str) -> None:
    print(f"⚠️  {message}")
    input("按 Enter 继续")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Pico full-body human pose data directly into LeRobot format."
    )
    parser.add_argument("--no-web-console", action="store_true", help="Use terminal recording controls")
    parser.add_argument("--web-port", type=int, default=8080, help="Local recording UI port")
    parser.add_argument("--data-dir", default="data", help="Output root directory")
    parser.add_argument(
        "--task-name",
        "--name",
        dest="task_name",
        default="task_test",
        help="Task dataset directory name. --name is kept as a compatibility alias.",
    )
    parser.add_argument("--repo-id", default="local/eh2-human", help="LeRobot repo_id metadata")
    parser.add_argument("--fps", type=int, default=50, help="Dataset fps metadata")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.02,
        help="Pico polling interval in seconds (default: 0.02 = 50 Hz)",
    )
    parser.add_argument("--task", default="task_test", help="Task label stored in LeRobot")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Auto-save each episode after this many seconds. Omit for interactive s/d control.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to collect. 0 keeps collecting until Ctrl+C.",
    )
    parser.add_argument("--mock-pico", action="store_true", help="Use deterministic fake Pico data")
    parser.add_argument(
        "--gopro-only",
        action="store_true",
        help="Only record GoPro video; skip Pico SDK and LeRobot parquet writing",
    )
    parser.add_argument(
        "--body-timeout-sec",
        type=float,
        default=10.0,
        help="Seconds to wait for Pico body tracking before recording",
    )
    parser.add_argument(
        "--allow-missing-body",
        action="store_true",
        help="Record even if Pico body tracking is unavailable",
    )
    parser.add_argument(
        "--no-gopro",
        dest="no_gopro",
        action="store_true",
        default=False,
        help="Disable GoPro video recording for offline tests or diagnostics",
    )
    parser.add_argument(
        "--enable-gopro",
        dest="no_gopro",
        action="store_false",
        help="Enable GoPro recording. This is the default unless --mock-pico or --no-gopro is set.",
    )
    parser.add_argument("--gopro-width", type=int, default=1920, help="Recorded GoPro width")
    parser.add_argument("--gopro-height", type=int, default=1080, help="Recorded GoPro height")
    parser.add_argument("--gopro-codec", choices=("h264", "hevc"), default="h264", help="Recorded GoPro codec")
    parser.add_argument(
        "--gopro-fps",
        type=float,
        default=29.97,
        help="Recorded GoPro fps (default: 29.97 NTSC).",
    )
    parser.add_argument(
        "--gopro-cohn-creds",
        default=_default_gopro_file("cohn_creds.json"),
        help="COHN credentials JSON with GoPro IP and password",
    )
    parser.add_argument(
        "--gopro-cohn-cert",
        default=_default_gopro_file("cohn.crt"),
        help="COHN certificate file",
    )
    parser.add_argument(
        "--gopro-finalize-sec",
        type=float,
        default=2.5,
        help="Seconds to wait after stopping board recording before listing/downloading MP4",
    )
    parser.add_argument(
        "--gopro-download-timeout",
        type=float,
        default=180.0,
        help="GoPro WiFi MP4 download timeout in seconds",
    )
    parser.add_argument(
        "--gopro-wifi-ssid",
        default=DEFAULT_GOPRO_WIFI_SSID,
        help=(
            "Host LAN SSID for GoPro to join; detect the host WiFi if omitted."
        ),
    )
    parser.add_argument(
        "--gopro-wifi-password",
        default=None,
        help="Host LAN WiFi password. If omitted, try nmcli or use an already-configured GoPro AP.",
    )
    parser.add_argument(
        "--gopro-cohn-ready-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for COHN to become ready after BLE WiFi setup",
    )
    parser.add_argument(
        "--gopro-cohn-poll-sec",
        type=float,
        default=1.0,
        help="COHN status poll interval during BLE setup",
    )
    parser.add_argument(
        "--gopro-ble-address",
        default=None,
        help="GoPro BLE MAC address. If omitted, scan or use paired devices by name prefix.",
    )
    parser.add_argument(
        "--gopro-ble-name-prefix",
        default="GoPro",
        help="GoPro BLE device name prefix used when --gopro-ble-address is omitted",
    )
    parser.add_argument(
        "--gopro-ble-timeout",
        type=float,
        default=15.0,
        help="GoPro BLE connection timeout in seconds",
    )
    parser.add_argument(
        "--gopro-ble-retries",
        type=int,
        default=4,
        help="GoPro BLE connection retry count",
    )
    return parser


def run_collection(args: argparse.Namespace, *, control=None) -> Path:
    dataset_root = Path(args.data_dir) / args.task_name
    gopro_config = _gopro_config_from_args(args)
    if args.gopro_only:
        return run_gopro_only_collection(
            args, dataset_root=dataset_root, gopro_config=gopro_config, control=control,
        )

    gopro_enabled = not args.no_gopro and not args.mock_pico
    gopro_video = (
        GoProVideoInfo(
            width=gopro_config.width,
            height=gopro_config.height,
            fps=gopro_config.fps,
            codec=gopro_config.codec,
        )
        if gopro_enabled
        else None
    )
    dataset = create_dataset(
        root=dataset_root,
        repo_id=args.repo_id,
        fps=args.fps,
        gopro_video=gopro_video,
        append=True,
    )
    pico_client = MockPicoClient() if args.mock_pico else XRoboToolkitPicoClient()
    gopro = (
        create_gopro_recorder(root=dataset_root, config=gopro_config)
        if gopro_enabled
        else DisabledGoProRecorder()
    )
    display = control or StatusDisplay()

    try:
        pico_client.open()
        episode_count = 0
        while args.episodes <= 0 or episode_count < args.episodes:
            episode_index = dataset.next_episode_index
            display.update(episode_index, 0, STATUS_WAITING, force=True)
            if args.duration_sec is None:
                (control.wait_for_start if control else _wait_for_recording_start)()
            if not args.mock_pico and not wait_for_body_tracking(
                pico_client, timeout_sec=args.body_timeout_sec,
            ) and not args.allow_missing_body:
                raise RuntimeError("PICO body tracking is unavailable. Enable full-body tracking.")

            gopro.start_episode(episode_index)
            action = ACTION_DISCARD
            samples = []
            try:
                with _cbreak_stdin(args.duration_sec is None and control is None):
                    samples, requested_action = collect_episode(
                        pico_client,
                        duration_sec=args.duration_sec,
                        interval_sec=args.interval,
                        episode_index=episode_index,
                        display=display,
                        **({"action_reader": control.read_action} if control else {}),
                    )
                if requested_action == ACTION_SAVE:
                    # Commit motion before WiFi transfer so a failed download is recoverable.
                    save_samples_as_episode(dataset, samples, task=args.task)
                action = requested_action
            except PicoStreamReset as exc:
                print(f"\n{exc}", flush=True)
            finally:
                gopro.stop_episode(discard=action == ACTION_DISCARD)
            if action == ACTION_DISCARD:
                _discard_episode_artifacts(gopro, episode_index)
                display.update(episode_index, len(samples), STATUS_DISCARDED, force=True)
                display.newline()
                continue
            display.update(episode_index, len(samples), STATUS_SAVED, force=True)
            display.newline()
            episode_count += 1
    finally:
        try:
            gopro.close()
        finally:
            pico_client.close()
    return dataset_root


def _gopro_config_from_args(args: argparse.Namespace) -> GoProConfig:
    return GoProConfig(
        fps=float(args.gopro_fps if args.gopro_fps is not None else args.fps),
        width=int(args.gopro_width),
        height=int(args.gopro_height),
        codec=args.gopro_codec,
        cohn_creds=args.gopro_cohn_creds,
        cohn_cert=args.gopro_cohn_cert,
        finalize_sec=float(args.gopro_finalize_sec),
        download_timeout=float(args.gopro_download_timeout),
        wifi_ssid=args.gopro_wifi_ssid,
        wifi_password=args.gopro_wifi_password,
        cohn_ready_timeout=float(args.gopro_cohn_ready_timeout),
        cohn_poll_sec=float(args.gopro_cohn_poll_sec),
        ble_address=args.gopro_ble_address,
        ble_name_prefix=args.gopro_ble_name_prefix,
        ble_timeout=float(args.gopro_ble_timeout),
        ble_retries=int(args.gopro_ble_retries),
    )


def run_gopro_only_collection(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    gopro_config: GoProConfig,
    control=None,
) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    gopro = create_gopro_recorder(root=dataset_root, config=gopro_config)
    display = control or StatusDisplay()

    try:
        episode_count = 0
        next_episode_index = _next_gopro_episode_index(dataset_root)
        while args.episodes <= 0 or episode_count < args.episodes:
            episode_index = next_episode_index
            display.update(episode_index, 0, STATUS_WAITING, force=True)
            if args.duration_sec is None:
                (control.wait_for_start if control else _wait_for_recording_start)()

            gopro.start_episode(episode_index)
            action = ACTION_DISCARD
            try:
                with _cbreak_stdin(args.duration_sec is None and control is None):
                    action = wait_for_gopro_only_episode(
                        duration_sec=args.duration_sec,
                        episode_index=episode_index,
                        display=display,
                        **({"action_reader": control.read_action} if control else {}),
                    )
            finally:
                gopro.stop_episode(discard=action == ACTION_DISCARD)

            if action == ACTION_DISCARD:
                _discard_episode_artifacts(gopro, episode_index)
                display.update(episode_index, 0, STATUS_DISCARDED, force=True)
                display.newline()
                continue

            display.update(episode_index, 0, STATUS_SAVED, force=True)
            display.newline()
            episode_count += 1
            next_episode_index = episode_index + 1
    finally:
        gopro.close()
    return dataset_root


def wait_for_gopro_only_episode(
    *,
    duration_sec: float | None,
    episode_index: int,
    display: StatusDisplay,
    action_reader: Callable[[], str | None] | None = None,
) -> str:
    display.update(episode_index, 0, STATUS_RECORDING, force=True)
    if duration_sec is not None:
        deadline = time.monotonic() + max(0.0, float(duration_sec))
        while time.monotonic() < deadline:
            display.update(episode_index, 0, STATUS_RECORDING)
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        display.update(episode_index, 0, STATUS_RECORDING, force=True)
        return ACTION_SAVE

    while True:
        action = (action_reader or _recording_action_pressed)()
        if action is not None:
            display.update(episode_index, 0, STATUS_RECORDING, force=True)
            return action
        display.update(episode_index, 0, STATUS_RECORDING)
        time.sleep(0.05)


def _next_gopro_episode_index(root: Path) -> int:
    max_index = -1
    for base in (root / "videos", root / "data"):
        for path in base.glob("chunk-*/**/episode_*.*"):
            try:
                max_index = max(max_index, int(path.stem.removeprefix("episode_")))
            except ValueError:
                continue
    return max_index + 1


def _wait_for_recording_start() -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("interactive recording requires a TTY or --duration-sec")
    print()
    input("▶️  按 Enter 开始；录制中按 s 保存，按 d 丢弃；Ctrl+C 退出")


def _discard_episode_artifacts(gopro, episode_index: int) -> None:
    if not getattr(gopro, "enabled", False):
        return
    candidates = [gopro.video_path(episode_index)]
    root = getattr(gopro, "root", None)
    if root is not None:
        sync_dir = Path(root) / "sync"
        candidates.extend(
            [
                sync_dir / f"episode_{episode_index:06d}.gopro_capture.json",
                sync_dir / f"episode_{episode_index:06d}.gopro_alignment.json",
            ]
        )
    for path in candidates:
        path = Path(path)
        if path.exists():
            path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv_list)
    if args.no_web_console and len(argv_list) == 1 and sys.stdin.isatty():
        prompted_args = prompt_for_config(parser, args)
        if prompted_args is None:
            return 130
        args = prompted_args
    try:
        if not args.no_web_console and args.duration_sec is None:
            from human.ui.server import run_web_console

            run_web_console(args)
        else:
            run_collection(args)
    except KeyboardInterrupt:
        print()
        return 130
    except (FileExistsError, RuntimeError, ValueError, TimeoutError, asyncio.TimeoutError) as exc:
        print(f"\n❌ 错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
