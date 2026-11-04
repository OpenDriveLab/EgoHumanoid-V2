"""Local browser controls for the shared PICO/GoPro collection loop."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import threading

from human.collect import (
    STATUS_DISCARDED, STATUS_RECORDING, STATUS_SAVED, STATUS_WAITING, run_collection,
)


CONFIG_FIELDS = (
    "data_dir", "task_name", "task", "gopro_wifi_ssid", "gopro_ble_address",
)


class CollectionControl:
    def __init__(self, args):
        self.args = argparse.Namespace(**vars(args))
        self._condition = threading.Condition()
        self._thread = None
        self._closing = False
        self._action = None
        self._state = dict(phase="disconnected", episode=0, frames=0, result="", error="")

    def snapshot(self):
        with self._condition:
            return {
                **self._state,
                "config": {key: getattr(self.args, key) or "" for key in CONFIG_FIELDS},
                "mock": self.args.mock_pico,
                "gopro_only": self.args.gopro_only,
                "gopro": self.args.gopro_only or not (self.args.no_gopro or self.args.mock_pico),
            }

    def connect(self, config):
        if not isinstance(config, dict) or set(config) - {*CONFIG_FIELDS, "gopro_wifi_password"}:
            raise ValueError("Unknown configuration field")
        if any(not isinstance(value, str) or len(value) > 1024 or "\0" in value
               for value in config.values()):
            raise ValueError("Configuration values must be short text")
        with self._condition:
            if self._thread and self._thread.is_alive():
                raise ValueError("Finish the current session before changing configuration")
            args = argparse.Namespace(**vars(self.args))
            for key, value in config.items():
                if key == "gopro_wifi_password" and not value:
                    continue
                setattr(args, key, value)
            if not args.task_name.strip() or args.task_name in (".", "..") or any(
                char in args.task_name for char in "/\\"
            ):
                raise ValueError("Task folder must be a single folder name")
            if not args.data_dir.strip() or not args.task.strip():
                raise ValueError("Output folder and task description are required")
            args.data_dir = str(Path(args.data_dir).expanduser())
            args.duration_sec = None
            self.args = args
            self._closing, self._action = False, None
            self._state.update(phase="connecting", frames=0, result="", error="")
            self._thread = threading.Thread(target=self._collect, name="eh2-collector")
            self._thread.start()

    def _collect(self):
        try:
            run_collection(self.args, control=self)
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            # Device exceptions can contain credentials or authenticated URLs.
            with self._condition:
                self._state.update(
                    phase="error",
                    error=f"Collection stopped ({type(exc).__name__}). Check devices and output folder, then prepare again. Saved motion is retained if video transfer failed.",
                )
        finally:
            with self._condition:
                if self._state["phase"] != "error":
                    self._state["phase"] = "disconnected"
                self._condition.notify_all()

    def command(self, action):
        with self._condition:
            phase = self._state["phase"]
            if action == "disconnect":
                self._closing = True
                if self._thread and self._thread.is_alive():
                    self._state["phase"] = "stopping"
                else:
                    self._state["phase"] = "disconnected"
            elif action == "start" and phase == "ready":
                self._action = action
                self._state.update(phase="starting", result="")
            elif action in ("save", "discard") and phase == "recording":
                self._action = action
                self._state["phase"] = "saving" if action == "save" else "discarding"
            else:
                raise ValueError("Action unavailable in the current recording state")
            self._condition.notify_all()

    def wait_for_start(self):
        with self._condition:
            self._condition.wait_for(lambda: self._closing or self._action == "start")
            if self._closing:
                raise KeyboardInterrupt
            self._action = None

    def read_action(self):
        with self._condition:
            if self._closing:
                raise KeyboardInterrupt
            action, self._action = self._action, None
            return action

    def update(self, episode_index, frame_count, state, *, force=False):
        with self._condition:
            self._state.update(episode=episode_index, frames=frame_count)
            if self._closing:
                return
            if state == STATUS_WAITING:
                self._state["phase"] = "ready"
            elif state == STATUS_RECORDING and self._state["phase"] not in ("saving", "discarding"):
                self._state["phase"] = "recording"
            elif state in (STATUS_SAVED, STATUS_DISCARDED):
                self._state["result"] = "Saved" if state == STATUS_SAVED else "Discarded"

    def newline(self):
        pass

    def close(self):
        self.command("disconnect")
        if self._thread:
            # Finish a pending transfer or discard before releasing devices.
            self._thread.join()


def create_server(control, *, port=8080):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *args):
            pass

        def reply(self, code, payload, *, html=False):
            data = payload.encode() if html else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", f"default-src 'none'; script-src 'nonce-{self.server.token}'; style-src 'nonce-{self.server.token}'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self, *, token=False):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            valid = host in allowed and self.headers.get("Origin", f"http://{host}") == f"http://{host}"
            if token:
                valid = valid and secrets.compare_digest(self.headers.get("X-EH2-Token", ""), self.server.token)
            if not valid:
                self.reply(403, {"error": "Open the recording UI on this computer"})
            return valid

        def do_GET(self):
            if not self.authorized(token=self.path == "/api/state"):
                return
            if self.path == "/":
                page = Path(__file__).with_name("index.html").read_text()
                self.reply(200, page.replace("__TOKEN__", self.server.token), html=True)
            elif self.path == "/api/state":
                self.reply(200, control.snapshot())
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if not self.authorized(token=True):
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected a small JSON request")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/connect":
                    control.connect(body)
                elif self.path in ("/api/start", "/api/save", "/api/discard", "/api/disconnect"):
                    control.command(self.path.rsplit("/", 1)[1])
                else:
                    self.reply(404, {"error": "Not found"})
                    return
            except (ValueError, UnicodeError):
                self.reply(400, {"error": "Invalid configuration or action unavailable. Check the fields and recording state."})
                return
            self.reply(200, control.snapshot())

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.token = secrets.token_hex(24)
    return server


def run_web_console(args):
    control = CollectionControl(args)
    with create_server(control, port=args.web_port) as server:
        print(f"Open http://127.0.0.1:{server.server_port} — Ctrl+C to exit.", flush=True)
        try:
            server.serve_forever()
        finally:
            control.close()
