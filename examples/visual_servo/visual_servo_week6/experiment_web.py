"""Local HTTP console for the shared camera/calibration/servo station."""

import contextlib
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .experiment_runtime import ExperimentStation


def serve(
    config,
    source="simulation",
    port=8767,
    camera_index=2,
    broker_url="http://127.0.0.1:8766",
    allow_hardware=False,
):
    station = ExperimentStation(config, source, camera_index, broker_url, allow_hardware)
    token = secrets.token_urlsafe(32)
    page = Path(__file__).with_name("experiment_console.html").read_text(encoding="utf-8")
    page = page.replace("__STATION_TOKEN__", token).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data, mime="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.reply(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self.reply(200, station.status())
            elif self.path.startswith("/frame.jpg"):
                with station.lock:
                    frame = station.jpeg
                self.reply(200 if frame else 503, frame or b"Frame not ready", "image/jpeg")
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/api/control":
                self.reply(404, {"error": "Not found"})
                return
            allowed_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
            if self.headers.get("Origin") not in allowed_origins or not secrets.compare_digest(
                self.headers.get("X-Station-Token", ""), token
            ):
                self.reply(403, {"error": "Only the local station page may control this service"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("Expected a small JSON request")
                payload = json.loads(self.rfile.read(length))
                result = station.control(payload.pop("command"), payload)
                self.reply(200, result)
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
                self.reply(400, {"error": str(error)})

        def log_message(self, *_args):
            pass

    try:
        with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
            print(f"实验控制台：http://127.0.0.1:{port}/ | source={source}", flush=True)
            server.serve_forever()
    finally:
        station.close()
