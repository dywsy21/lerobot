"""Serial IPC isolates native dependencies and grants one process bus ownership."""

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


class RobotProcess:
    def __init__(self, model, repository, log_directory):
        self.model, self.repository = model, Path(repository)
        self.log_directory = Path(log_directory)
        self.process = None
        self.armed = False
        self.responses, self.lock = queue.Queue(), threading.Lock()
        self.stderr = None
        self.failed = False

    def connect(self):
        worker = self.repository / "examples/visual_servo/hardware_worker.py"
        self.stderr = (self.log_directory / "hardware-worker.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-X",
                "utf8",
                "-u",
                str(worker),
                "--model",
                str(self.model.path),
                "--repository",
                str(self.repository),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        def reader():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)

        threading.Thread(target=reader, daemon=True).start()
        try:
            self._call("connect")
        except Exception:
            self.close()
            raise

    def _call(self, command, *arguments):
        with self.lock:
            if self.failed or self.process is None or self.process.poll() is not None:
                raise RuntimeError("串口进程已退出；检查 hardware-worker.log")
            self.process.stdin.write(
                json.dumps({"command": command, "arguments": arguments}, allow_nan=False) + "\n"
            )
            self.process.stdin.flush()
            try:
                line = self.responses.get(timeout=4)
            except queue.Empty as error:
                self.failed = True
                self.process.stdin.close()
                raise TimeoutError("串口进程响应超时；已暂停新的实物指令") from error
            if line is None:
                raise RuntimeError("串口进程关闭；检查 hardware-worker.log")
            data = json.loads(line)
            self.armed = data["armed"]
            if not data["ok"]:
                raise ValueError(data["error"])
            return data.get("result")

    def observe(self):
        return self._call("observe")

    def arm(self):
        self._call("arm")

    def teach(self):
        self._call("teach")

    def move(
        self,
        q,
        observed_at,
        maximum_step_rad,
        maximum_age_s,
        camera_captured_at,
        t_end_from_camera=None,
        max_linear_speed=0.01,
        max_angular_speed=0.1,
    ):
        return self._call(
            "move",
            q,
            observed_at,
            maximum_step_rad,
            maximum_age_s,
            camera_captured_at,
            t_end_from_camera,
            max_linear_speed,
            max_angular_speed,
        )

    def hold(self):
        if self.process is not None and self.armed:
            self._call("hold")

    def close(self):
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    if not self.failed:
                        try:
                            self._call("close")
                        finally:
                            self.process.stdin.close()
                    self.process.wait(timeout=4)
            finally:
                if self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait(timeout=4)
                self.process.stdout.close()
                self.stderr.close()
                self.process = None
                self.armed = False
