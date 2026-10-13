"""One owner per camera; native frames for pairing and optional preview broker."""

import json
import os
import time
from http.client import HTTPConnection
from urllib.parse import urlparse

import cv2
import numpy as np


class CameraSource:
    kind = "camera"
    allow_resize = False

    def __init__(self, index, size, expected_name=None):
        if os.name == "nt" and expected_name:
            from pygrabber.dshow_graph import FilterGraph

            names = FilterGraph().get_input_devices()
            if not 0 <= index < len(names) or names[index] != expected_name:
                raise ValueError(f"相机编号对应的设备与标定 {expected_name} 不同；可用相机：{names}")
        self.camera = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not self.camera.isOpened():
            self.camera.release()
            raise RuntimeError("相机无法打开；请先关闭旧采集程序对该相机的占用")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.focus_disabled = bool(self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0))
        self.size = tuple(size)
        try:
            for _ in range(10):
                image, _ = self.read()
            if tuple(image.shape[1::-1]) != self.size:
                raise ValueError(f"相机实际分辨率 {image.shape[1::-1]} 与标定 {self.size} 不同")
        except Exception:
            self.close()
            raise

    def read(self):
        started = time.monotonic()
        ok, image = self.camera.read()
        if not ok or image is None:
            raise RuntimeError("相机未返回有效图像")
        return image, started

    def close(self):
        self.camera.release()


class BrokerSource:
    """Old task-1 resized JPEG preview, explicitly unsuitable for robot motion."""

    kind = "broker"
    allow_resize = True

    def __init__(self, url, size):
        parsed = urlparse(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"localhost", "127.0.0.1"}
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username
        ):
            raise ValueError("Preview broker must be a local HTTP service")
        self.url, self.size = url.rstrip("/"), tuple(size)
        self.port = parsed.port or 80

    def _get(self, path):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=1)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(
                    f"Local preview broker returned HTTP {response.status}; redirects are not followed"
                )
            body = response.read(5 * 1024 * 1024 + 1)
            if len(body) > 5 * 1024 * 1024:
                raise ValueError("Preview response is too large")
            return body
        finally:
            connection.close()

    def read(self):
        started = time.monotonic()
        status = json.loads(self._get("/status"))
        if tuple(status.get("image_size", [])) != self.size:
            raise ValueError("Broker camera metadata differs from intrinsic resolution")
        image = cv2.imdecode(np.frombuffer(self._get("/frame.jpg"), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Broker did not provide a JPEG image")
        return image, started

    def close(self):
        pass
