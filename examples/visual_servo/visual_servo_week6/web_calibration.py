"""Local camera calibration station. No robot commands or cloud uploads."""

import argparse
import contextlib
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .calibration import board_points, calibrate_images, find_board, save_yaml
from .image_io import read_image, write_image


def pose_descriptor(corners, image_size):
    return np.asarray(corners).reshape(-1, 2) / np.asarray(image_size)


def pose_distance(a, b):
    # A chessboard detector may reverse corner ordering by 180 degrees.
    return min(
        float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))),
        float(np.sqrt(np.mean(np.sum((a[::-1] - b) ** 2, axis=1)))),
    )


def is_distinct(descriptor, previous, threshold=0.035):
    return not previous or min(pose_distance(descriptor, p) for p in previous) >= threshold


class CalibrationSession:
    def __init__(self, index, output, width=1280, height=720, board=(9, 6), square=0.025, target=25):
        from pygrabber.dshow_graph import FilterGraph

        devices = FilterGraph().get_input_devices()
        if index < 0 or index >= len(devices):
            raise ValueError(f"Invalid camera index; available cameras: {devices}")
        self.index, self.name = index, devices[index]
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.board, self.square, self.target = tuple(board), square, target
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None
        self.jpeg = None
        self.paths, self.descriptors, self.validation = [], [], []
        self.result = None
        self.automatic = True
        self.stage = "collecting"
        self.message = "请让整个棋盘格进入画面，每次改变姿态后停稳约 2 秒。"
        self.detected = False
        self.stable = False
        self.sharpness = None
        self.frame_number = 0
        self.last_saved_at = 0
        self.camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open {self.name}")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.focus_request_accepted = bool(self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0))
        for _ in range(20):
            ok, frame = self.camera.read()
        if not ok:
            self.camera.release()
            raise RuntimeError("Camera returned no frames")
        self.size = (frame.shape[1], frame.shape[0])
        self.focus_readback = self.camera.get(cv2.CAP_PROP_AUTOFOCUS)
        self.frame = frame
        self.metadata = {
            "data_source": "real_usb_camera",
            "camera_name": self.name,
            "camera_index": index,
            "image_size": list(self.size),
            "board_inner_corners": list(self.board),
            "square_size_m": square,
            "autofocus_disable_request_accepted": self.focus_request_accepted,
            "autofocus_readback": self.focus_readback,
            "created_local": datetime.now().isoformat(timespec="seconds"),
        }
        if (self.output / "intrinsics.yaml").exists():
            self.camera.release()
            raise ValueError("This session already contains calibration results; use a new output directory")
        saved_paths = sorted((self.output / "images").glob("view_*.png"))
        if saved_paths:
            previous_metadata = json.loads((self.output / "session.json").read_text(encoding="utf-8"))
            for field in ("camera_name", "image_size", "board_inner_corners", "square_size_m"):
                if previous_metadata[field] != self.metadata[field]:
                    self.camera.release()
                    raise ValueError(f"Cannot mix calibration sessions with different {field}")
            for path in saved_paths:
                image = read_image(path)
                corners = None if image is None else find_board(image, self.board)
                if corners is None:
                    self.camera.release()
                    raise ValueError(f"Previously captured board could not be recovered: {path}")
                self.paths.append(path)
                self.descriptors.append(pose_descriptor(corners, self.size))
            self.metadata["created_local"] = previous_metadata["created_local"]
            self.metadata["resumed_local"] = datetime.now().isoformat(timespec="seconds")
            print(f"Resumed {len(self.paths)} existing views", flush=True)
        (self.output / "session.json").write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.reader = threading.Thread(target=self.read_camera, daemon=True)
        self.detector = threading.Thread(target=self.detect_and_collect, daemon=True)
        self.reader.start()
        self.detector.start()

    def read_camera(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.camera.read()
                if not ok:
                    with self.lock:
                        self.message = "相机读取失败，请检查 USB 连接。"
                    self.stop.wait(0.3)
                    continue
                if (frame.shape[1], frame.shape[0]) != self.size:
                    with self.lock:
                        self.message = "相机分辨率发生变化，采集已暂停。"
                        self.automatic = False
                    continue
                preview = cv2.resize(frame, (960, round(self.size[1] * 960 / self.size[0])))
                ok, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
                with self.lock:
                    self.frame = frame
                    self.frame_number += 1
                    if ok:
                        self.jpeg = jpeg.tobytes()
                self.stop.wait(0.04)
        finally:
            self.camera.release()

    def detect_and_collect(self):
        previous = None
        stable_streak = 0
        while not self.stop.is_set():
            with self.lock:
                frame = self.frame.copy()
                automatic = self.automatic
                stage = self.stage
            if stage in {"calibrating", "complete", "error"}:
                self.stop.wait(0.5)
                continue
            try:
                # SB handles tilted boards better; all stored coordinates are refined
                # in the original, unresized frame used for calibration.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scale = min(1.0, 960 / frame.shape[1])
                small = cv2.resize(gray, None, fx=scale, fy=scale)
                ok, corners = cv2.findChessboardCornersSB(small, self.board, cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ok:
                    corners = np.float32(corners / scale)
                    corners = cv2.cornerSubPix(
                        gray,
                        corners,
                        (5, 5),
                        (-1, -1),
                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
                    )
                else:
                    corners = None
                descriptor = None if corners is None else pose_descriptor(corners, self.size)
                steady = (
                    descriptor is not None
                    and previous is not None
                    and pose_distance(descriptor, previous) < 0.005
                )
                stable_streak = stable_streak + 1 if steady else 0
                stable = stable_streak >= 3
                previous = descriptor
                sharpness = None
                if corners is not None:
                    x, y, w, h = cv2.boundingRect(corners)
                    sharpness = float(cv2.Laplacian(gray[y : y + h, x : x + w], cv2.CV_64F).var())
                with self.lock:
                    self.detected, self.stable, self.sharpness = corners is not None, stable, sharpness
                if corners is None:
                    message = f"未检测到完整 {self.board[0]}×{self.board[1]} 内角点：请拉远棋盘格，避免裁切、遮挡和反光。"
                elif not stable:
                    message = f"已检测到 {self.board[0] * self.board[1]} 个角点，请停稳约 2 秒。"
                elif sharpness < 30:
                    message = "棋盘格不够清晰，请调整距离或照明，保持焦距不变。"
                elif not is_distinct(descriptor, self.descriptors):
                    message = "这个姿态已经采集过，请改变位置、距离或倾斜角度。"
                elif automatic and time.monotonic() - self.last_saved_at >= 1.5:
                    if stage == "collecting":
                        path = self.output / "images" / f"view_{len(self.paths) + 1:03d}.png"
                        write_image(path, frame)
                        self.paths.append(path)
                        self.descriptors.append(descriptor)
                        self.last_saved_at = time.monotonic()
                        message = f"已保存第 {len(self.paths)} 张，请换一个姿态。"
                        print(message, flush=True)
                        if len(self.paths) >= self.target:
                            self.solve()
                            continue
                    elif stage == "validating":
                        self.validate(frame, corners, descriptor)
                        continue
                else:
                    message = "采集已暂停。" if not automatic else "姿态有效，等待采集。"
                with self.lock:
                    self.message = message
            except Exception as exc:
                with self.lock:
                    self.stage = "error"
                    self.automatic = False
                    self.message = str(exc)
                print(f"Calibration error: {exc}", flush=True)
            self.stop.wait(0.55)

    def solve(self):
        if len(self.paths) < 15:
            raise ValueError("至少需要 15 张有效图像。")
        with self.lock:
            self.stage = "calibrating"
            self.message = "正在计算内参和畸变系数，请稍候。"
        result, _ = calibrate_images(self.paths, self.board, self.square)
        result.update(self.metadata)
        result["validation"] = []
        save_yaml(self.output / "intrinsics.yaml", result)
        (self.output / "intrinsics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        np.savez(
            self.output / "intrinsics.npz",
            camera_matrix=np.asarray(result["K"]),
            distortion=np.asarray(result["distortion"]),
            image_size=np.asarray(self.size),
        )
        with self.lock:
            self.result = result
            self.stage = "validating"
            self.message = "内参已计算。请再摆 3 个新姿态，用于独立验证。"
        print(json.dumps({"stage": "validating", "rms_px": result["rms_px"], "K": result["K"]}), flush=True)

    def validate(self, frame, corners, descriptor):
        camera_matrix = np.asarray(self.result["K"])
        dist = np.asarray(self.result["distortion"])
        objects = board_points(self.board, self.square)
        ok, r, t = cv2.solvePnP(objects, corners, camera_matrix, dist)
        if not ok:
            return
        predicted, _ = cv2.projectPoints(objects, r, t, camera_matrix, dist)
        error = float(
            np.sqrt(np.mean(np.sum((predicted.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1)))
        )
        path = self.output / "validation" / f"view_{len(self.validation) + 1:03d}.png"
        write_image(path, frame)
        entry = {"file": str(path), "rmse_px": error}
        with self.lock:
            self.validation.append(entry)
            self.result["validation"] = list(self.validation)
            self.message = f"验证 {len(self.validation)}/3，当前重投影误差 {error:.3f} px。"
            if len(self.validation) >= 3:
                self.stage = "complete"
                self.automatic = False
                self.message = "采集和标定完成。请查看训练误差、独立验证误差与去畸变效果。"
        self.descriptors.append(descriptor)
        self.last_saved_at = time.monotonic()
        save_yaml(self.output / "intrinsics.yaml", self.result)
        (self.output / "intrinsics.json").write_text(
            json.dumps(self.result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self.stage == "complete":
            self.write_report()

    def write_report(self):
        r = self.result
        validation_errors = ", ".join(format(v["rmse_px"], ".4f") for v in self.validation)
        text = (
            f"相机内参标定（真实 USB 相机图像）\n相机：{self.name}\n分辨率：{self.size[0]} × {self.size[1]}\n"
            f"棋盘格：{self.board[0]} × {self.board[1]} 内角点，方格边长 {self.square * 1000:g} mm\n"
            f"有效标定图像：{r['valid_images']}\n训练重投影 RMS：{r['rms_px']:.4f} px\n"
            f"独立验证 RMS：{validation_errors} px\n"
            f"K（像素单位）：\n{np.asarray(r['K'])}\n畸变系数 k1,k2,p1,p2,k3：\n{r['distortion']}\n"
            "使用时必须保持相同分辨率和焦距。RMS 小于 1 px 是课程质量目标，不代表真实几何误差为零。\n"
            "关闭自动对焦的请求与设备读回值记录在 session.json 中；这些值不能独立证明光学焦距完全固定。\n"
        )
        (self.output / "标定报告.txt").write_text(text, encoding="utf-8-sig")

    def status(self):
        with self.lock:
            return {
                "camera": self.name,
                "camera_index": self.index,
                "image_size": list(self.size),
                "board": list(self.board),
                "square_mm": self.square * 1000,
                "target": self.target,
                "count": len(self.paths),
                "validation_count": len(self.validation),
                "automatic": self.automatic,
                "stage": self.stage,
                "message": self.message,
                "detected": self.detected,
                "stable": self.stable,
                "sharpness": self.sharpness,
                "result": self.result,
                "output": str(self.output),
                "frame_number": self.frame_number,
            }


def serve(index, output, port=8766, board=(9, 6), square=0.025, target=25):
    session = CalibrationSession(index, output, board=board, square=square, target=target)
    root = Path(__file__).parent

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, data, content_type, code=200):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self.send((root / "calibration_console.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/status":
                self.send(
                    json.dumps(session.status(), ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )
            elif path in {"/frame.jpg", "/undistorted.jpg"}:
                with session.lock:
                    jpeg = session.jpeg
                    frame = session.frame.copy()
                    result = session.result
                if path == "/undistorted.jpg" and result:
                    corrected = cv2.undistort(
                        frame, np.asarray(result["K"]), np.asarray(result["distortion"])
                    )
                    corrected = cv2.resize(corrected, (960, round(session.size[1] * 960 / session.size[0])))
                    _, encoded = cv2.imencode(".jpg", corrected)
                    jpeg = encoded.tobytes()
                self.send(jpeg or b"", "image/jpeg", 200 if jpeg else 503)
            elif path == "/intrinsics.yaml" and (session.output / "intrinsics.yaml").exists():
                self.send((session.output / "intrinsics.yaml").read_bytes(), "text/plain; charset=utf-8")
            else:
                self.send(b"Not found", "text/plain", 404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
                return self.send(b"Origin rejected", "text/plain", 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 1024 or self.path != "/action":
                    raise ValueError("Invalid request")
                action = json.loads(self.rfile.read(length))["action"]
                if action == "pause":
                    with session.lock:
                        session.automatic = False
                elif action == "resume":
                    with session.lock:
                        if session.stage in {"complete", "error", "calibrating"}:
                            raise ValueError("当前阶段不能恢复采集")
                        session.automatic = True
                else:
                    raise ValueError("Unknown action")
                self.send(
                    json.dumps(session.status(), ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )
            except Exception as exc:
                self.send(
                    json.dumps({"error": str(exc)}, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                    400,
                )

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(json.dumps({"url": f"http://127.0.0.1:{port}", **session.metadata}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    finally:
        session.stop.set()
        server.server_close()
        session.reader.join(timeout=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--board", nargs=2, type=int, default=[9, 6])
    parser.add_argument("--square-size-m", type=float, default=0.025)
    parser.add_argument("--target", type=int, default=25)
    args = parser.parse_args()
    if args.target < 15 or args.square_size_m <= 0:
        parser.error("Need at least 15 views and a positive square size")
    serve(args.camera, args.output, args.port, tuple(args.board), args.square_size_m, args.target)


if __name__ == "__main__":
    main()
