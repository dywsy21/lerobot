import time
from pathlib import Path

import cv2

from .calibration import find_board
from .image_io import write_image


def capture_images(index, output, width=1280, height=720, count=1, interval_s=3.0, board=None):
    """Explicit local camera acquisition; no network service and no robot commands.

    With --board, accept only frames with all interior corners detected. The
    operator must change the board pose between accepted frames.
    """
    output = Path(output)
    if count < 1 or interval_s < 0:
        raise ValueError("count must be positive and interval nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Camera {index} could not be opened")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Attempt fixed focus; device support varies. Confirm it before real calibration.
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        for _ in range(15):
            camera.read()
        deadline = time.monotonic() + max(30, count * (interval_s + 15))
        next_capture = 0.0
        captured = []
        while len(captured) < count and time.monotonic() < deadline:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            if time.monotonic() < next_capture:
                continue
            if board is not None and find_board(frame, board) is None:
                time.sleep(0.05)
                continue
            stamp = time.time_ns()
            path = output / f"capture_{stamp}.png"
            write_image(path, frame)
            captured.append(str(path))
            print(f"Saved {len(captured)}/{count}: {path} ({frame.shape[1]}x{frame.shape[0]})", flush=True)
            next_capture = time.monotonic() + interval_s
        if len(captured) != count:
            raise RuntimeError(f"Captured only {len(captured)}/{count}; check board visibility")
        return captured
    finally:
        camera.release()
