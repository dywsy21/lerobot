from pathlib import Path

import cv2
import numpy as np
import yaml

from .geometry import transform
from .image_io import read_image


def board_points(inner_corners=(9, 6), square_size_m=0.025):
    nx, ny = inner_corners
    if nx < 2 or ny < 2 or square_size_m <= 0:
        raise ValueError("Need at least 2x2 interior corners and a positive square size")
    points = np.zeros((nx * ny, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square_size_m
    return points


def find_board(image, inner_corners):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ok, corners = cv2.findChessboardCorners(
        gray, tuple(inner_corners), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ok:
        ok, corners = cv2.findChessboardCornersSB(gray, tuple(inner_corners), cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            return None
    return cv2.cornerSubPix(
        gray, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    )


def calibrate_images(paths, inner_corners=(9, 6), square_size_m=0.025, minimum_images=15):
    objects, pixels, used, rejected = [], [], [], []
    image_size = None
    for path in map(Path, paths):
        image = read_image(path)
        if image is None:
            rejected.append({"file": str(path), "reason": "Unreadable image"})
            continue
        size = image.shape[1], image.shape[0]
        if image_size is not None and size != image_size:
            raise ValueError("All calibration images must have the same resolution")
        corners = find_board(image, inner_corners)
        if corners is None:
            rejected.append({"file": str(path), "reason": "Interior corners not detected"})
            continue
        image_size = size
        objects.append(board_points(inner_corners, square_size_m))
        pixels.append(corners)
        used.append(str(path))
    if len(used) < minimum_images:
        raise ValueError(f"Only {len(used)} valid images; need {minimum_images}")
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        objects, pixels, image_size, None, None
    )
    errors = []
    for obj, observed, rotation, translation, path in zip(
        objects, pixels, rotations, translations, used, strict=True
    ):
        predicted, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
        error = float(
            np.sqrt(np.mean(np.sum((observed.reshape(-1, 2) - predicted.reshape(-1, 2)) ** 2, axis=1)))
        )
        errors.append({"file": path, "rmse_px": error})
    result = {
        "K": matrix.tolist(),
        "distortion": distortion.reshape(-1).tolist(),
        "image_size": list(image_size),
        "rms_px": float(rms),
        "valid_images": len(used),
        "board_inner_corners": list(inner_corners),
        "square_size_m": square_size_m,
        "per_view": errors,
        "rejected": rejected,
        "object_frame_unit": "metre",
        "views": [
            {"file": path, "T_camera_from_board": transform(cv2.Rodrigues(r)[0], t).tolist()}
            for path, r, t in zip(used, rotations, translations, strict=True)
        ],
    }
    return result, list(zip(used, rotations, translations, strict=True))


def save_yaml(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding="utf-8")
