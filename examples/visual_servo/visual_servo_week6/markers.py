import cv2
import numpy as np

from .geometry import transform


def marker_points(side_length_m):
    """Top-left, top-right, bottom-right, bottom-left; marker y points down."""
    if side_length_m <= 0:
        raise ValueError("Marker side must be positive and measured in metres")
    half = side_length_m / 2
    return np.array(
        [[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], dtype=np.float64
    )


def detect_marker(
    image, camera_matrix, distortion, side_length_m=0.06, marker_id=0, dictionary="DICT_4X4_50"
):
    if not dictionary.startswith("DICT_") or not hasattr(cv2.aruco, dictionary):
        raise ValueError("Unknown OpenCV ArUco dictionary")
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)))
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or marker_id not in ids.reshape(-1):
        return None
    index = int(np.flatnonzero(ids.reshape(-1) == marker_id)[0])
    points = corners[index].reshape(4, 2).astype(np.float64)
    objects = marker_points(side_length_m)
    ok, rvec, tvec = cv2.solvePnP(
        objects, points, np.asarray(camera_matrix), np.asarray(distortion), flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        return None
    pose = transform(cv2.Rodrigues(rvec)[0], tvec)
    if np.any((objects @ pose[:3, :3].T + pose[:3, 3])[:, 2] <= 0):
        return None
    return points, pose
