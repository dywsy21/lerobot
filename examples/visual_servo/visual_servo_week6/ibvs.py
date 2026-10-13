from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ControlResult:
    velocity_camera: np.ndarray
    rmse_px: float
    status: str


def simple_camera_step(points_px, image_size, depth_m, step_m=0.01, minimum_depth_m=0.15):
    """Course's image-region rule; returns an advisory camera translation in metres."""
    if points_px is None or depth_m <= minimum_depth_m:
        return np.zeros(3)
    width, height = image_size
    u, v = np.asarray(points_px).reshape(-1, 2).mean(axis=0)
    if u < 0.4 * width:
        return np.array([-step_m, 0, 0])
    if u > 0.6 * width:
        return np.array([step_m, 0, 0])
    if v < 0.4 * height:
        return np.array([0, -step_m, 0])
    if v > 0.6 * height:
        return np.array([0, step_m, 0])
    return np.array([0, 0, step_m])


def interaction_matrix(normalized_points, depths_m):
    """Point-feature IBVS: normalized x/y and camera twist [v, omega]."""
    points = np.asarray(normalized_points, dtype=float).reshape(-1, 2)
    depths = np.asarray(depths_m, dtype=float).reshape(-1)
    if (
        len(points) != len(depths)
        or not np.isfinite(points).all()
        or not np.isfinite(depths).all()
        or np.any(depths <= 0)
    ):
        raise ValueError("Need finite normalized points and one positive depth per point")
    rows = []
    for (x, y), z in zip(points, depths, strict=True):
        rows.extend([[-1 / z, 0, x / z, x * y, -(1 + x * x), y], [0, -1 / z, y / z, 1 + y * y, -x * y, -x]])
    return np.asarray(rows)


def compute_ibvs(
    points_px,
    desired_px,
    camera_matrix,
    distortion,
    depths_m,
    *,
    gain=1.2,
    damping=0.02,
    max_linear_speed=0.05,
    max_angular_speed=0.3,
    stop_rmse_px=2.0,
):
    """Compute an advisory camera velocity; this function never drives hardware."""
    if points_px is None:
        return ControlResult(np.zeros(6), float("inf"), "target_lost")
    current = np.asarray(points_px, dtype=float).reshape(-1, 2)
    desired = np.asarray(desired_px, dtype=float).reshape(-1, 2)
    if (
        len(current) < 4
        or current.shape != desired.shape
        or not np.isfinite(current).all()
        or not np.isfinite(desired).all()
    ):
        raise ValueError("Need matching, finite sets of at least four points")
    if gain <= 0 or damping < 0 or max_linear_speed <= 0 or max_angular_speed <= 0 or stop_rmse_px <= 0:
        raise ValueError("Invalid controller parameters")
    matrix = np.asarray(camera_matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    rmse = float(np.sqrt(np.mean(np.sum((current - desired) ** 2, axis=1))))
    normalized = cv2.undistortPoints(current.reshape(-1, 1, 2), matrix, np.asarray(distortion)).reshape(-1, 2)
    goal = cv2.undistortPoints(desired.reshape(-1, 1, 2), matrix, np.asarray(distortion)).reshape(-1, 2)
    interaction = interaction_matrix(normalized, depths_m)
    if rmse <= stop_rmse_px:
        return ControlResult(np.zeros(6), rmse, "reached")
    error = (normalized - goal).reshape(-1)
    velocity = -gain * np.linalg.solve(
        interaction.T @ interaction + damping**2 * np.eye(6), interaction.T @ error
    )
    # Scale the whole twist so translational/rotational clipping preserves its direction.
    scale = max(
        1.0, np.linalg.norm(velocity[:3]) / max_linear_speed, np.linalg.norm(velocity[3:]) / max_angular_speed
    )
    velocity /= scale
    return ControlResult(velocity, rmse, "tracking")
