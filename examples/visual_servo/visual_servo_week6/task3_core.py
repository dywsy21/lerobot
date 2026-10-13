"""Task 3: image-feature control and joint proposals, without motor I/O."""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .geometry import inverse, validate_transform
from .ibvs import compute_ibvs, interaction_matrix, simple_camera_step
from .markers import marker_points


@dataclass(frozen=True)
class ServoSettings:
    dictionary: str = "DICT_4X4_50"
    marker_id: int = 0
    side_length_m: float = 0.06
    marker_size_confirmed: bool = False
    mode: str = "ibvs"
    gain: float = 0.8
    damping: float = 0.02
    dt_s: float = 0.1
    max_linear_speed_m_s: float = 0.01
    max_angular_speed_rad_s: float = 0.10
    stop_rmse_px: float = 2.0
    minimum_depth_m: float = 0.15
    maximum_depth_m: float = 2.0
    maximum_frame_age_s: float = 0.5
    maximum_pose_rmse_px: float = 2.0
    maximum_joint_step_rad: float = 0.006

    def __post_init__(self):
        values = [
            self.side_length_m,
            self.gain,
            self.dt_s,
            self.damping,
            self.max_linear_speed_m_s,
            self.max_angular_speed_rad_s,
            self.stop_rmse_px,
            self.minimum_depth_m,
            self.maximum_depth_m,
            self.maximum_frame_age_s,
            self.maximum_pose_rmse_px,
            self.maximum_joint_step_rad,
        ]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Servo settings must be finite and positive")
        if self.mode not in {"ibvs", "rule"} or self.minimum_depth_m >= self.maximum_depth_m:
            raise ValueError("Invalid mode or depth limits")
        if not self.dictionary.startswith("DICT_") or not hasattr(cv2.aruco, self.dictionary):
            raise ValueError("Unknown ArUco dictionary")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        if not 0 <= self.marker_id < len(dictionary.bytesList):
            raise ValueError("Marker ID is outside the selected dictionary")


def load_intrinsics(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    camera_matrix, distortion = np.asarray(data["K"], float), np.asarray(data["distortion"], float)
    size = tuple(data["image_size"])
    if (
        camera_matrix.shape != (3, 3)
        or not np.isfinite(camera_matrix).all()
        or camera_matrix[0, 0] <= 0
        or camera_matrix[1, 1] <= 0
        or not np.isfinite(distortion).all()
        or distortion.size not in {4, 5, 8, 12, 14}
        or not np.allclose(camera_matrix[2], [0, 0, 1])
        or len(size) != 2
        or min(size) <= 0
    ):
        raise ValueError("Invalid intrinsic calibration file")
    return camera_matrix, distortion, size, data


def project_marker(pose, camera_matrix, distortion, side_length_m):
    pose = validate_transform(pose)
    points, _ = cv2.projectPoints(
        marker_points(side_length_m), cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3], camera_matrix, distortion
    )
    return points.reshape(4, 2)


def central_goal(camera_matrix, distortion, size, side_length_m, depth_m=0.55):
    pose = np.eye(4)
    pose[:3, 3] = [
        (size[0] / 2 - camera_matrix[0, 2]) * depth_m / camera_matrix[0, 0],
        (size[1] / 2 - camera_matrix[1, 2]) * depth_m / camera_matrix[1, 1],
        depth_m,
    ]
    return project_marker(pose, camera_matrix, distortion, side_length_m), pose


def detect_aruco_pose(image, camera_matrix, distortion, settings):
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, settings.dictionary)), parameters
    )
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or settings.marker_id not in ids.reshape(-1):
        return None
    # A duplicate ID is ambiguous, e.g. a marker and its live image on a screen.
    indices = np.flatnonzero(ids.reshape(-1) == settings.marker_id)
    if len(indices) != 1:
        return None
    points = corners[int(indices[0])].reshape(4, 2).astype(np.float64)
    if abs(cv2.contourArea(points.astype(np.float32))) < 100:
        return None
    objects = marker_points(settings.side_length_m)
    solutions = cv2.solvePnPGeneric(objects, points, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    if not solutions[0]:
        return None
    candidates = []
    for rvec, tvec in zip(solutions[1], solutions[2], strict=True):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.reshape(3)
        depths = (objects @ pose[:3, :3].T + pose[:3, 3])[:, 2]
        if np.any(depths <= 0):
            continue
        projected, _ = cv2.projectPoints(objects, rvec, tvec, camera_matrix, distortion)
        error = float(np.sqrt(np.mean(np.sum((points - projected.reshape(4, 2)) ** 2, axis=1))))
        candidates.append((error, pose, depths))
    if not candidates:
        return None
    error, pose, depths = min(candidates, key=lambda x: x[0])
    return points, pose, depths, error


class VisualServo:
    def __init__(self, camera_matrix, distortion, image_size, settings=None, goal=None):
        self.camera_matrix, self.distortion = np.asarray(camera_matrix, float), np.asarray(distortion, float)
        self.size, self.settings = tuple(image_size), settings or ServoSettings()
        default_goal, _ = central_goal(
            self.camera_matrix, self.distortion, self.size, self.settings.side_length_m
        )
        self.set_goal(default_goal if goal is None else goal)

    def set_goal(self, points):
        points = np.asarray(points, float)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError("Goal must contain four finite [u,v] pixel coordinates")
        if np.any(points < 0) or np.any(points >= np.asarray(self.size)):
            raise ValueError("Goal points must lie inside the calibrated image")
        polygon = points.astype(np.float32)
        if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon, oriented=True) <= 100:
            raise ValueError(
                "Goal must be a convex ordered quadrilateral: top-left, top-right, bottom-right, bottom-left"
            )
        self.goal = points.copy()

    def stopped(self, status, reason):
        return {
            "status": status,
            "reason": reason,
            "rmse_px": None,
            "corners_px": None,
            "desired_px": self.goal.tolist(),
            "velocity_camera": [0.0] * 6,
            "step_camera_m": [0.0] * 3,
            "hardware_commands_sent": False,
            "marker_size_confirmed": self.settings.marker_size_confirmed,
        }

    def evaluate(self, image, *, captured_at=None, now=None, active=True, allow_resize=False):
        now = time.monotonic() if now is None else now
        captured_at = now if captured_at is None else captured_at
        age = now - captured_at
        if not np.isfinite(age) or age < -0.05 or age > self.settings.maximum_frame_age_s:
            return self.stopped("stale_frame", "Camera frame is not fresh")
        if image is None:
            return self.stopped("target_lost", "No camera image")
        h, w = image.shape[:2]
        scale = np.array([w / self.size[0], h / self.size[1]])
        if not np.allclose(scale, 1) and (not allow_resize or not np.isclose(*scale)):
            return self.stopped(
                "resolution_mismatch", "Use the calibrated resolution or an explicitly resized broker frame"
            )
        camera_matrix = self.camera_matrix.copy()
        camera_matrix[0, :] *= scale[0]
        camera_matrix[1, :] *= scale[1]
        camera_matrix[0, 2] += (scale[0] - 1) / 2
        camera_matrix[1, 2] += (scale[1] - 1) / 2
        detection = detect_aruco_pose(image, camera_matrix, self.distortion, self.settings)
        if detection is None:
            return self.stopped("target_lost", "Requested marker missing or ambiguous")
        pixels, pose, depths, pose_error = detection
        pixels = (pixels + 0.5) / scale - 0.5
        if pose_error / min(scale) > self.settings.maximum_pose_rmse_px:
            return self.stopped("pose_unreliable", "Marker pose reprojection error is too high")
        if min(depths) <= self.settings.minimum_depth_m or max(depths) >= self.settings.maximum_depth_m:
            return self.stopped("depth_limit", "Estimated marker depth is outside the configured range")
        result = compute_ibvs(
            pixels,
            self.goal,
            self.camera_matrix,
            self.distortion,
            depths,
            gain=self.settings.gain,
            damping=self.settings.damping,
            max_linear_speed=self.settings.max_linear_speed_m_s,
            max_angular_speed=self.settings.max_angular_speed_rad_s,
            stop_rmse_px=self.settings.stop_rmse_px,
        )
        velocity = result.velocity_camera.copy()
        step = velocity[:3] * self.settings.dt_s
        status = result.status
        if self.settings.mode == "rule":
            step = simple_camera_step(pixels, self.size, float(depths.mean()))
            velocity = np.r_[step / self.settings.dt_s, np.zeros(3)]
            status = "rule_step"
        if not active:
            velocity, step, status = np.zeros(6), np.zeros(3), "paused"
        return {
            "status": status,
            "reason": "",
            "rmse_px": result.rmse_px,
            "corners_px": pixels.tolist(),
            "desired_px": self.goal.tolist(),
            "T_camera_from_marker": pose.tolist(),
            "depths_m": depths.tolist(),
            "pose_rmse_px": pose_error / min(scale),
            "velocity_camera": velocity.tolist(),
            "step_camera_m": step.tolist(),
            "frame_age_s": age,
            "marker_size_confirmed": self.settings.marker_size_confirmed,
            "hardware_commands_sent": False,
        }


def camera_body_jacobian(fk, q, t_end_from_camera, epsilon=1e-5):
    """J_camera maps active joint radians/s to body camera twist [m/s, rad/s]."""
    q = np.asarray(q, float).reshape(-1)
    x = validate_transform(t_end_from_camera)
    current = validate_transform(fk(q)) @ x
    jacobian = np.empty((6, len(q)))
    for i in range(len(q)):
        plus, minus = q.copy(), q.copy()
        plus[i] += epsilon
        minus[i] -= epsilon
        a, b = (
            inverse(current) @ validate_transform(fk(plus)) @ x,
            inverse(current) @ validate_transform(fk(minus)) @ x,
        )
        jacobian[:3, i] = (a[:3, 3] - b[:3, 3]) / (2 * epsilon)
        jacobian[3:, i] = (
            Rotation.from_matrix(a[:3, :3]).as_rotvec() - Rotation.from_matrix(b[:3, :3]).as_rotvec()
        ) / (2 * epsilon)
    return jacobian


def propose_joint_step(advice, servo, fk, q, limits, t_end_from_camera):
    """A bounded joint-space IBVS proposal. Gripper must be excluded by caller.

    This handles arms with fewer than six camera degrees of freedom; it does not
    claim that an arbitrary six-DOF camera pose is reachable, and sends no commands.
    """
    q, limits = np.asarray(q, float), np.asarray(limits, float)
    if advice["status"] != "tracking":
        return {"status": advice["status"], "q_next_rad": q.tolist(), "dq_rad_s": [0.0] * len(q)}
    if limits.shape != (len(q), 2) or not np.isfinite(q).all() or not np.isfinite(limits).all():
        raise ValueError("Invalid joint vector or calibrated joint limits")
    if np.any(limits[:, 0] >= limits[:, 1]) or np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        raise ValueError("Current joints are outside valid calibrated limits")
    points = np.asarray(advice["corners_px"])
    current = cv2.undistortPoints(points[:, None, :], servo.camera_matrix, servo.distortion).reshape(4, 2)
    desired = cv2.undistortPoints(servo.goal[:, None, :], servo.camera_matrix, servo.distortion).reshape(4, 2)
    jacobian = camera_body_jacobian(fk, q, t_end_from_camera)
    a = interaction_matrix(current, advice["depths_m"]) @ jacobian
    error = (current - desired).reshape(-1)
    dq = -servo.settings.gain * np.linalg.solve(
        a.T @ a + servo.settings.damping**2 * np.eye(len(q)), a.T @ error
    )
    delta = dq * servo.settings.dt_s
    delta /= max(1.0, float(np.max(np.abs(delta))) / servo.settings.maximum_joint_step_rad)
    achievable = jacobian @ (delta / servo.settings.dt_s)
    delta /= max(
        1.0,
        float(np.linalg.norm(achievable[:3])) / servo.settings.max_linear_speed_m_s,
        float(np.linalg.norm(achievable[3:])) / servo.settings.max_angular_speed_rad_s,
    )
    fraction = 1.0
    for i, value in enumerate(delta):
        if value > 0:
            fraction = min(fraction, (limits[i, 1] - q[i]) / value)
        elif value < 0:
            fraction = min(fraction, (limits[i, 0] - q[i]) / value)
    delta *= max(0.0, fraction)
    return {
        "status": "joint_limit" if fraction < 0.05 else "proposal",
        "q_next_rad": (q + delta).tolist(),
        "dq_rad_s": (delta / servo.settings.dt_s).tolist(),
        "camera_velocity_achievable": (jacobian @ (delta / servo.settings.dt_s)).tolist(),
        "hardware_commands_sent": False,
    }
