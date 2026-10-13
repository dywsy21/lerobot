import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


def transform(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = np.asarray(rotation)
    if translation is not None:
        result[:3, 3] = np.asarray(translation).reshape(3)
    return result


def validate_transform(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Pose must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(
        pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5
    ):
        raise ValueError("Pose is not a rigid transform")
    if not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-5):
        raise ValueError("Rotation must have determinant +1")
    return pose


def inverse(pose):
    pose = validate_transform(pose)
    rotation = pose[:3, :3].T
    return transform(rotation, -rotation @ pose[:3, 3])


def se3_exp(velocity, dt):
    """Body twist ordered [vx, vy, vz, wx, wy, wz]; metres/radians per second."""
    velocity = np.asarray(velocity, dtype=float).reshape(6)
    if not np.isfinite(velocity).all() or dt <= 0:
        raise ValueError("Twist must be finite and dt positive")
    wx, wy, wz = velocity[3:]
    hat = np.zeros((4, 4))
    hat[:3, :3] = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]
    hat[:3, 3] = velocity[:3]
    return expm(hat * dt)


def camera_velocity_to_end_target(t_base_from_end, t_end_from_camera, velocity, dt):
    """Matrices explicitly map coordinates FROM the child frame TO the parent."""
    t_base_from_camera = validate_transform(t_base_from_end) @ validate_transform(t_end_from_camera)
    next_t_base_from_camera = t_base_from_camera @ se3_exp(velocity, dt)
    return next_t_base_from_camera @ inverse(t_end_from_camera)


def rotation_error_rad(first, second):
    return float(np.linalg.norm(Rotation.from_matrix(first @ second.T).as_rotvec()))
