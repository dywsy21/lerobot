import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import validate_transform


class DHRobot:
    """Generic standard-DH robot; a/d in metres, alpha/offset/q in radians.

    This is NOT a calibrated SCS215 robot model. Supply the actual link geometry,
    joint signs/zeros and limits when hardware becomes available.
    """

    def __init__(self, parameters, joint_limits):
        self.parameters = np.asarray(parameters, dtype=float)
        self.limits = np.asarray(joint_limits, dtype=float)
        if (
            self.parameters.ndim != 2
            or self.parameters.shape[1] != 4
            or self.limits.shape != (len(self.parameters), 2)
        ):
            raise ValueError("DH rows are [a, alpha, d, theta_offset]; limits are Nx2")
        if (
            not np.isfinite(self.parameters).all()
            or not np.isfinite(self.limits).all()
            or np.any(self.limits[:, 0] >= self.limits[:, 1])
        ):
            raise ValueError("Invalid DH parameters or joint limits")

    def forward(self, angles):
        angles = np.asarray(angles, dtype=float).reshape(-1)
        if len(angles) != len(self.parameters) or not np.isfinite(angles).all():
            raise ValueError("Invalid joint angle vector")
        pose = np.eye(4)
        for q, (a, alpha, d, offset) in zip(angles, self.parameters, strict=True):
            ct, st = np.cos(q + offset), np.sin(q + offset)
            ca, sa = np.cos(alpha), np.sin(alpha)
            pose = pose @ np.array(
                [
                    [ct, -st * ca, st * sa, a * ct],
                    [st, ct * ca, -ct * sa, a * st],
                    [0, sa, ca, d],
                    [0, 0, 0, 1],
                ]
            )
        return pose

    def inverse(self, target, initial, position_tolerance_m=1e-4, rotation_tolerance_rad=1e-3):
        target = validate_transform(target)
        initial = np.asarray(initial, dtype=float).reshape(-1)
        if (
            len(initial) != len(self.parameters)
            or np.any(initial < self.limits[:, 0])
            or np.any(initial > self.limits[:, 1])
        ):
            raise ValueError("Initial angles must be inside the supplied joint limits")

        def residual(q):
            current = self.forward(q)
            return np.r_[
                current[:3, 3] - target[:3, 3],
                Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(),
            ]

        solution = least_squares(
            residual,
            initial,
            bounds=(self.limits[:, 0], self.limits[:, 1]),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=500,
        )
        error = residual(solution.x)
        if (
            np.linalg.norm(error[:3]) > position_tolerance_m
            or np.linalg.norm(error[3:]) > rotation_tolerance_rad
        ):
            raise ValueError("Target is unreachable or IK did not converge within the tolerances")
        return solution.x
