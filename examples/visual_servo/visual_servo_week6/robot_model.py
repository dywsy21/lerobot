"""Portable URDF FK/IK and the explicit SCS215 angle contract."""

import hashlib
import json
from pathlib import Path

import numpy as np
from defusedxml import ElementTree
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import inverse, transform, validate_transform
from .hand_eye_workflow import joint_positions_in_degrees, pose_from_observation


def vector(element, attribute, default):
    values = np.fromstring(element.get(attribute, default) if element is not None else default, sep=" ")
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"URDF {attribute} must contain three finite values")
    return values


class UrdfKinematics:
    """Serial path FK without meshes or placo; external angles are degrees.

    Implements fixed and revolute joints, including continuous joints with
    operator-supplied finite limits. Rejects unsupported/mimic joints explicitly.
    URDF origins use Rz(yaw) Ry(pitch) Rx(roll), axes in the joint frame.
    """

    def __init__(self, urdf_path, end_frame, joint_names, base_frame=None):
        root = ElementTree.parse(urdf_path).getroot()
        links = {link.get("name") for link in root.findall("link")}
        parents, chain = {}, []
        for joint in root.findall("joint"):
            child = joint.find("child").get("link")
            if child in parents:
                raise ValueError("URDF is not a tree: duplicate child link")
            parents[child] = joint
        if end_frame not in links:
            raise ValueError(f"Unknown URDF end_frame: {end_frame}")
        current, visited = end_frame, set()
        while current in parents and current != base_frame:
            if current in visited:
                raise ValueError("URDF contains a cycle")
            visited.add(current)
            joint = parents[current]
            chain.append(joint)
            current = joint.find("parent").get("link")
        if base_frame is not None and current != base_frame:
            raise ValueError("end_frame is not a descendant of base_frame")
        self.base_frame = current
        self.target_frame_name = end_frame
        self.joint_names = list(joint_names)
        self.chain = list(reversed(chain))
        movable = [j.get("name") for j in self.chain if j.get("type") != "fixed"]
        if len(set(joint_names)) != len(joint_names) or set(movable) != set(joint_names):
            raise ValueError(f"joint_names must contain exactly the movable end-frame path: {movable}")
        self.urdf_limits = {}
        for joint in self.chain:
            kind, name = joint.get("type"), joint.get("name")
            if joint.find("mimic") is not None or kind not in {"fixed", "revolute", "continuous"}:
                raise ValueError(f"Unsupported URDF joint {name}: {kind} or mimic")
            if kind != "fixed" and np.linalg.norm(vector(joint.find("axis"), "xyz", "1 0 0")) < 1e-12:
                raise ValueError(f"Zero joint axis: {name}")
            if kind == "revolute":
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError(f"URDF revolute joint requires limits: {name}")
                self.urdf_limits[name] = [float(limit.get("lower")), float(limit.get("upper"))]

    def forward_kinematics(self, angles):
        angles = np.asarray(angles, float)
        if angles.shape != (len(self.joint_names),) or not np.isfinite(angles).all():
            raise ValueError("FK needs one finite degree angle per configured joint")
        positions = dict(zip(self.joint_names, np.deg2rad(angles), strict=True))
        result = np.eye(4)
        for joint in self.chain:
            origin = joint.find("origin")
            result = result @ transform(
                Rotation.from_euler("xyz", vector(origin, "rpy", "0 0 0")).as_matrix(),
                vector(origin, "xyz", "0 0 0"),
            )
            if joint.get("type") != "fixed":
                axis = vector(joint.find("axis"), "xyz", "1 0 0")
                rotation = Rotation.from_rotvec(axis / np.linalg.norm(axis) * positions[joint.get("name")])
                result = result @ transform(rotation.as_matrix())
        return validate_transform(result)

    def inverse_kinematics(self, current_angles, desired_pose, limits_deg):
        """Bounded numerical IK; reject unreachable targets instead of moving."""
        current = np.deg2rad(np.asarray(current_angles, float))
        limits = np.deg2rad(np.asarray(limits_deg, float))
        target = validate_transform(desired_pose)

        def residual(q):
            relative = inverse(target) @ self.forward_kinematics(np.rad2deg(q))
            return np.r_[relative[:3, 3], 0.1 * Rotation.from_matrix(relative[:3, :3]).as_rotvec()]

        solved = least_squares(residual, current, bounds=(limits[:, 0], limits[:, 1]), max_nfev=150)
        actual = self.forward_kinematics(np.rad2deg(solved.x))
        translation_error = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        rotation_error = float(
            np.linalg.norm(Rotation.from_matrix(actual[:3, :3].T @ target[:3, :3]).as_rotvec())
        )
        if not solved.success or translation_error > 0.003 or rotation_error > np.deg2rad(3):
            raise ValueError(
                f"IK target unreachable: {translation_error:.4f} m, {np.rad2deg(rotation_error):.2f} deg"
            )
        return np.rad2deg(solved.x)


class RobotModel:
    def __init__(self, path, repository):
        self.path = Path(path).resolve()
        self.config = json.loads(self.path.read_text(encoding="utf-8-sig"))
        config = self.config
        if not config.get("urdf_path") or not config.get("end_frame") or not config.get("joint_names"):
            raise ValueError("填写 robot_model.json 的 urdf_path、end_frame 和 joint_names")
        urdf = Path(repository) / config["urdf_path"]
        if not urdf.is_file():
            raise ValueError(f"URDF 文件不存在：{urdf}")
        self.names = config["joint_names"]
        fingerprint = self.path.read_bytes() + urdf.read_bytes()
        motor_file = config.get("motor_calibration_path")
        if motor_file and (Path(repository) / motor_file).is_file():
            fingerprint += (Path(repository) / motor_file).read_bytes()
        self.fingerprint = hashlib.sha256(fingerprint).hexdigest()
        joint_positions_in_degrees(
            {f"{name}.pos": 0.5 for name in self.names},
            self.names,
            config["observation_units"],
            config.get("normalized_joint_calibration"),
        )
        backend = config.get("backend", "portable_urdf")
        # Always inspect the URDF chain, even when using the repository backend.
        portable = UrdfKinematics(urdf, config["end_frame"], self.names, config.get("base_frame"))
        if backend == "portable_urdf":
            self.kinematics = portable
        elif backend == "lerobot_placo":
            from lerobot.model.kinematics import RobotKinematics

            if config.get("base_frame") not in {None, portable.base_frame}:
                raise ValueError("Repository placo FK uses the URDF root as base")
            self.kinematics = RobotKinematics(str(urdf), config["end_frame"], self.names)
        else:
            raise ValueError("backend must be portable_urdf or lerobot_placo")
        self.portable = portable
        configured = config.get("joint_limits_deg") or {}
        bounds = []
        for name in self.names:
            given = configured.get(name)
            if given is None:
                raise ValueError(f"填写实测安全关节范围 joint_limits_deg.{name}")
            bound = np.asarray(given, float)
            if bound.shape != (2,) or not np.isfinite(bound).all() or bound[0] >= bound[1]:
                raise ValueError(f"Invalid calibrated joint limits: {name}")
            if name in portable.urdf_limits:
                urdf_bound = np.rad2deg(portable.urdf_limits[name])
                bound = [max(bound[0], urdf_bound[0]), min(bound[1], urdf_bound[1])]
            if bound[0] >= bound[1]:
                raise ValueError(f"Calibrated and URDF limits do not overlap: {name}")
            bounds.append(bound)
        self.limits_rad = np.deg2rad(bounds)

    def angles(self, observation):
        degrees = joint_positions_in_degrees(
            observation,
            self.names,
            self.config["observation_units"],
            self.config.get("normalized_joint_calibration"),
        )
        q = np.deg2rad(degrees)
        if np.any(q < self.limits_rad[:, 0]) or np.any(q > self.limits_rad[:, 1]):
            raise ValueError("Measured joints exceed calibrated/URDF limits")
        return q

    def fk_radians(self, q):
        return self.kinematics.forward_kinematics(np.rad2deg(q))

    def pose(self, observation):
        self.angles(observation)
        pose = pose_from_observation(
            self.kinematics,
            observation,
            self.names,
            self.config["observation_units"],
            self.config.get("normalized_joint_calibration"),
        )
        pose["source"] = f"{self.config.get('backend', 'portable_urdf')}.forward_kinematics"
        pose["base_frame"] = self.portable.base_frame
        pose["robot_model_sha256"] = self.fingerprint
        return pose

    def action(self, q):
        q = np.asarray(q, float)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise ValueError("Invalid commanded joint vector")
        if np.any(q < self.limits_rad[:, 0]) or np.any(q > self.limits_rad[:, 1]):
            raise ValueError("Command exceeds joint limits")
        values = np.rad2deg(q)
        if self.config["observation_units"] == "normalized_0_1":
            mapping = self.config["normalized_joint_calibration"]
            values = np.array(
                [
                    (value - mapping[name]["q_at_0_deg"])
                    / (mapping[name]["q_at_1_deg"] - mapping[name]["q_at_0_deg"])
                    for name, value in zip(self.names, values, strict=True)
                ]
            )
            if np.any(values < 0) or np.any(values > 1):
                raise ValueError("Command is outside normalized calibration range")
        return dict(zip(self.names, values.tolist(), strict=True))
