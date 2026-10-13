"""Task 2: pair board photographs with robot FK and estimate eye-in-hand geometry."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .calibration import board_points, find_board, save_yaml
from .geometry import inverse, rotation_error_rad, transform, validate_transform
from .hand_eye import calibrate_hand_eye
from .image_io import read_image, write_image


def read_calibration(path):
    """Load the fixed camera model, board dimensions and units from task 1."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    matrix = np.asarray(data["K"], dtype=float)
    distortion = np.asarray(data["distortion"], dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("K must be a finite 3x3 camera matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("Invalid focal lengths or camera matrix bottom row")
    if distortion.size not in {4, 5, 8, 12, 14} or not np.isfinite(distortion).all():
        raise ValueError("Invalid OpenCV distortion coefficients")
    size = list(data["image_size"])
    if len(size) != 2 or any(not isinstance(value, int) or value <= 0 for value in size):
        raise ValueError("image_size must contain positive integer width and height")
    corners = tuple(data["board_inner_corners"])
    square = float(data["square_size_m"])
    board_points(corners, square)
    return {
        "K": matrix.tolist(),
        "distortion": distortion.reshape(-1).tolist(),
        "image_size": size,
        "board_inner_corners": list(corners),
        "square_size_m": square,
        "data_source": data.get("data_source", "provided_calibration"),
        "source_file": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def estimate_board_pose(image, calibration, reverse_corners=False, max_rmse_px=2.0):
    """Estimate C_T_M; retain lens distortion and the calibrated pixel resolution."""
    if image is None:
        raise ValueError("Image could not be read")
    if list(image.shape[1::-1]) != calibration["image_size"]:
        raise ValueError("Image resolution differs from task 1 calibration")
    if not np.isfinite(max_rmse_px) or max_rmse_px <= 0:
        raise ValueError("Reprojection threshold must be positive and finite")
    board = tuple(calibration["board_inner_corners"])
    corners = find_board(image, board)
    if corners is None:
        raise ValueError("The complete chessboard was not detected")
    # The SAME physical corner must be M's origin in every photo. The overlay
    # labels corner 0; reverse only when the detector picked the opposite end.
    if reverse_corners:
        corners = corners[::-1].copy()
    objects = board_points(board, calibration["square_size_m"])
    matrix = np.asarray(calibration["K"], dtype=float)
    distortion = np.asarray(calibration["distortion"], dtype=float)
    ok, rvec, tvec = cv2.solvePnP(objects, corners, matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Board pose estimation failed")
    pose = validate_transform(transform(cv2.Rodrigues(rvec)[0], tvec))
    if np.any((objects @ pose[:3, :3].T + pose[:3, 3])[:, 2] <= 0):
        raise ValueError("Estimated board is behind the camera")
    predicted, _ = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)
    rmse = float(np.sqrt(np.mean(np.sum((predicted.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
    if not np.isfinite(rmse) or rmse > max_rmse_px:
        raise ValueError(f"Board reprojection RMS {rmse:.3f} px exceeds {max_rmse_px:g} px")
    annotated = image.copy()
    cv2.drawChessboardCorners(annotated, board, corners, True)
    cv2.drawFrameAxes(annotated, matrix, distortion, rvec, tvec, calibration["square_size_m"] * 3)
    origin = tuple(np.rint(corners[0, 0]).astype(int))
    cv2.circle(annotated, origin, 8, (0, 0, 255), 2)
    cv2.putText(
        annotated,
        "M origin: corner 0",
        (origin[0] + 10, origin[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    return {
        "T_camera_from_board": pose.tolist(),
        "board_rmse_px": rmse,
        "reverse_corners": bool(reverse_corners),
    }, annotated


def _save_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def create_session(directory, intrinsics):
    """Start an empty hand-eye dataset without inventing robot poses."""
    directory = Path(directory).resolve()
    if (directory / "poses.json").exists():
        raise FileExistsError("Session already exists; use status/add instead of reinitializing it")
    calibration = read_calibration(intrinsics)
    directory.mkdir(parents=True, exist_ok=True)
    session = {
        "schema_version": 1,
        "configuration": "eye_in_hand",
        "translation_unit": "metre",
        "created_utc": datetime.now(UTC).isoformat(),
        "intrinsics": calibration,
        "frame_convention": "T_parent_from_child maps child-frame coordinates to the parent frame",
        "pairs": [],
    }
    _save_json(directory / "poses.json", session)
    save_yaml(directory / "camera_intrinsics.yaml", calibration)
    _save_json(
        directory / "robot_pose.template.json",
        {
            "T_base_from_end": None,
            "translation_unit": "metre",
            "end_frame": None,
            "source": "replace with FK computed from measured joint angles",
            "joint_positions_deg": None,
            "observed_at_utc": None,
        },
    )
    return session


def load_session(directory):
    return json.loads((Path(directory) / "poses.json").read_text(encoding="utf-8"))


def read_robot_pose(path):
    """Require a real rigid transform, explicit metre units and a named FK frame."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("translation_unit") != "metre":
        raise ValueError("Robot FK translations must be in metres, not millimetres")
    if not isinstance(data.get("end_frame"), str) or not data["end_frame"].strip():
        raise ValueError("Specify the FK end_frame used for every pose")
    data["T_base_from_end"] = validate_transform(data.get("T_base_from_end")).tolist()
    return data


def append_pair(
    directory,
    image_path,
    robot_pose,
    *,
    validation=False,
    reverse_corners=False,
    max_rmse_px=2.0,
    capture_metadata=None,
):
    """Save a stationary photo/FK pair; origin ordering is checked by the operator."""
    directory = Path(directory).resolve()
    session = load_session(directory)
    if robot_pose.get("translation_unit") != "metre":
        raise ValueError("Robot FK translations must be in metres")
    end_frame = robot_pose.get("end_frame")
    if not isinstance(end_frame, str) or not end_frame.strip():
        raise ValueError("Missing named FK end_frame")
    robot_matrix = validate_transform(robot_pose.get("T_base_from_end"))
    for previous in session["pairs"]:
        if previous["end_frame"] != end_frame:
            raise ValueError("All pairs must use the same FK end_frame")
        if previous.get("base_frame") != robot_pose.get("base_frame"):
            raise ValueError("All pairs must use the same FK base_frame")
        if previous.get("robot_model_sha256") != robot_pose.get("robot_model_sha256"):
            raise ValueError("Robot model or joint calibration changed within the session")
        if not validation and previous["split"] == "train":
            old = np.asarray(previous["T_base_from_end"])
            if (
                np.linalg.norm(robot_matrix[:3, 3] - old[:3, 3]) < 1e-4
                and rotation_error_rad(robot_matrix[:3, :3], old[:3, :3]) < 1e-4
            ):
                raise ValueError("This robot pose is already in the training dataset")
    image_path = Path(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if any(pair["image_sha256"] == digest for pair in session["pairs"]):
        raise ValueError("This photograph is already paired; capture a new photograph")
    estimate, annotated = estimate_board_pose(
        read_image(image_path), session["intrinsics"], reverse_corners, max_rmse_px
    )
    index = len(session["pairs"]) + 1
    relative_image = f"images/pair_{index:03d}.png"
    relative_overlay = f"overlays/pair_{index:03d}.png"
    pair = {
        "id": index,
        "split": "validation" if validation else "train",
        "image": relative_image,
        "overlay": relative_overlay,
        "image_sha256": digest,
        "source_image": str(image_path.resolve()),
        "T_base_from_end": robot_matrix.tolist(),
        "end_frame": end_frame,
        "base_frame": robot_pose.get("base_frame"),
        "robot_model_sha256": robot_pose.get("robot_model_sha256"),
        "robot_pose_source": robot_pose.get("source", "supplied_FK"),
        "joint_positions_deg": robot_pose.get("joint_positions_deg"),
        "robot_observed_at_utc": robot_pose.get("observed_at_utc"),
        "pairing_mode": "operator-held stationary pose",
        "capture_metadata": capture_metadata,
        **estimate,
    }
    write_image(directory / relative_image, read_image(image_path))
    write_image(directory / relative_overlay, annotated)
    session["pairs"].append(pair)
    _save_json(directory / "poses.json", session)
    return pair


def session_status(directory):
    """Report sample count and rotational excitation without driving hardware."""
    session = load_session(directory)
    training = [pair for pair in session["pairs"] if pair["split"] == "train"]
    vectors = []
    if training:
        first = np.asarray(training[0]["T_base_from_end"])[:3, :3]
        vectors = [
            Rotation.from_matrix(first.T @ np.asarray(pair["T_base_from_end"])[:3, :3]).as_rotvec()
            for pair in training[1:]
        ]
    rank = int(np.linalg.matrix_rank(np.asarray(vectors), tol=1e-3)) if vectors else 0
    return {
        "training_pairs": len(training),
        "validation_pairs": len(session["pairs"]) - len(training),
        "rotation_axis_rank": rank,
        "ready_to_solve": len(training) >= 12 and rank >= 2,
        "image_size": session["intrinsics"]["image_size"],
        "board_inner_corners": session["intrinsics"]["board_inner_corners"],
    }


def solve_session(directory, output, measured_board_origin_m=None):
    """Fit training pairs and check held-out pairs and an independently measured board origin."""
    directory = Path(directory)
    output = Path(output)
    session = load_session(directory)
    training = [pair for pair in session["pairs"] if pair["split"] == "train"]
    result = calibrate_hand_eye(
        [pair["T_base_from_end"] for pair in training], [pair["T_camera_from_board"] for pair in training]
    )
    hand_eye = np.asarray(result["T_end_from_camera"])
    board_poses = [
        np.asarray(pair["T_base_from_end"]) @ hand_eye @ np.asarray(pair["T_camera_from_board"])
        for pair in training
    ]
    mean_position = np.asarray(result["board_origin_mean_m"])
    mean_rotation = Rotation.from_matrix(np.array([pose[:3, :3] for pose in board_poses])).mean().as_matrix()
    validation = []
    for pair in session["pairs"]:
        if pair["split"] == "validation":
            predicted = (
                np.asarray(pair["T_base_from_end"]) @ hand_eye @ np.asarray(pair["T_camera_from_board"])
            )
            validation.append(
                {
                    "pair_id": pair["id"],
                    "board_origin_in_base_m": predicted[:3, 3].tolist(),
                    "difference_from_training_mean_m": (predicted[:3, 3] - mean_position).tolist(),
                    "rotation_difference_deg": float(
                        np.degrees(rotation_error_rad(predicted[:3, :3], mean_rotation))
                    ),
                }
            )
    measurement = None
    if measured_board_origin_m is not None:
        measured = np.asarray(measured_board_origin_m, dtype=float)
        if measured.shape != (3,) or not np.isfinite(measured).all():
            raise ValueError("Measured board origin must be three finite coordinates in metres")
        axis_errors = np.abs(mean_position - measured)
        measurement = {
            "measured_board_origin_m": measured.tolist(),
            "absolute_axis_errors_m": axis_errors.tolist(),
            "each_axis_within_5cm": bool(np.all(axis_errors <= 0.05)),
        }
    result.update(
        {
            "T_camera_from_end": inverse(hand_eye).tolist(),
            "camera_origin_in_end_m": hand_eye[:3, 3].tolist(),
            "camera_offset_norm_m": float(np.linalg.norm(hand_eye[:3, 3])),
            "camera_axes_in_end": {
                name: hand_eye[:3, index].tolist()
                for index, name in enumerate(["x_right", "y_down", "z_forward"])
            },
            "end_frame": training[0]["end_frame"],
            "base_frame": training[0].get("base_frame"),
            "robot_model_sha256": training[0].get("robot_model_sha256"),
            "T_base_from_board_mean": transform(mean_rotation, mean_position).tolist(),
            "validation": validation,
            "independent_board_measurement": measurement,
            "data_source": session["intrinsics"]["data_source"],
            "intrinsics_sha256": session["intrinsics"]["source_sha256"],
            "pairs_file": str((directory / "poses.json").resolve()),
            "note": "Board spread checks consistency; the course's 5cm accuracy check needs an independent measurement.",
        }
    )
    result["validation_origin_max_error_m"] = max(
        (float(np.linalg.norm(row["difference_from_training_mean_m"])) for row in validation), default=None
    )
    result["validated_for_servo"] = bool(
        measurement
        and measurement["each_axis_within_5cm"]
        and len(validation) >= 3
        and result["validation_origin_max_error_m"] <= 0.05
        and result["board_origin_spread_rms_m"] <= 0.05
        and result["base_frame"]
        and result["data_source"] == "real_usb_camera"
    )
    output.mkdir(parents=True, exist_ok=True)
    save_yaml(output / "hand_eye.yaml", result)
    _save_json(output / "hand_eye.json", result)
    lines = [
        "任务2手眼标定",
        f"FK末端坐标系：{result['end_frame']}",
        f"训练配对：{len(training)}；独立验证配对：{len(validation)}",
        "T_end_from_camera 将相机坐标转换到末端坐标，平移单位为米。",
        f"相机原点在末端系的位置：{result['camera_origin_in_end_m']}",
        f"相机到末端原点的距离：{result['camera_offset_norm_m']:.4f} m",
        f"板原点位置一致性RMS：{result['board_origin_spread_rms_m']:.4f} m",
        "手眼矩阵：",
        np.array2string(hand_eye, precision=6),
        f"独立实测板原点检查：{measurement}",
        "板原点必须对应照片中标出的同一个物理内角点。",
    ]
    (output / "手眼标定报告.txt").write_text("\n".join(lines), encoding="utf-8-sig")
    return result


def joint_positions_in_degrees(observation, joint_names, units, calibration=None):
    """Map SCS215 normalized positions using measured URDF angle endpoints."""
    values = []
    for name in joint_names:
        value = float(observation[f"{name}.pos"])
        if not np.isfinite(value):
            raise ValueError("Joint positions must be finite")
        if units == "degrees":
            angle = value
        elif units == "normalized_0_1":
            if not 0 <= value <= 1:
                raise ValueError("Normalized SCS215 positions must be between 0 and 1")
            if calibration is None or name not in calibration:
                raise ValueError(f"Missing normalized-to-URDF-degree calibration for {name}")
            limits = calibration[name]
            endpoints = np.asarray([limits.get("q_at_0_deg"), limits.get("q_at_1_deg")], dtype=float)
            if not np.isfinite(endpoints).all() or endpoints[0] == endpoints[1]:
                raise ValueError(f"Supply measured angular endpoints for {name}; do not guess the zero/sign")
            angle = endpoints[0] + value * (endpoints[1] - endpoints[0])
        else:
            raise ValueError("Explicitly select degrees or normalized_0_1 input units")
        values.append(angle)
    return np.asarray(values)


def pose_from_observation(kinematics, observation, joint_names, units, calibration=None):
    """Reuse LeRobot's FK with an explicit joint order and correct angle units."""
    if not joint_names or len(set(joint_names)) != len(joint_names):
        raise ValueError("Supply distinct URDF joint names in the FK input order")
    angles = joint_positions_in_degrees(observation, joint_names, units, calibration)
    pose = validate_transform(kinematics.forward_kinematics(angles))
    return {
        "T_base_from_end": pose.tolist(),
        "translation_unit": "metre",
        "end_frame": kinematics.target_frame_name,
        "source": "lerobot.model.RobotKinematics.forward_kinematics",
        "joint_positions_deg": dict(zip(joint_names, angles.tolist(), strict=True)),
        "observed_at_utc": datetime.now(UTC).isoformat(),
    }
