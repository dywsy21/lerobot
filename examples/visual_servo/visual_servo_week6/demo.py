import json
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .calibration import calibrate_images, save_yaml
from .geometry import inverse, rotation_error_rad, se3_exp, transform
from .hand_eye import calibrate_hand_eye
from .ibvs import compute_ibvs
from .markers import detect_marker, marker_points
from .synthetic import generate_board_views, project, render_marker


def run_demo(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    print("Generating 20 synthetic checkerboard views...", flush=True)
    records, true_camera_matrix, true_hand_eye_matrix, fixed_board = generate_board_views(output / "images")
    intrinsics, _ = calibrate_images(sorted(records))
    intrinsics["data_source"] = "synthetic; do not use these intrinsics for a real camera"
    save_yaml(output / "intrinsics.synthetic.yaml", intrinsics)
    camera_matrix = np.asarray(intrinsics["K"])
    distortion = np.asarray(intrinsics["distortion"])
    print(f"Calibration: {intrinsics['valid_images']} images, RMS={intrinsics['rms_px']:.4f} px", flush=True)
    views = intrinsics["views"]
    robot_poses = [records[view["file"]]["T_base_from_end"] for view in views]
    board_poses = [np.asarray(view["T_camera_from_board"]) for view in views]
    pairs = [
        {"image": view["file"], "T_base_from_end": robot.tolist(), "T_camera_from_board": board.tolist()}
        for view, robot, board in zip(views, robot_poses, board_poses, strict=True)
    ]
    (output / "poses.synthetic.json").write_text(
        json.dumps({"data_source": "synthetic", "pairs": pairs}, indent=2), encoding="utf-8"
    )
    hand_eye = calibrate_hand_eye(robot_poses, board_poses)
    hand_eye["data_source"] = "synthetic; do not use this transform for a real robot"
    save_yaml(output / "hand-eye.synthetic.yaml", hand_eye)
    hand_eye_matrix = np.asarray(hand_eye["T_end_from_camera"])
    translation_error = float(np.linalg.norm(hand_eye_matrix[:3, 3] - true_hand_eye_matrix[:3, 3]))
    rotation_error = float(
        np.degrees(rotation_error_rad(hand_eye_matrix[:3, :3], true_hand_eye_matrix[:3, :3]))
    )
    board_origin_error = np.abs(np.asarray(hand_eye["board_origin_mean_m"]) - fixed_board[:3, 3])
    print(f"Hand-eye translation error: {translation_error * 1000:.2f} mm", flush=True)

    objects = marker_points(0.06)
    pose = transform(Rotation.from_euler("xyz", [0.1, -0.12, 0.08]).as_matrix(), [0.06, -0.04, 0.65])
    goal_pose = transform(translation=[0, 0, 0.45])
    goal = project(objects, goal_pose, true_camera_matrix)
    history = []
    video = output / "ibvs.offline.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = 1280, 720, "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        for step in range(160):
            image = render_marker(pose, true_camera_matrix)
            detection = detect_marker(image, camera_matrix, distortion)
            if detection is None:
                raise RuntimeError(f"Synthetic marker lost at step {step}")
            pixels, estimated_pose = detection
            depths = (objects @ estimated_pose[:3, :3].T + estimated_pose[:3, 3])[:, 2]
            command = compute_ibvs(pixels, goal, camera_matrix, distortion, depths)
            history.append(
                {
                    "step": step,
                    "rmse_px": command.rmse_px,
                    "velocity_camera": command.velocity_camera.tolist(),
                    "status": command.status,
                }
            )
            cv2.polylines(image, [goal.round().astype(np.int32)], True, (40, 150, 40), 2)
            cv2.polylines(image, [pixels.round().astype(np.int32)], True, (200, 100, 30), 2)
            cv2.putText(
                image,
                f"OFFLINE simulation | step {step} | error {command.rmse_px:.2f} px",
                (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (30, 30, 30),
                2,
            )
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
            if step % 20 == 0:
                print(f"IBVS step {step}: {command.rmse_px:.2f} px", flush=True)
            if command.status == "reached":
                break
            # Static target, moving camera: point coordinates change by inverse camera motion.
            pose = inverse(se3_exp(command.velocity_camera, 0.1)) @ pose
        for packet in stream.encode():
            container.mux(packet)
    (output / "ibvs-history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    focal_relative_error = np.abs(camera_matrix.diagonal()[:2] / true_camera_matrix.diagonal()[:2] - 1)
    report = {
        "data_source": "synthetic",
        "physical_hardware_used": False,
        "calibration_images": intrinsics["valid_images"],
        "calibration_rms_px": intrinsics["rms_px"],
        "focal_length_relative_errors": focal_relative_error.tolist(),
        "hand_eye_translation_error_m": translation_error,
        "hand_eye_rotation_error_deg": rotation_error,
        "board_origin_absolute_axis_errors_m": board_origin_error.tolist(),
        "ibvs_initial_rmse_px": history[0]["rmse_px"],
        "ibvs_final_rmse_px": history[-1]["rmse_px"],
        "ibvs_frames": len(history),
        "ibvs_status": history[-1]["status"],
    }
    report["passed"] = bool(
        intrinsics["rms_px"] < 1.0
        and max(focal_relative_error) < 0.05
        and translation_error < 0.01
        and np.all(board_origin_error < 0.05)
        and history[-1]["status"] == "reached"
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("Offline checks did not all pass; see report.json")
    return report
