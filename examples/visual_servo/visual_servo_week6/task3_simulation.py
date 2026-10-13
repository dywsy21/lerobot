"""Task 3 synthetic scene using the measured camera model; no physical robot."""

import json
from dataclasses import asdict
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import inverse, se3_exp
from .image_io import write_image
from .task3_core import ServoSettings, VisualServo, load_intrinsics, project_marker


class MarkerScene:
    def __init__(self, servo):
        self.servo = servo
        self.reset()
        w, h = servo.size
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        distorted = np.stack([xx, yy], axis=-1).astype(np.float32).reshape(-1, 1, 2)
        ideal = cv2.undistortPoints(
            distorted, servo.camera_matrix, servo.distortion, P=servo.camera_matrix
        ).reshape(h, w, 2)
        self.map_x, self.map_y = ideal[:, :, 0].copy(), ideal[:, :, 1].copy()
        self.texture = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, servo.settings.dictionary)),
            servo.settings.marker_id,
            600,
        )

    def reset(self):
        self.pose = np.eye(4)
        self.pose[:3, :3] = Rotation.from_euler("xyz", [0.12, -0.16, 0.08]).as_matrix()
        self.pose[:3, 3] = [0.065, -0.045, 0.72]
        self.steps = 0

    def image(self, occluded=False):
        w, h = self.servo.size
        if occluded:
            return np.full((h, w, 3), 210, np.uint8)
        ideal_pixels = project_marker(
            self.pose, self.servo.camera_matrix, np.zeros(5), self.servo.settings.side_length_m
        ).astype(np.float32)
        source = np.float32([[0, 0], [599, 0], [599, 599], [0, 599]])
        homography = cv2.getPerspectiveTransform(source, ideal_pixels)
        ideal = cv2.warpPerspective(self.texture, homography, (w, h), borderValue=210)
        distorted = cv2.remap(ideal, self.map_x, self.map_y, cv2.INTER_LINEAR, borderValue=210)
        return cv2.cvtColor(distorted, cv2.COLOR_GRAY2BGR)

    def apply(self, velocity):
        self.pose = inverse(se3_exp(velocity, self.servo.settings.dt_s)) @ self.pose
        self.steps += 1


def annotate(image, advice, size, label):
    output = image.copy()
    scale = np.asarray(output.shape[1::-1]) / np.asarray(size)
    goal = np.asarray(advice["desired_px"]) * scale
    cv2.polylines(output, [goal.round().astype(np.int32)], True, (50, 170, 40), 2)
    if advice["corners_px"] is not None:
        corners = np.asarray(advice["corners_px"]) * scale
        cv2.polylines(output, [corners.round().astype(np.int32)], True, (230, 120, 35), 2)
        for i, point in enumerate(corners):
            cv2.putText(
                output,
                str(i + 1),
                tuple(point.round().astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (240, 50, 20),
                2,
            )
    error = "--" if advice["rmse_px"] is None else f"{advice['rmse_px']:.2f} px"
    cv2.putText(
        output,
        f"{label} | {advice['status']} | error {error}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (50, 50, 50),
        2,
    )
    return output


def run_task3_simulation(intrinsics_file, output, maximum_steps=400):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    camera_matrix, dist, size, calibration = load_intrinsics(intrinsics_file)
    settings = ServoSettings(marker_size_confirmed=False)
    servo = VisualServo(camera_matrix, dist, size, settings)
    scene = MarkerScene(servo)
    history, reached_count = [], 0
    movie = output / "task3_ibvs_simulation.mp4"
    with av.open(str(movie), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height, stream.pix_fmt = *size, "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        for step in range(maximum_steps):
            occluded = 40 <= step < 46
            image = scene.image(occluded)
            advice = servo.evaluate(image)
            history.append(
                {
                    "step": step,
                    "time_s": step * settings.dt_s,
                    "status": advice["status"],
                    "rmse_px": advice["rmse_px"],
                    "velocity_camera": advice["velocity_camera"],
                    "occluded": occluded,
                }
            )
            frame = annotate(image, advice, size, "SIMULATION, NO ROBOT")
            if step == 0:
                write_image(output / "initial.png", frame)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                container.mux(packet)
            if step % 50 == 0:
                print(f"Step {step}: {advice['status']}, error {advice['rmse_px']}", flush=True)
            reached_count = reached_count + 1 if advice["status"] == "reached" else 0
            if reached_count >= 5:
                write_image(output / "final.png", frame)
                break
            if advice["status"] in {"depth_limit", "pose_unreliable", "resolution_mismatch"}:
                raise RuntimeError(f"Simulation stopped: {advice['status']}")
            scene.apply(advice["velocity_camera"])
        for packet in stream.encode():
            container.mux(packet)
    visible = [row for row in history if row["rmse_px"] is not None]
    lost = [row for row in history if row["occluded"]]
    report = {
        "data_source": "synthetic_marker_scene_with_measured_task1_intrinsics",
        "camera_intrinsics_file": str(Path(intrinsics_file).resolve()),
        "camera_calibration_rms_px": calibration.get("rms_px"),
        "physical_hardware_used": False,
        "physical_task3_completed": False,
        "marker_side_m_in_simulation": settings.side_length_m,
        "image_size": list(size),
        "initial_rmse_px": visible[0]["rmse_px"],
        "final_rmse_px": visible[-1]["rmse_px"],
        "steps": len(history),
        "reached": reached_count >= 5,
        "target_loss_zero_velocity": bool(lost) and all(not any(row["velocity_camera"]) for row in lost),
        "maximum_linear_speed_m_s": max(float(np.linalg.norm(row["velocity_camera"][:3])) for row in history),
        "maximum_angular_speed_rad_s": max(
            float(np.linalg.norm(row["velocity_camera"][3:])) for row in history
        ),
        "settings": asdict(settings),
        "video": str(movie),
    }
    report["passed"] = (
        report["reached"]
        and report["target_loss_zero_velocity"]
        and report["final_rmse_px"] <= settings.stop_rmse_px
    )
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("IBVS simulation failed; inspect report.json")
    return report
