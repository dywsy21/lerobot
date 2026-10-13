"""Shared task 2/3 station; simulation, camera-only and explicitly enabled robot."""

import json
import threading
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from .frame_source import BrokerSource, CameraSource
from .geometry import camera_velocity_to_end_target, rotation_error_rad
from .hand_eye_workflow import (
    append_pair,
    create_session,
    estimate_board_pose,
    read_calibration,
    session_status,
    solve_session,
)
from .hardware import read_hand_eye
from .image_io import write_image
from .robot_model import RobotModel
from .robot_process import RobotProcess
from .task3_core import ServoSettings, VisualServo, camera_body_jacobian, load_intrinsics, propose_joint_step
from .task3_simulation import MarkerScene, annotate


def load_configuration(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    root = path.parent.parent
    for key in ("intrinsics", "robot_model", "hand_eye", "task2_directory", "log_directory"):
        config[key] = str((root / config[key]).resolve())
    marker = config.get("marker", {})
    settings = dict(config.get("controller", {}))
    allowed = {field.name for field in fields(ServoSettings)}
    if set(settings) - allowed:
        raise ValueError(f"Unknown controller parameters: {set(settings) - allowed}")
    settings.update(
        dictionary=marker.get("dictionary") or "DICT_4X4_50",
        marker_id=0 if marker.get("id") is None else marker["id"],
        side_length_m=marker.get("side_length_m") or 0.06,
        marker_size_confirmed=bool(
            marker.get("confirmed")
            and marker.get("side_length_m")
            and marker.get("dictionary")
            and marker.get("id") is not None
        ),
    )
    config["settings"] = ServoSettings(**settings)
    config["root"] = str(root)
    return config


def preflight(config_path):
    config = load_configuration(config_path)
    calibration = read_calibration(config["intrinsics"])
    problems = []
    model = None
    raw_model = json.loads(Path(config["robot_model"]).read_text(encoding="utf-8-sig"))
    names = raw_model.get("joint_names", [])
    mapping = raw_model.get("normalized_joint_calibration", {})
    if raw_model.get("observation_units") == "normalized_0_1" and any(
        mapping.get(name, {}).get("q_at_0_deg") is None or mapping.get(name, {}).get("q_at_1_deg") is None
        for name in names
    ):
        problems.append("尚未填写各关节的实测角度端点 normalized_joint_calibration")
    if any(raw_model.get("joint_limits_deg", {}).get(name) is None for name in names):
        problems.append("尚未填写各关节的实测安全范围 joint_limits_deg")
    motor_file = raw_model.get("motor_calibration_path")
    if not motor_file or not (Path(config["root"]).parents[1] / motor_file).is_file():
        problems.append("缺少自己的舵机标定 motor_calibration_path")
    if not Path(config["hand_eye"]).is_file():
        problems.append("真实手眼矩阵待采集：至少12组训练＋3组独立验证，另需实测棋盘原点")
    try:
        model = RobotModel(config["robot_model"], Path(config["root"]).parents[1])
    except (ValueError, OSError, KeyError, ImportError) as error:
        problems.append(str(error))
    if model is not None:
        try:
            read_hand_eye(config["hand_eye"], model, calibration)
        except (OSError, ValueError, KeyError) as error:
            problems.append(str(error))
    if not config["settings"].marker_size_confirmed:
        problems.append("填写真实 ArUco 字典、ID、黑色外边框边长，并设置 marker.confirmed: true")
    return {
        "software_ready": True,
        "offline_simulation_ready": True,
        "hardware_configuration_ready": not problems,
        "hardware_connected": False,
        "camera_size": calibration["image_size"],
        "board": calibration["board_inner_corners"],
        "square_size_m": calibration["square_size_m"],
        "missing_hardware_configuration": problems,
        "hardware_commands_sent": False,
    }


class ExperimentStation:
    def __init__(
        self,
        config_path,
        source="simulation",
        camera_index=2,
        broker_url="http://127.0.0.1:8766",
        allow_hardware=False,
    ):
        self.config_path = Path(config_path)
        self.config = load_configuration(config_path)
        self.root = Path(self.config["root"])
        self.repository = self.root.parents[1]
        matrix, distortion, size, camera_calibration = load_intrinsics(self.config["intrinsics"])
        self.servo = VisualServo(matrix, distortion, size, self.config["settings"])
        self.source_kind = source
        self.scene = MarkerScene(self.servo) if source == "simulation" else None
        if source == "camera":
            self.source = CameraSource(camera_index, size, camera_calibration.get("camera_name"))
        elif source == "broker":
            self.source = BrokerSource(broker_url, size)
        elif source == "simulation":
            self.source = None
        else:
            raise ValueError("Unknown camera source")
        self.allow_hardware = allow_hardware and source == "camera"
        self.lock, self.stop_event = threading.RLock(), threading.Event()
        self.active, self.execute, self.occluded = False, False, False
        self.model = self.robot = self.hand_eye = self.rule_target = None
        self.latest_image = self.latest_advice = self.jpeg = None
        self.message = "软件已就绪；仿真标记默认 60 mm，实物标记信息待确认。"
        self.sequence, self.command_count, self.lease_until = 0, 0, 0.0
        self.board_view, self.reverse_corners, self.board_pose = False, False, None
        self.log_path = Path(self.config["log_directory"]) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.log_path.mkdir(parents=True, exist_ok=False)
        self.log = (self.log_path / "servo.jsonl").open("a", encoding="utf-8")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _pause(self):
        self.active, self.execute, self.rule_target = False, False, None
        if self.robot is not None:
            self.robot.hold()

    def _proposal(self, advice, observation):
        q = self.model.angles(observation)
        if self.servo.settings.mode == "rule" and advice["status"] == "rule_step":
            if self.rule_target is None:
                # A 1 cm camera request is solved once, then split into bounded joint steps.
                step = np.asarray(advice["step_camera_m"])
                target = camera_velocity_to_end_target(
                    self.model.fk_radians(q), self.hand_eye, np.r_[step, np.zeros(3)], 1.0
                )
                self.rule_target = np.deg2rad(
                    self.model.portable.inverse_kinematics(
                        np.rad2deg(q), target, np.rad2deg(self.model.limits_rad)
                    )
                )
            delta = self.rule_target - q
            delta /= max(1.0, float(np.max(np.abs(delta))) / self.servo.settings.maximum_joint_step_rad)
            achievable = camera_body_jacobian(self.model.fk_radians, q, self.hand_eye) @ (
                delta / self.servo.settings.dt_s
            )
            delta /= max(
                1.0,
                np.linalg.norm(achievable[:3]) / self.servo.settings.max_linear_speed_m_s,
                np.linalg.norm(achievable[3:]) / self.servo.settings.max_angular_speed_rad_s,
            )
            if np.max(np.abs(delta)) < 1e-4:
                self.rule_target = None
            return {"status": "proposal", "q_next_rad": (q + delta).tolist(), "hardware_commands_sent": False}
        return propose_joint_step(
            advice, self.servo, self.model.fk_radians, q, self.model.limits_rad, self.hand_eye
        )

    def _loop(self):
        while not self.stop_event.is_set():
            began = time.monotonic()
            with self.lock:
                try:
                    image, captured = (
                        (self.scene.image(self.occluded), began) if self.scene else self.source.read()
                    )
                    self.latest_image = image.copy()
                    advice = self.servo.evaluate(
                        image,
                        captured_at=captured,
                        active=self.active,
                        allow_resize=self.source_kind == "broker",
                    )
                    if self.execute and time.monotonic() > self.lease_until:
                        self._pause()
                        advice = self.servo.stopped("browser_timeout", "控制页面心跳超时，已暂停")
                    if self.active and self.scene:
                        if self.servo.settings.mode == "rule":
                            velocity = np.asarray(advice["velocity_camera"])
                            velocity /= max(
                                1.0, np.linalg.norm(velocity[:3]) / self.servo.settings.max_linear_speed_m_s
                            )
                            self.scene.apply(velocity)
                        else:
                            self.scene.apply(advice["velocity_camera"])
                    elif self.execute:
                        if advice["status"] in {"tracking", "rule_step"}:
                            observation, sampled = self.robot.observe()
                            proposal = self._proposal(advice, observation)
                            if time.monotonic() - captured > self.servo.settings.maximum_frame_age_s:
                                raise ValueError("Camera frame became stale while computing robot command")
                            if proposal["status"] == "proposal":
                                sent = self.robot.move(
                                    proposal["q_next_rad"],
                                    sampled,
                                    self.servo.settings.maximum_joint_step_rad,
                                    self.servo.settings.maximum_frame_age_s,
                                    captured,
                                    self.hand_eye.tolist(),
                                    self.servo.settings.max_linear_speed_m_s,
                                    self.servo.settings.max_angular_speed_rad_s,
                                )
                                self.command_count += int(sent)
                                advice["hardware_commands_sent"] = bool(sent)
                            else:
                                self._pause()
                            advice["joint_proposal"] = proposal
                        else:
                            self._pause()
                    self.latest_advice = advice
                    frame = annotate(image, advice, self.servo.size, "SIMULATION" if self.scene else "CAMERA")
                    if self.board_view:
                        try:
                            self.board_pose, frame = estimate_board_pose(
                                image, read_calibration(self.config["intrinsics"]), self.reverse_corners
                            )
                        except ValueError as error:
                            self.board_pose = {"error": str(error)}
                    self.jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()
                    self.sequence += 1
                    self.log.write(
                        json.dumps(
                            {
                                "sequence": self.sequence,
                                "monotonic_s": began,
                                "source": self.source_kind,
                                **advice,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    self.log.flush()
                except Exception as error:
                    try:
                        self._pause()
                    except Exception as hold_error:
                        self.message = f"{error}; 保持命令失败: {hold_error}"
                    else:
                        self.message = str(error)
                    self.latest_advice = self.servo.stopped("error", self.message)
            self.stop_event.wait(max(0.01, self.servo.settings.dt_s - (time.monotonic() - began)))

    def status(self):
        with self.lock:
            task2 = (
                session_status(self.config["task2_directory"])
                if (Path(self.config["task2_directory"]) / "poses.json").exists()
                else {"training_pairs": 0, "validation_pairs": 0, "ready_to_solve": False}
            )
            return {
                "source": self.source_kind,
                "active": self.active,
                "executing": self.execute,
                "robot_connected": self.robot is not None,
                "robot_armed": bool(self.robot and self.robot.armed),
                "hardware_enabled_by_launch": self.allow_hardware,
                "hardware_command_count": self.command_count,
                "sequence": self.sequence,
                "image_size": list(self.servo.size),
                "message": self.message,
                "controller_mode": self.servo.settings.mode,
                "advice": self.latest_advice,
                "board_pose": self.board_pose,
                "task2": task2,
                "log_directory": str(self.log_path),
                "marker_size_confirmed": self.servo.settings.marker_size_confirmed,
            }

    def control(self, command, payload=None):
        payload = payload or {}
        with self.lock:
            if command == "heartbeat":
                self.lease_until = time.monotonic() + 1.5
            elif command == "pause":
                self._pause()
                self.message = "已暂停，虚拟相机保持当前位置。" if self.scene else "已暂停新的运动指令。"
            elif command == "start":
                self._pause()
                self.active, self.execute = True, False
                self.message = "仿真推进中" if self.scene else "相机算法观察中；尚未发送实物运动指令。"
            elif command == "reset":
                if not self.scene:
                    raise ValueError("Reset is only available in simulation")
                self._pause()
                self.scene.reset()
                self.occluded = False
            elif command == "occlude":
                if not self.scene:
                    raise ValueError("Occlusion button is only available in simulation")
                self.occluded = not self.occluded
            elif command == "goal":
                if self.latest_advice is None or self.latest_advice["corners_px"] is None:
                    raise ValueError("先将唯一的目标 ArUco 完整放入画面")
                self._pause()
                self.servo.set_goal(self.latest_advice["corners_px"])
                self.message = "当前四角点已设为目标；像素坐标已保存。"
                (self.log_path / "goal.json").write_text(
                    json.dumps(self.servo.goal.tolist()), encoding="utf-8"
                )
            elif command == "connect":
                if not self.allow_hardware:
                    raise ValueError("需以 --source camera --allow-hardware 启动后才能连接机械臂")
                if self.robot is not None:
                    raise ValueError("机械臂已经连接")
                model = RobotModel(self.config["robot_model"], self.repository)
                robot = RobotProcess(model, self.repository, self.log_path)
                robot.connect()
                self.model, self.robot = model, robot
                self.message = "已读取关节状态；连接操作未启用力矩。可开始任务2静止采集。"
            elif command == "execute":
                if not self.allow_hardware or self.robot is None:
                    raise ValueError("先连接机械臂，并使用原生相机采集")
                if not self.servo.settings.marker_size_confirmed:
                    raise ValueError("先在配置中确认真实标记字典、ID 和实测边长")
                calibration = read_calibration(self.config["intrinsics"])
                self.hand_eye = read_hand_eye(self.config["hand_eye"], self.model, calibration)
                if self.latest_advice is None or self.latest_advice.get("corners_px") is None:
                    raise ValueError("目标标记不可见")
                self.robot.arm()
                self.active, self.execute = True, True
                self.lease_until = time.monotonic() + 1.5
                self.message = "实物伺服运行；页面心跳、相机新鲜度、目标检测和步长限制生效。"
            elif command == "teach":
                if self.robot is None or not payload.get("supported"):
                    raise ValueError("先连接并支撑机械臂，再确认释放关节力矩")
                self._pause()
                self.robot.teach()
                self.message = "已释放活动关节力矩；请支撑机械臂，手动调整并停稳后采集。"
            elif command in {"view-board", "view-marker"}:
                self._pause()
                self.board_view = command == "view-board"
                self.reverse_corners = bool(payload.get("reverse_corners"))
            elif command == "capture":
                self._capture(payload)
            elif command == "solve":
                self._pause()
                measurement = payload.get("board_origin_m")
                result = solve_session(
                    self.config["task2_directory"], Path(self.config["hand_eye"]).parent, measurement
                )
                self.message = f"已保存手眼矩阵；可用于实物伺服验证状态：{result['validated_for_servo']}"
            else:
                raise ValueError("Unknown station command")
        return self.status()

    def _capture(self, payload):
        if self.source_kind != "camera" or self.robot is None:
            raise ValueError("真实配对采集需要原生相机与实时机械臂反馈")
        self._pause()
        directory = Path(self.config["task2_directory"])
        if not (directory / "poses.json").exists():
            create_session(directory, self.config["intrinsics"])
        before, sampled_before = self.robot.observe()
        time.sleep(0.15)
        # Fresh image read between two real observations, never reuses preview JPEG.
        image, captured = self.source.read()
        after, sampled_after = self.robot.observe()
        first, second = self.model.pose(before), self.model.pose(after)
        a, b = np.asarray(first["T_base_from_end"]), np.asarray(second["T_base_from_end"])
        q_before, q_after = self.model.angles(before), self.model.angles(after)
        if (
            np.max(np.abs(q_after - q_before)) > np.deg2rad(0.5)
            or np.linalg.norm(a[:3, 3] - b[:3, 3]) > 0.001
            or rotation_error_rad(a[:3, :3], b[:3, :3]) > np.deg2rad(0.5)
        ):
            raise ValueError("机械臂拍照前后仍有运动；停稳后重新采集")
        if time.monotonic() - captured > self.servo.settings.maximum_frame_age_s:
            raise ValueError("配对图像过旧，请检查采集速度")
        staging = directory / "staging" / f"{time.time_ns()}.png"
        write_image(staging, image)
        pair = append_pair(
            directory,
            staging,
            second,
            validation=bool(payload.get("validation")),
            reverse_corners=bool(payload.get("reverse_corners")),
            capture_metadata={
                "automatic_live_robot_feedback": True,
                "robot_before_monotonic_s": sampled_before,
                "camera_monotonic_s": captured,
                "robot_after_monotonic_s": sampled_after,
                "maximum_joint_change_deg": float(np.rad2deg(np.max(np.abs(q_after - q_before)))),
                "stationary_translation_tolerance_m": 0.001,
            },
        )
        self.message = f"已保存配对 {pair['id']}，棋盘重投影误差 {pair['board_rmse_px']:.3f} px"

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        with self.lock:
            try:
                if self.robot:
                    self.robot.close()
            finally:
                if self.source:
                    self.source.close()
                self.log.close()
