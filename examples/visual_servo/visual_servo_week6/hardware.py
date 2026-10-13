"""Explicit hardware adapter; connecting reads registers and never enables torque."""

import json
import time
from pathlib import Path

import numpy as np
import yaml

from .geometry import inverse, rotation_error_rad, validate_transform


def read_hand_eye(path, model, intrinsics):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if data.get("translation_unit") != "metre" or data.get("end_frame") != model.config["end_frame"]:
        raise ValueError("Hand-eye unit or end_frame does not match robot model")
    if data.get("base_frame") != model.portable.base_frame:
        raise ValueError("Hand-eye base_frame does not match URDF root")
    if data.get("robot_model_sha256") != model.fingerprint:
        raise ValueError("Robot model or joint calibration differs from the hand-eye experiment")
    if data.get("data_source") != "real_usb_camera" or not data.get("validated_for_servo"):
        raise ValueError("需要真实且通过独立验证的手眼结果；仿真结果不能用于实物运动")
    if data.get("intrinsics_sha256") != intrinsics["source_sha256"]:
        raise ValueError("Hand-eye result was computed with different camera intrinsics")
    return validate_transform(data["T_end_from_camera"])


class FeetechAdapter:
    def __init__(self, model, repository):
        self.model, self.repository = model, Path(repository)
        self.bus = None
        self.armed = False
        self.observed_at = None
        self.pending_delta = np.zeros(len(model.names))
        self.last_motion_at = time.monotonic()

    def connect(self):
        # Keep optional hardware dependencies out of offline/simulation imports.
        from .scs215_bus import Calibration, Scs215Bus

        config = self.model.config
        path = config.get("motor_calibration_path")
        if not path or not (self.repository / path).is_file():
            raise ValueError("填写自己的 motor_calibration_path；不能使用网页作者的舵机标定")
        records = json.loads((self.repository / path).read_text(encoding="utf-8"))
        motor_model = config.get("motor_model", "scs215")
        if motor_model != "scs215" or config["observation_units"] != "normalized_0_1":
            raise ValueError("SCS215 adapter requires normalized_0_1 and measured degree mapping")
        ids = config.get("motor_ids", {})
        if set(ids) != set(self.model.names) or len(set(ids.values())) != len(ids):
            raise ValueError("motor_ids must map each active URDF joint to a distinct motor ID")
        calibration = {name: Calibration(**records[name]) for name in self.model.names}
        for record in calibration.values():
            record.segments()
        if any(calibration[name].id != ids[name] for name in self.model.names):
            raise ValueError("Motor calibration IDs do not match configured motor IDs")
        bus = Scs215Bus(config["port"], ids, calibration, self.repository)
        self.bus = bus
        try:
            bus.connect(handshake=False)
            bus.set_baudrate(config.get("baudrate", 1_000_000))
            bus.set_timeout(100)
            for name in self.model.names:
                if bus.ping(name) != bus.model_number_table[motor_model]:
                    raise ValueError(f"Motor ID/model mismatch: {name}")
            self.observe()
        except Exception:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
            self.bus = None
            raise

    def observe(self):
        if self.bus is None:
            raise RuntimeError("机械臂尚未连接")
        started = time.monotonic()
        observation = {
            f"{name}.pos": value
            for name, value in self.bus.sync_read("Present_Position", num_retry=0).items()
        }
        self.model.angles(observation)
        self.observed_at = started
        return observation, started

    def arm(self):
        observation, _ = self.observe()
        # Set the present position as goal before enabling torque; exclude gripper.
        self.bus.sync_write("Goal_Position", self.model.action(self.model.angles(observation)))
        try:
            self.bus.enable_torque(self.model.names)
            self.armed = True
            self.last_motion_at = time.monotonic()
        except Exception:
            self.bus.disable_torque(self.model.names)
            raise

    def teach(self):
        """Operator-supported freehand positioning for task 2."""
        if self.bus is None:
            raise RuntimeError("机械臂尚未连接")
        self.bus.disable_torque(self.model.names)
        self.armed = False
        self.pending_delta[:] = 0

    def move(
        self,
        q,
        observed_at,
        maximum_step_rad,
        maximum_age_s,
        camera_captured_at,
        t_end_from_camera=None,
        max_linear_speed=0.01,
        max_angular_speed=0.1,
    ):
        if not self.armed:
            raise RuntimeError("尚未启用实物运动")
        if not np.isfinite(observed_at) or not 0 <= time.monotonic() - observed_at <= maximum_age_s:
            raise ValueError("Robot feedback is stale")
        observation, sampled = self.observe()
        current = self.model.angles(observation)
        if time.monotonic() - sampled > maximum_age_s:
            raise ValueError("Robot feedback is stale")
        if np.max(np.abs(np.asarray(q) - current)) > maximum_step_rad + 1e-9:
            raise ValueError("Proposed step exceeds fresh-feedback motion bound")
        self.pending_delta = np.clip(
            self.pending_delta + np.asarray(q) - current, -maximum_step_rad, maximum_step_rad
        )
        action = self.model.action(current + self.pending_delta)
        quantized = {
            f"{name}.pos": self.bus.calibration[name].normalize(self.bus.calibration[name].raw(value))
            for name, value in action.items()
        }
        actual_q = self.model.angles(quantized)
        if np.max(np.abs(actual_q - current)) > maximum_step_rad + 1e-9:
            raise ValueError(
                "Configured joint step is smaller than encoder quantization; increase it after measuring the joint"
            )
        if np.max(np.abs(actual_q - current)) < 1e-10:
            return False
        action = self.model.action(actual_q)
        mount = np.eye(4) if t_end_from_camera is None else validate_transform(t_end_from_camera)
        current_camera = self.model.fk_radians(current) @ mount
        actual_camera = self.model.fk_radians(actual_q) @ mount
        relative = inverse(current_camera) @ actual_camera
        minimum_interval = max(
            float(np.linalg.norm(relative[:3, 3])) / max_linear_speed,
            rotation_error_rad(relative[:3, :3], np.eye(3)) / max_angular_speed,
        )
        if time.monotonic() - self.last_motion_at < minimum_interval:
            return False
        # SCS215 seam bridging changes modes and can block. Keep experiments in
        # one raw position interval until that separate hardware behavior is tested.
        if self.model.config.get("motor_model", "scs215") == "scs215":
            raw = self.bus.sync_read("Present_Position", normalize=False)
            goals = self.bus._unnormalize({self.bus.motors[n].id: v for n, v in action.items()})
            if any(abs(goals[self.bus.motors[n].id] % 1024 - raw[n] % 1024) > 512 for n in action):
                raise ValueError("SCS215 raw seam crossing is excluded from this experiment")
        if time.monotonic() - camera_captured_at > maximum_age_s:
            raise ValueError("Camera frame is stale before transmission")
        self.bus.sync_write("Goal_Position", action)
        self.pending_delta -= actual_q - current
        self.last_motion_at = time.monotonic()
        return True

    def hold(self):
        self.pending_delta[:] = 0
        if self.bus is not None and self.armed:
            observation, _ = self.observe()
            self.bus.sync_write("Goal_Position", self.model.action(self.model.angles(observation)))

    def close(self):
        if self.bus is not None:
            try:
                self.hold()
            finally:
                self.bus.disconnect(disable_torque=False)
                self.bus = None
                self.armed = False
