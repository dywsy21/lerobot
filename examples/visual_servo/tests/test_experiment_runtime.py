import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from visual_servo_week6.experiment_runtime import ExperimentStation, preflight
from visual_servo_week6.hardware import FeetechAdapter, read_hand_eye

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def station(tmp_path):
    configuration = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    configuration.update(
        intrinsics=str(ROOT / "results/task1/intrinsics.yaml"),
        log_directory=str(tmp_path / "logs"),
        task2_directory=str(tmp_path / "pairs"),
    )
    path = tmp_path / "configs/experiment.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(configuration))
    value = ExperimentStation(path)
    try:
        yield value
    finally:
        value.close()


def test_simulation_cannot_connect_or_execute_hardware_even_if_launch_flag_set(station):
    assert not station.allow_hardware
    for name in ["connect", "execute", "capture"]:
        with pytest.raises(ValueError):
            station.control(name)
    assert station.status()["hardware_command_count"] == 0


def test_pause_occlusion_and_reset_keep_physical_command_count_zero(station):
    station.control("start")
    station.control("occlude")
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = station.status()
        if status["advice"] and status["advice"]["status"] == "target_lost":
            break
        time.sleep(0.03)
    assert status["advice"]["status"] == "target_lost"
    assert not any(status["advice"]["velocity_camera"])
    station.control("pause")
    assert not station.status()["active"]
    station.control("reset")
    assert not station.occluded and station.scene.steps == 0
    assert station.command_count == 0


def test_default_preflight_reports_unconfigured_hardware_without_connecting():
    report = preflight(ROOT / "configs/experiment.yaml")
    assert report["offline_simulation_ready"]
    assert not report["hardware_configuration_ready"]
    assert not report["hardware_connected"]
    assert not report["hardware_commands_sent"]


def test_hand_eye_gate_rejects_synthetic_units_frames_and_different_intrinsics(tmp_path):
    model = SimpleNamespace(config={"end_frame": "wrist"}, portable=SimpleNamespace(base_frame="base"))
    model.fingerprint = "fixture"
    data = {
        "translation_unit": "metre",
        "end_frame": "wrist",
        "base_frame": "base",
        "data_source": "synthetic",
        "validated_for_servo": False,
        "intrinsics_sha256": "known",
        "robot_model_sha256": "fixture",
        "T_end_from_camera": np.eye(4).tolist(),
    }
    path = tmp_path / "hand_eye.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="真实"):
        read_hand_eye(path, model, {"source_sha256": "known"})
    data.update(data_source="real_usb_camera", validated_for_servo=True)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="intrinsics"):
        read_hand_eye(path, model, {"source_sha256": "different"})
    np.testing.assert_allclose(read_hand_eye(path, model, {"source_sha256": "known"}), np.eye(4))
    data["end_frame"] = "gripper"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="end_frame"):
        read_hand_eye(path, model, {"source_sha256": "known"})


def test_motor_adapter_connect_only_reads_and_arm_sets_goal_before_enabling_torque(tmp_path, monkeypatch):
    import visual_servo_week6.scs215_bus
    from visual_servo_week6.scs215_bus import Calibration

    events = []

    class FakeBus:
        model_number_table = {"scs215": 1315}
        is_connected = False

        def __init__(self, _port, ids, _calibration, _repository):
            self.motors = {name: SimpleNamespace(id=id_) for name, id_ in ids.items()}

        def connect(self, handshake):
            assert handshake is False
            self.is_connected = True

        def set_baudrate(self, _value):
            pass

        def set_timeout(self, _value):
            pass

        def ping(self, _name):
            return 1315

        def sync_read(self, _register, **_kwargs):
            return {"pan": 0.5}

        def sync_write(self, _register, value):
            events.append(("goal", value))

        def enable_torque(self, _names):
            events.append(("torque", True))

        def disable_torque(self, _names):
            events.append(("torque", False))

        def disconnect(self, disable_torque):
            assert not disable_torque
            self.is_connected = False

    from dataclasses import asdict

    (tmp_path / "calibration.json").write_text(json.dumps({"pan": asdict(Calibration(1, 0, 0, 100, 900))}))
    model = SimpleNamespace(
        names=["pan"],
        config={
            "motor_calibration_path": "calibration.json",
            "observation_units": "normalized_0_1",
            "motor_ids": {"pan": 1},
            "port": "FAKE",
        },
        angles=lambda _observation: np.array([0.0]),
        action=lambda _q: {"pan": 0.5},
    )
    monkeypatch.setattr(visual_servo_week6.scs215_bus, "Scs215Bus", FakeBus)
    adapter = FeetechAdapter(model, tmp_path)
    adapter.connect()
    assert events == [] and not adapter.armed
    adapter.arm()
    assert [event[0] for event in events] == ["goal", "torque"]
    with pytest.raises(ValueError, match="stale"):
        adapter.move([0], time.monotonic() - 2, 0.005, 0.5, time.monotonic())
    with pytest.raises(ValueError, match="bound"):
        adapter.move([0.02], time.monotonic(), 0.005, 0.5, time.monotonic())
    assert len(events) == 2
    adapter.teach()
    assert not adapter.armed and events[-1] == ("torque", False)
    adapter.close()


def test_quantized_motion_accumulates_small_proposals_and_obeys_camera_rate(tmp_path):
    from visual_servo_week6.geometry import transform
    from visual_servo_week6.scs215_bus import Calibration

    calibration = Calibration(1, 0, 0, 100, 900)
    events = []

    class QuantizedBus:
        motors = {"pan": SimpleNamespace(id=1)}

        def __init__(self):
            self.calibration = {"pan": calibration}

        def sync_read(self, _register, normalize=True, **_kwargs):
            return {"pan": 0.5 if normalize else 500}

        def _unnormalize(self, values):
            return {id_: calibration.raw(value) for id_, value in values.items()}

        def sync_write(self, _register, action):
            events.append(action)

    model = SimpleNamespace(
        names=["pan"],
        config={"motor_model": "scs215"},
        angles=lambda observation: np.array([observation["pan.pos"] * 0.8]),
        action=lambda q: {"pan": float(q[0]) / 0.8},
        fk_radians=lambda q: transform(translation=[0.5 * q[0], 0, 0]),
    )
    adapter = FeetechAdapter(model, tmp_path)
    adapter.bus, adapter.armed = QuantizedBus(), True
    for _ in range(3):
        now = time.monotonic()
        assert not adapter.move([0.4002], now, 0.006, 0.5, now)
    assert not events
    adapter.last_motion_at -= 1
    now = time.monotonic()
    assert adapter.move([0.4002], now, 0.006, 0.5, now)
    assert calibration.raw(events[0]["pan"]) == 501
