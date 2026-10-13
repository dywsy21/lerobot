# Fake SDK methods preserve the SDK's public camelCase interface.
# ruff: noqa: N802, N815
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from visual_servo_week6.scs215_bus import Calibration, Scs215Bus

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("drive_mode", [0, 1])
@pytest.mark.parametrize("range_min,range_max", [(100, 900), (800, 1300)])
def test_calibrated_arc_mapping_and_drive_mode_match_fork_contract(drive_mode, range_min, range_max):
    calibration = Calibration(1, drive_mode, 0, range_min, range_max)
    for value in np.linspace(0, 1, 17):
        raw = calibration.raw(value)
        assert 0 <= raw <= 1023
        assert abs(calibration.normalize(raw) - value) <= 1 / (range_max - range_min) + 1e-12
    with pytest.raises(ValueError, match="measured"):
        calibration.normalize(700 if range_min == 800 else 950)
    with pytest.raises(ValueError):
        calibration.raw(1.1)


def test_sdk_writes_only_after_validation_and_rejects_seam_transition(monkeypatch):
    import scservo_sdk

    events = []
    positions = {1: 900, 2: 200}

    class FakePort:
        def __init__(self, _name):
            self.ser = SimpleNamespace(write_timeout=None)
            self.is_open = False

        def openPort(self):
            self.is_open = True
            return True

        def setBaudRate(self, _baud):
            return True

        def setPacketTimeoutMillis(self, _timeout):
            pass

        def closePort(self):
            self.is_open = False

    class FakePacket:
        def __init__(self, protocol):
            assert protocol == 1

        def ping(self, _port, _id):
            return 1315, 0, 0

        def read2ByteTxRx(self, _port, id_, address):
            return ({56: positions[id_], 9: 0, 11: 1023}[address], 0, 0)

        def write2ByteTxRx(self, _port, id_, address, value):
            events.append((id_, address, value))
            return 0, 0

        write1ByteTxRx = write2ByteTxRx

    monkeypatch.setattr(scservo_sdk, "PortHandler", FakePort)
    monkeypatch.setattr(scservo_sdk, "PacketHandler", FakePacket)
    bus = Scs215Bus(
        "FAKE",
        {"a": 1, "b": 2},
        {"a": Calibration(1, 0, 0, 800, 1300), "b": Calibration(2, 0, 0, 100, 900)},
        ROOT.parents[1],
    )
    bus.connect()
    bus.set_baudrate(1000000)
    assert bus.ping("a") == 1315
    assert events == []
    with pytest.raises(ValueError, match="seam"):
        bus.sync_write("Goal_Position", {"a": 0.7, "b": 0.25})
    assert events == []
    bus.sync_write("Goal_Position", {"a": 0.2, "b": 0.125})
    assert events == [(1, 42, 900), (2, 42, 200)]
    bus.disconnect()


def test_hardware_worker_loads_sdk_without_torch_or_opening_a_port():
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "hardware_worker.py"), "--probe"],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    report = json.loads(result.stdout)
    assert report["sdk_loaded"]
    assert not report["torch_loaded"]
    assert not report["hardware_opened"]
