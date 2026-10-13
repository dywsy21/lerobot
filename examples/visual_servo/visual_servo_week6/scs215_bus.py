"""Small SCS215 position-mode bus using the fork's tables and Feetech SDK.

No PyTorch import. Mirrors the fork's RANGE_0_1 degree contract, but rejects
out-of-range feedback and seam transitions rather than clamping or using PWM.
"""

import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


@dataclass
class Calibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int
    range_segments: list | None = None

    def segments(self):
        start, end = self.range_min, self.range_max
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < 1024
            or not start < end < start + 1024
        ):
            raise ValueError("Invalid SCS215 calibrated continuous range")
        expected = [[start, end]] if end < 1024 else [[start, 1023], [0, end - 1024]]
        if self.range_segments is not None and [list(s) for s in self.range_segments] != expected:
            raise ValueError("SCS215 range_segments differ from continuous calibrated range")
        if self.drive_mode not in {0, 1}:
            raise ValueError("drive_mode must be 0 or 1")
        return expected

    def normalize(self, raw):
        offset = 0
        for start, end in self.segments():
            if start <= raw <= end:
                value = (offset + raw - start) / (self.range_max - self.range_min)
                return 1 - value if self.drive_mode else value
            offset += end - start + 1
        raise ValueError("Motor feedback is outside its measured raw range")

    def raw(self, value):
        self.segments()
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("SCS215 command must be inside normalized range")
        value = 1 - value if self.drive_mode else value
        return int(np.rint(value * (self.range_max - self.range_min) + self.range_min)) % 1024


class Scs215Bus:
    model_number_table = {"scs215": 1315}

    def __init__(self, port, ids, calibration, repository):
        import scservo_sdk

        tables = runpy.run_path(str(Path(repository) / "src/lerobot/motors/feetech/tables.py"))
        self.table = tables["MODEL_CONTROL_TABLE"]["scs215"]
        self.sdk, self.calibration = scservo_sdk, calibration
        self.motors = {name: SimpleNamespace(id=id_) for name, id_ in ids.items()}
        self.port = scservo_sdk.PortHandler(port)
        self.packet = scservo_sdk.PacketHandler(tables["MODEL_PROTOCOL"]["scs215"])
        self.is_connected = False
        self.timeout_ms = 100

    def connect(self, handshake=False):
        if handshake:
            raise ValueError("Explicit per-ID model verification required")
        if not self.port.openPort():
            raise OSError("Cannot open SCS215 serial port")
        self.is_connected = True
        self.set_timeout(self.timeout_ms)

    def set_baudrate(self, value):
        if not self.port.setBaudRate(value):
            raise ValueError("Cannot set host serial baudrate")

    def set_timeout(self, value):
        self.timeout_ms = value
        # SDK calls setPacketTimeout internally; bound every read, not just the
        # initial timeout. Bound pyserial writes as well.
        self.port.setPacketTimeout = lambda _length: self.port.setPacketTimeoutMillis(value)
        if self.port.is_open:
            self.port.ser.write_timeout = value / 1000

    def _check(self, result, error):
        if result != self.sdk.COMM_SUCCESS or error:
            raise OSError(f"SCS215 packet failure: communication={result}, device_status={error}")

    def ping(self, name):
        value, result, error = self.packet.ping(self.port, self.motors[name].id)
        self._check(result, error)
        # Check position mode without changing EEPROM.
        if self.read("Max_Position_Limit", name) <= self.read("Min_Position_Limit", name):
            raise ValueError(f"{name} is not in native position mode")
        return value

    def read(self, register, name):
        address, length = self.table[register]
        reader = self.packet.read1ByteTxRx if length == 1 else self.packet.read2ByteTxRx
        value, result, error = reader(self.port, self.motors[name].id, address)
        self._check(result, error)
        return value

    def sync_read(self, register, normalize=True, num_retry=0):
        if num_retry:
            raise ValueError("Automatic retries are disabled during paired capture")
        values = {name: self.read(register, name) for name in self.motors}
        if register == "Present_Position" and normalize:
            return {name: self.calibration[name].normalize(raw) for name, raw in values.items()}
        return values

    def _unnormalize(self, values):
        by_id = {motor.id: name for name, motor in self.motors.items()}
        return {id_: self.calibration[by_id[id_]].raw(value) for id_, value in values.items()}

    def _write(self, register, name, value):
        if register not in {"Goal_Position", "Running_Time", "Goal_Velocity", "Torque_Enable", "Lock"}:
            raise ValueError("Course adapter writes position-mode SRAM registers only")
        address, length = self.table[register]
        writer = self.packet.write1ByteTxRx if length == 1 else self.packet.write2ByteTxRx
        result, error = writer(self.port, self.motors[name].id, address, value)
        self._check(result, error)

    def sync_write(self, register, values):
        if register != "Goal_Position":
            raise ValueError("Only position goals are accepted")
        goals = {name: self.calibration[name].raw(value) for name, value in values.items()}
        present = {name: self.read("Present_Position", name) for name in values}
        for name, goal in goals.items():
            segments = self.calibration[name].segments()
            if not any(start <= present[name] <= end and start <= goal <= end for start, end in segments):
                raise ValueError("SCS215 calibrated seam crossing is excluded")
        for name, goal in goals.items():
            self._write(register, name, goal)

    def enable_torque(self, names):
        for name in names:
            self._write("Running_Time", name, 0)
            self._write("Goal_Velocity", name, 60)
            self._write("Torque_Enable", name, 1)

    def disable_torque(self, names):
        for name in names:
            self._write("Torque_Enable", name, 0)

    def disconnect(self, disable_torque=False):
        try:
            if disable_torque:
                self.disable_torque(list(self.motors))
        finally:
            self.port.closePort()
            self.is_connected = False
