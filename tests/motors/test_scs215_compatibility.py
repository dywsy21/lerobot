#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, call, patch

import pytest

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.feetech.tables import (
    MODEL_BAUDRATE_TABLE,
    MODEL_CONTROL_TABLE,
    MODEL_DEGREE_RANGE,
    MODEL_ENCODING_TABLE,
    MODEL_NUMBER_TABLE,
    MODEL_POSITION_WRAPAROUND,
    MODEL_PROTOCOL,
    MODEL_RESOLUTION,
    SCS_SERIES_CONTROL_TABLE,
)
from lerobot.motors.motors_bus import split_circular_range


def make_scs215_bus() -> FeetechMotorsBus:
    motors = {
        "motor_1": Motor(1, "scs215", MotorNormMode.RANGE_M100_100),
        "motor_2": Motor(2, "scs215", MotorNormMode.RANGE_M100_100),
        "motor_3": Motor(3, "scs215", MotorNormMode.RANGE_M100_100),
    }
    return FeetechMotorsBus("", motors, protocol_version=1)


def test_scs215_model_metadata():
    assert MODEL_NUMBER_TABLE["scs215"] == 1315
    assert MODEL_CONTROL_TABLE["scs215"] is SCS_SERIES_CONTROL_TABLE
    assert MODEL_RESOLUTION["scs215"] == 1024
    assert MODEL_DEGREE_RANGE["scs215"] == 300.0
    assert "scs215" in MODEL_POSITION_WRAPAROUND
    assert MODEL_BAUDRATE_TABLE["scs215"][1_000_000] == 0
    assert MODEL_ENCODING_TABLE["scs215"] == {}
    assert MODEL_PROTOCOL["scs215"] == 1


def test_scs215_degree_normalization_uses_300_degree_sensor_range():
    bus = FeetechMotorsBus(
        "",
        {"motor_1": Motor(1, "scs215", MotorNormMode.DEGREES)},
        protocol_version=1,
    )
    bus.calibration = {
        "motor_1": MotorCalibration(
            id=1,
            drive_mode=0,
            homing_offset=0,
            range_min=0,
            range_max=1023,
        )
    }

    assert bus._normalize({1: 0}) == {1: -150.0}
    assert bus._normalize({1: 1023}) == {1: 150.0}
    assert bus._unnormalize({1: -150.0}) == {1: 0}
    assert bus._unnormalize({1: 150.0}) == {1: 1023}


@pytest.mark.parametrize(
    ("samples", "expected_min", "expected_max"),
    [
        ([1000, 1018, 5, 30], 1000, 1054),
        ([30, 5, 1018, 1000], -24, 30),
    ],
)
def test_scs215_range_recording_unwraps_zero_crossing(samples, expected_min, expected_max):
    bus = FeetechMotorsBus(
        "",
        {"motor_1": Motor(1, "scs215", MotorNormMode.RANGE_M100_100)},
        protocol_version=1,
    )
    bus.sync_read = MagicMock(side_effect=[{"motor_1": position} for position in samples])

    with (
        patch("lerobot.motors.motors_bus.enter_pressed", side_effect=[False, False, True]),
        patch("lerobot.motors.motors_bus.time.sleep"),
    ):
        mins, maxes = bus.record_ranges_of_motion(display_values=False)

    assert mins == {"motor_1": expected_min}
    assert maxes == {"motor_1": expected_max}


def test_scs215_zero_one_normalization_is_continuous_across_wrap():
    bus = FeetechMotorsBus(
        "",
        {"motor_1": Motor(1, "scs215", MotorNormMode.RANGE_0_1)},
        protocol_version=1,
    )
    bus.calibration = {
        "motor_1": MotorCalibration(
            id=1,
            drive_mode=0,
            homing_offset=0,
            range_min=1000,
            range_max=1100,
        )
    }

    assert bus._normalize({1: 1000})[1] == pytest.approx(0.0)
    assert bus._normalize({1: 1023})[1] == pytest.approx(0.23)
    assert bus._normalize({1: 0})[1] == pytest.approx(0.24)
    assert bus._normalize({1: 76})[1] == pytest.approx(1.0)
    assert bus._unnormalize({1: 0.0}) == {1: 1000}
    assert bus._unnormalize({1: 0.24}) == {1: 0}
    assert bus._unnormalize({1: 1.0}) == {1: 76}


def test_scs215_zero_one_normalization_handles_reverse_wrap_range():
    bus = FeetechMotorsBus(
        "",
        {"motor_1": Motor(1, "scs215", MotorNormMode.RANGE_0_1)},
        protocol_version=1,
    )
    bus.calibration = {
        "motor_1": MotorCalibration(
            id=1,
            drive_mode=0,
            homing_offset=0,
            range_min=-24,
            range_max=30,
        )
    }

    assert bus._normalize({1: 1000})[1] == pytest.approx(0.0)
    assert bus._normalize({1: 0})[1] == pytest.approx(24 / 54)
    assert bus._normalize({1: 30})[1] == pytest.approx(1.0)
    assert bus._unnormalize({1: 0.0}) == {1: 1000}
    assert bus._unnormalize({1: 1.0}) == {1: 30}


@pytest.mark.parametrize(
    ("range_min", "range_max", "expected"),
    [
        (776, 1215, [(776, 1023), (0, 191)]),
        (0, 640, [(0, 640)]),
        (-360, 580, [(664, 1023), (0, 580)]),
    ],
)
def test_scs215_calibration_records_ordered_segments(range_min, range_max, expected):
    assert split_circular_range(range_min, range_max, 1024) == expected


def test_scs215_wide_split_range_maps_both_segments_without_a_jump():
    bus = FeetechMotorsBus(
        "",
        {"motor_1": Motor(1, "scs215", MotorNormMode.RANGE_0_1)},
        protocol_version=1,
    )
    bus.calibration = {
        "motor_1": MotorCalibration(
            id=1,
            drive_mode=0,
            homing_offset=0,
            range_min=-360,
            range_max=580,
            range_segments=[(664, 1023), (0, 580)],
        )
    }

    assert bus._normalize({1: 664})[1] == pytest.approx(0.0)
    assert bus._normalize({1: 1023})[1] == pytest.approx(359 / 940)
    assert bus._normalize({1: 0})[1] == pytest.approx(360 / 940)
    assert bus._normalize({1: 580})[1] == pytest.approx(1.0)


@pytest.mark.parametrize(("present_raw", "goal_unit"), [(1023, 0.24), (0, 0.23)])
def test_scs215_crosses_split_seam_using_pwm_then_returns_to_position(present_raw, goal_unit):
    bus = FeetechMotorsBus(
        "",
        {"motor_1": Motor(1, "scs215", MotorNormMode.RANGE_0_1)},
        protocol_version=1,
    )
    bus.calibration = {
        "motor_1": MotorCalibration(
            id=1,
            drive_mode=0,
            homing_offset=0,
            range_min=1000,
            range_max=1100,
            range_segments=[(1000, 1023), (0, 76)],
        )
    }
    bus.port_handler = MagicMock(is_open=True)
    samples = [present_raw, 0, 8, 9] if present_raw == 1023 else [present_raw, 1023, 1015, 1014]
    bus.sync_read = MagicMock(side_effect=[{"motor_1": raw} for raw in samples])
    bus._sync_write = MagicMock()
    registers = {9: 0, 11: 1023, 46: 300, 48: 1}
    bus.read = MagicMock(side_effect=lambda name, *_args, **_kwargs: registers[SCS_SERIES_CONTROL_TABLE[name][0]])
    bus._write = MagicMock(side_effect=lambda addr, _length, _id, value, **_kw: registers.__setitem__(addr, value))

    with patch("lerobot.motors.feetech.feetech.time.sleep"):
        bus.sync_write("Goal_Position", {"motor_1": goal_unit})

    writes = [(c.args[0], c.args[3]) for c in bus._write.call_args_list]
    pwm = 1204 if present_raw == 1023 else 180
    assert (11, 0) in writes
    assert (44, pwm) in writes
    assert writes.index((11, 0)) < writes.index((44, pwm))
    assert (44, 0) in writes[writes.index((44, pwm)) + 1 :]
    assert registers[11] == 1023
    assert registers[44] == 0
    bus._sync_write.assert_called_once()


def test_scs215_accepts_absolute_position_goal_within_one_segment():
    bus = FeetechMotorsBus(
        "",
        {"motor_1": Motor(1, "scs215", MotorNormMode.RANGE_0_1)},
        protocol_version=1,
    )
    bus.calibration = {
        "motor_1": MotorCalibration(
            id=1,
            drive_mode=0,
            homing_offset=0,
            range_min=1000,
            range_max=1100,
            range_segments=[(1000, 1023), (0, 76)],
        )
    }
    bus.port_handler = MagicMock(is_open=True)
    bus.sync_read = MagicMock(return_value={"motor_1": 1023})
    bus._sync_write = MagicMock()

    bus.sync_write("Goal_Position", {"motor_1": 0.0})

    bus._sync_write.assert_called_once()
    assert bus._sync_write.call_args.args[2] == {1: 1000}


def test_scs215_bridge_timeout_stops_output_and_restores_the_native_mode():
    bus = FeetechMotorsBus(
        "",
        {"motor_1": Motor(1, "scs215", MotorNormMode.RANGE_0_1)},
        calibration={"motor_1": MotorCalibration(1, 0, 0, 1000, 1100)},
        protocol_version=1,
    )
    bus.port_handler = MagicMock(is_open=True)
    bus.sync_read = MagicMock(return_value={"motor_1": 1023})
    bus._sync_write = MagicMock()
    registers = {9: 0, 11: 1023, 46: 300, 48: 1}
    bus.read = MagicMock(side_effect=lambda name, *_args, **_kwargs: registers[SCS_SERIES_CONTROL_TABLE[name][0]])
    bus._write = MagicMock(side_effect=lambda addr, _length, _id, value, **_kw: registers.__setitem__(addr, value))

    with (
        patch("lerobot.motors.feetech.feetech.time.monotonic", side_effect=[0, 4]),
        patch("lerobot.motors.feetech.feetech.time.sleep"),
        pytest.raises(TimeoutError),
    ):
        bus.sync_write("Goal_Position", {"motor_1": 0.5})

    assert registers[44] == 0
    assert registers[40] == 0
    assert registers[11] == 1023
    assert registers[42] == 1023
    bus._sync_write.assert_not_called()


def test_scs215_unwrapped_calibration_stays_software_only():
    bus = FeetechMotorsBus(
        "",
        {"motor_1": Motor(1, "scs215", MotorNormMode.RANGE_0_1)},
        protocol_version=1,
    )
    bus.write = MagicMock()
    calibration = {
        "motor_1": MotorCalibration(
            id=1,
            drive_mode=0,
            homing_offset=0,
            range_min=1000,
            range_max=1100,
        )
    }

    bus.write_calibration(calibration)

    assert bus.write.call_args_list == [
        call("Min_Position_Limit", "motor_1", 0),
        call("Max_Position_Limit", "motor_1", 1023),
    ]
    assert bus.calibration == calibration


@pytest.mark.parametrize(("device_max", "expected"), [(1023, True), (900, False)])
def test_scs215_calibration_check_keeps_full_device_range(device_max, expected):
    bus = FeetechMotorsBus(
        "",
        {"motor_1": Motor(1, "scs215", MotorNormMode.RANGE_0_1)},
        protocol_version=1,
    )
    bus.calibration = {
        "motor_1": MotorCalibration(
            id=1,
            drive_mode=0,
            homing_offset=0,
            range_min=1000,
            range_max=1100,
        )
    }
    bus.read = MagicMock(
        side_effect=lambda data_name, *_args, **_kwargs: (
            0 if data_name == "Min_Position_Limit" else device_max
        )
    )

    assert bus.is_calibrated is expected


def test_protocol_1_sync_read_uses_individual_reads():
    bus = make_scs215_bus()
    bus._read = MagicMock(
        side_effect=[
            (123, bus._comm_success, bus._no_error),
            (456, bus._comm_success, bus._no_error),
            (789, bus._comm_success, bus._no_error),
        ]
    )

    values, comm = bus._sync_read(56, 2, [1, 2, 3], num_retry=2)

    assert values == {1: 123, 2: 456, 3: 789}
    assert comm == bus._comm_success
    assert bus._read.call_args_list == [
        call(56, 2, 1, num_retry=2, raise_on_error=True, err_msg=""),
        call(56, 2, 2, num_retry=2, raise_on_error=True, err_msg=""),
        call(56, 2, 3, num_retry=2, raise_on_error=True, err_msg=""),
    ]


def test_protocol_1_reset_calibration_uses_1024_range_without_homing_write():
    bus = make_scs215_bus()
    bus.write = MagicMock()

    bus.reset_calibration()

    assert bus.write.call_args_list == [
        call("Min_Position_Limit", "motor_1", 0, normalize=False),
        call("Max_Position_Limit", "motor_1", 1023, normalize=False),
        call("Min_Position_Limit", "motor_2", 0, normalize=False),
        call("Max_Position_Limit", "motor_2", 1023, normalize=False),
        call("Min_Position_Limit", "motor_3", 0, normalize=False),
        call("Max_Position_Limit", "motor_3", 1023, normalize=False),
    ]


def test_protocol_1_half_turn_homings_are_software_only():
    bus = make_scs215_bus()
    bus.reset_calibration = MagicMock()

    homings = bus.set_half_turn_homings()

    assert homings == {"motor_1": 0, "motor_2": 0, "motor_3": 0}
    bus.reset_calibration.assert_called_once_with(["motor_1", "motor_2", "motor_3"])
