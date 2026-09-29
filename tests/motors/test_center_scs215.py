from unittest.mock import MagicMock, patch

import pytest

import center_scs215 as script


class FakeServo:
    def __init__(self, samples=(900, 700, 515, 514, 512), motor_mode=False):
        self.motor_id = 4
        self.samples = iter(samples)
        self.registers = {3: 1315, 62: 74, 9: 0, 11: 0 if motor_mode else 1023, 48: 0, 40: 1}
        self.writes = []

    def read(self, address, size=2):
        return self.registers[address]

    def position(self):
        return next(self.samples)

    def write(self, address, value, size=2):
        # Limits must never be persisted to EEPROM by this helper.
        if address in (9, 11):
            assert self.registers[48] == 1
        self.writes.append((address, value, size))
        self.registers[address] = value


@pytest.mark.parametrize("release", [False, True])
def test_raw_goal_is_512_and_torque_is_enabled_only_after_priming(release):
    servo = FakeServo()
    with patch.object(script.time, "sleep"):
        assert script.center_servo(servo, 200, 10, release) == 512
    assert servo.writes[:5] == [(40, 0, 1), (44, 0, 2), (46, 200, 2), (42, 512, 2), (40, 1, 1)]
    assert servo.registers[40] == (0 if release else 1)
    assert not any(address in (5, 6, 9, 11, 48) for address, _, _ in servo.writes)


def test_motor_mode_is_recovered_without_persistent_limit_changes():
    servo = FakeServo(motor_mode=True)
    with patch.object(script.time, "sleep"):
        script.center_servo(servo, 200, 10, False)
    assert servo.registers[11] == 1023
    assert servo.registers[48] == 0
    assert servo.writes.index((42, 512, 2)) < servo.writes.index((11, 1023, 2))
    assert servo.writes.index((11, 1023, 2)) < servo.writes.index((40, 1, 1))


def test_timeout_reports_failure_and_disables_only_the_selected_motor():
    servo = FakeServo(samples=(900, 850))
    with (
        patch.object(script.time, "monotonic", side_effect=[0, 0, 11]),
        patch.object(script.time, "sleep"),
        pytest.raises(TimeoutError, match="last position=850"),
    ):
        script.center_servo(servo, 200, 10, False)
    assert servo.writes[-2:] == [(40, 0, 1), (44, 0, 2)]


@pytest.mark.parametrize("error", [OSError("read failed"), KeyboardInterrupt()])
def test_feedback_failure_or_interrupt_requests_torque_off(error):
    servo = FakeServo()
    servo.position = MagicMock(side_effect=[900, error])
    with pytest.raises(type(error)):
        script.center_servo(servo, 200, 10, False)
    assert servo.writes[-2:] == [(40, 0, 1), (44, 0, 2)]


@pytest.mark.parametrize("register,value", [(3, 777), (9, 600), (11, 400)])
def test_invalid_model_or_limits_cause_no_writes(register, value):
    servo = FakeServo()
    servo.registers[register] = value
    with pytest.raises(RuntimeError):
        script.center_servo(servo, 200, 10, False)
    assert not servo.writes


@pytest.mark.parametrize("arguments", [["--id", "254"], ["--id", "-1"], ["--id", "1", "--timeout", "nan"]])
def test_invalid_cli_arguments(arguments):
    with pytest.raises(SystemExit):
        script.parse_args(arguments)


def test_packet_methods_always_address_the_requested_id():
    packet = MagicMock()
    packet.read2ByteTxRx.return_value = (512, 0, 0)
    packet.write2ByteTxRx.return_value = (0, 0)
    servo = script.Servo("fake-port", packet, 4)
    with patch.object(script, "scs", COMM_SUCCESS=0):
        assert servo.position() == 512
        servo.write(42, 512)
    packet.read2ByteTxRx.assert_called_once_with("fake-port", 4, 56)
    packet.write2ByteTxRx.assert_called_once_with("fake-port", 4, 42, 512)


def test_main_uses_protocol_one_and_closes_port_without_extra_motion():
    sdk = MagicMock()
    with patch.object(script, "scs", sdk), patch.object(script, "center_servo") as center:
        assert script.main(["--id", "4"]) == 0
    sdk.PortHandler.assert_called_once_with("COM5")
    sdk.PacketHandler.assert_called_once_with(1)
    sdk.PortHandler.return_value.closePort.assert_called_once()
    assert center.call_args.args[0].motor_id == 4


def test_main_closes_port_on_connection_error():
    sdk = MagicMock()
    sdk.PortHandler.return_value.openPort.side_effect = OSError("port busy")
    with patch.object(script, "scs", sdk), patch.object(script, "center_servo") as center:
        assert script.main(["--id", "4"]) == 1
    center.assert_not_called()
    sdk.PortHandler.return_value.closePort.assert_called_once()
