import pytest

from lerobot.motors.feetech.scs215 import Scs215SeamController


@pytest.mark.parametrize(
    ("samples", "target", "pwm"),
    [([990, 1011, 1023, 0, 3, 8, 10], 20, 1204), ([30, 12, 0, 1023, 1020, 1015, 1013], 1000, 180)],
)
def test_crossing_has_directional_drive_and_confirmed_position_handover(samples, target, pwm):
    controller = Scs215SeamController([(776, 1023), (0, 191)])
    commands = [controller.step(raw, target, i * 0.02) for i, raw in enumerate(samples)]
    assert commands[0].kind == "position"
    assert commands[1].kind == "enter_pwm"
    assert commands[1].value == pwm
    assert all(command is None or command.kind != "exit_pwm" for command in commands[:-1])
    assert commands[-1].kind == "exit_pwm"
    assert commands[-1].value == target


def test_retarget_before_crossing_cancels_approach():
    controller = Scs215SeamController([(776, 1023), (0, 191)])
    assert controller.step(900, 20, 0).value == 1011
    command = controller.step(910, 800, 0.02)
    assert (command.kind, command.value) == ("position", 800)


def test_native_steady_error_can_hand_over_without_reaching_the_exact_entry():
    controller = Scs215SeamController([(776, 1023), (0, 191)])
    assert controller.step(989, 20, 0).kind == "position"
    assert controller.step(1004, 20, 0.5).kind == "enter_pwm"


def test_retarget_during_blind_passage_hands_over_before_reversing():
    controller = Scs215SeamController([(776, 1023), (0, 191)])
    controller.step(1023, 20, 0)
    controller.step(0, 900, 0.02)
    controller.step(8, 900, 0.04)
    command = controller.step(10, 900, 0.06)
    assert (command.kind, command.value) == ("exit_pwm", 10)
    assert controller.step(10, 900, 0.08).kind == "enter_pwm"


def test_stuck_feedback_does_not_look_like_a_successful_crossing():
    controller = Scs215SeamController([(776, 1023), (0, 191)])
    controller.step(1023, 20, 0)
    with pytest.raises(TimeoutError):
        controller.step(1023, 20, 3.1)


def test_reverse_feedback_is_not_mistaken_for_progress():
    controller = Scs215SeamController([(776, 1023), (0, 191)])
    controller.step(1011, 20, 0)
    with pytest.raises(RuntimeError, match="direction"):
        controller.step(980, 20, 0.02)
