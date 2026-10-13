import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from visual_servo_week6.geometry import inverse, transform
from visual_servo_week6.task3_core import (
    ServoSettings,
    VisualServo,
    camera_body_jacobian,
    central_goal,
    project_marker,
    propose_joint_step,
)
from visual_servo_week6.task3_simulation import MarkerScene

camera_matrix = np.array([[900.0, 0, 640], [0, 920.0, 360], [0, 0, 1]])


def test_loss_and_stale_images_generate_zero_commands():
    servo = VisualServo(camera_matrix, np.zeros(5), (1280, 720))
    image = np.full((720, 1280, 3), 210, np.uint8)
    for result in [servo.evaluate(image), servo.evaluate(image, captured_at=1, now=2)]:
        assert not any(result["velocity_camera"])
        assert not any(result["step_camera_m"])
        assert result["hardware_commands_sent"] is False


def test_distorted_image_pose_and_velocity_limits():
    distortion = np.array([0.08, -0.03, 0.001, -0.002, 0.01])
    servo = VisualServo(camera_matrix, distortion, (1280, 720))
    scene = MarkerScene(servo)
    result = servo.evaluate(scene.image())
    assert result["status"] == "tracking"
    np.testing.assert_allclose(
        np.asarray(result["T_camera_from_marker"])[:3, 3], scene.pose[:3, 3], atol=0.006
    )
    assert np.linalg.norm(result["velocity_camera"][:3]) <= 0.01 + 1e-10
    assert np.linalg.norm(result["velocity_camera"][3:]) <= 0.1 + 1e-10
    paused = servo.evaluate(scene.image(), active=False)
    assert paused["status"] == "paused" and not any(paused["velocity_camera"])


def test_explicit_broker_resize_preserves_native_pixel_features():
    servo = VisualServo(camera_matrix, np.zeros(5), (1280, 720))
    image = MarkerScene(servo).image()
    scaled = cv2.resize(image, (960, 540))
    assert servo.evaluate(scaled)["status"] == "resolution_mismatch"
    original = servo.evaluate(image)
    resized = servo.evaluate(scaled, allow_resize=True)
    np.testing.assert_allclose(resized["corners_px"], original["corners_px"], atol=1.0)


def test_camera_body_jacobian_contains_camera_offset_motion():
    def fk(q):
        return transform(Rotation.from_rotvec([0, 0, q[0]]).as_matrix(), [q[1], 0, 0])

    q = np.array([0.1, 0.02])
    x = transform(translation=[0.1, 0, 0])
    jacobian = camera_body_jacobian(fk, q, x)
    assert abs(jacobian[1, 0] - 0.1) < 1e-7
    np.testing.assert_allclose(jacobian[3:, 0], [0, 0, 1], atol=1e-7)


def test_joint_ibvs_decreases_error_for_underactuated_three_joint_model():
    def fk(q):
        return transform(translation=q)

    q = np.array([0.035, -0.02, 0.0])
    x = np.eye(4)
    fixed_marker = transform(translation=[0, 0, 0.55])
    goal = project_marker(fixed_marker, camera_matrix, np.zeros(5), 0.06)
    servo = VisualServo(camera_matrix, np.zeros(5), (1280, 720), goal=goal)
    pose = inverse(fk(q)) @ fixed_marker
    points = project_marker(pose, camera_matrix, np.zeros(5), 0.06)
    advice = {"status": "tracking", "corners_px": points.tolist(), "depths_m": [pose[2, 3]] * 4}
    proposal = propose_joint_step(advice, servo, fk, q, [[-0.1, 0.1]] * 3, x)
    q_next = np.asarray(proposal["q_next_rad"])
    after = project_marker(inverse(fk(q_next)) @ fixed_marker, camera_matrix, np.zeros(5), 0.06)
    assert np.linalg.norm(after - goal) < np.linalg.norm(points - goal)
    assert max(abs(q_next - q)) <= 0.005 + 1e-10
    assert proposal["hardware_commands_sent"] is False


def test_goal_order_and_invalid_marker_configuration():
    with pytest.raises(ValueError):
        ServoSettings(marker_id=100)
    with pytest.raises(ValueError):
        ServoSettings(side_length_m=float("nan"))
    goal, _ = central_goal(camera_matrix, np.zeros(5), (1280, 720), 0.06)
    assert goal[0, 0] < goal[1, 0] and goal[0, 1] < goal[3, 1]
