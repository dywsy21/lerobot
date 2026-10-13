import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from visual_servo_week6.geometry import camera_velocity_to_end_target, inverse, se3_exp, transform
from visual_servo_week6.hand_eye import calibrate_hand_eye
from visual_servo_week6.ibvs import compute_ibvs, interaction_matrix, simple_camera_step
from visual_servo_week6.kinematics import DHRobot
from visual_servo_week6.markers import detect_marker, marker_points
from visual_servo_week6.synthetic import project, render_marker

camera_matrix = np.array([[900.0, 0, 640], [0, 920.0, 360], [0, 0, 1]])


def test_interaction_matrix_agrees_with_finite_camera_motion():
    points = np.array([[0.1, -0.05, 0.7], [-0.08, 0.03, 0.9]])
    velocity = np.array([0.03, -0.02, 0.01, 0.05, -0.04, 0.02])
    normalized = points[:, :2] / points[:, 2, None]
    moved = (inverse(se3_exp(velocity, 1e-6)) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    derivative = ((moved[:, :2] / moved[:, 2, None] - normalized) / 1e-6).reshape(-1)
    np.testing.assert_allclose(interaction_matrix(normalized, points[:, 2]) @ velocity, derivative, atol=1e-7)


def test_hand_eye_recovers_known_transform():
    hand_eye_matrix = transform(Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix(), [0.03, -0.02, 0.05])
    fixed_board = transform(translation=[0.4, 0.1, 0.3])
    rng = np.random.default_rng(10)
    grippers = [
        transform(Rotation.from_rotvec(rng.normal(0, 0.3, 3)).as_matrix(), rng.normal(0, 0.1, 3))
        for _ in range(15)
    ]
    cameras = [inverse(hand_eye_matrix) @ inverse(gripper_pose) @ fixed_board for gripper_pose in grippers]
    result = calibrate_hand_eye(grippers, cameras)
    np.testing.assert_allclose(result["T_end_from_camera"], hand_eye_matrix, atol=1e-8)


def test_hand_eye_rejects_translation_only_data():
    poses = [transform(translation=[i * 0.01, 0, 0]) for i in range(12)]
    with pytest.raises(ValueError, match="distinct axes"):
        calibrate_hand_eye(poses, poses)


def test_control_stops_for_missing_target_and_reached_goal():
    assert compute_ibvs(None, None, None, None, None).status == "target_lost"
    points = np.array([[600.0, 320], [680, 320], [680, 400], [600, 400]])
    result = compute_ibvs(points, points, camera_matrix, np.zeros(5), np.ones(4))
    assert result.status == "reached"
    np.testing.assert_array_equal(result.velocity_camera, np.zeros(6))


def test_controller_speed_limits_and_error_reduction():
    objects = marker_points(0.06)
    current = transform(translation=[0.07, -0.04, 0.7])
    goal = project(objects, transform(translation=[0, 0, 0.45]), camera_matrix)
    before = project(objects, current, camera_matrix)
    result = compute_ibvs(before, goal, camera_matrix, np.zeros(5), np.full(4, 0.7))
    assert np.linalg.norm(result.velocity_camera[:3]) <= 0.05 + 1e-10
    assert np.linalg.norm(result.velocity_camera[3:]) <= 0.3 + 1e-10
    after = project(objects, inverse(se3_exp(result.velocity_camera, 0.1)) @ current, camera_matrix)
    assert np.linalg.norm(after - goal) < np.linalg.norm(before - goal)


def test_nonpositive_depth_is_rejected():
    with pytest.raises(ValueError, match="positive depth"):
        interaction_matrix([[0, 0]], [0])


def test_camera_rotation_respects_camera_end_offset():
    hand_eye_matrix = transform(translation=[0.1, 0, 0])
    motion = [0, 0, 0, 0, 0, 0.1]
    target = camera_velocity_to_end_target(np.eye(4), hand_eye_matrix, motion, 0.1)
    np.testing.assert_allclose(target @ hand_eye_matrix, hand_eye_matrix @ se3_exp(motion, 0.1), atol=1e-10)
    assert np.linalg.norm(target[:3, 3]) > 0


def test_aruco_image_pose_round_trip():
    true_pose = transform(Rotation.from_euler("xyz", [0.05, -0.1, 0.05]).as_matrix(), [0.025, 0.01, 0.5])
    detected = detect_marker(render_marker(true_pose, camera_matrix), camera_matrix, np.zeros(5))
    assert detected is not None
    _, estimated = detected
    np.testing.assert_allclose(estimated[:3, 3], true_pose[:3, 3], atol=0.005)


def test_simple_rule_camera_directions():
    assert simple_camera_step([[100, 360]] * 4, (1280, 720), 0.5)[0] < 0
    assert simple_camera_step([[1100, 360]] * 4, (1280, 720), 0.5)[0] > 0
    assert simple_camera_step([[640, 360]] * 4, (1280, 720), 0.1).sum() == 0


def test_generic_dh_ik_round_trip():
    parameters = [
        [0, np.pi / 2, 0.08, 0],
        [0.1, 0, 0, 0],
        [0.14, 0, 0, 0],
        [0.1, np.pi / 2, 0, 0],
        [0, -np.pi / 2, 0, 0],
        [0, 0, 0.05, 0],
    ]
    robot = DHRobot(parameters, [[-np.pi, np.pi]] * 6)
    joints = np.array([0.2, -0.6, 0.8, 0.2, -0.3, 0.1])
    target = robot.forward(joints)
    recovered = robot.inverse(target, joints + 0.02)
    np.testing.assert_allclose(robot.forward(recovered), target, atol=1e-4)
