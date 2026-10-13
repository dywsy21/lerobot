import json

import numpy as np
import pytest
from visual_servo_week6.calibration import save_yaml
from visual_servo_week6.geometry import rotation_error_rad, transform
from visual_servo_week6.hand_eye_workflow import (
    append_pair,
    create_session,
    estimate_board_pose,
    joint_positions_in_degrees,
    load_session,
    pose_from_observation,
    read_robot_pose,
    session_status,
    solve_session,
)
from visual_servo_week6.image_io import read_image, write_image
from visual_servo_week6.synthetic import generate_board_views


@pytest.fixture(scope="module")
def board_data(tmp_path_factory):
    directory = tmp_path_factory.mktemp("hand-eye-images")
    records, camera_matrix, expected, fixed_board = generate_board_views(directory / "images", count=16)
    intrinsics = directory / "intrinsics.synthetic.yaml"
    calibration = {
        "K": camera_matrix.tolist(),
        "distortion": [0.0] * 5,
        "image_size": [1280, 720],
        "board_inner_corners": [9, 6],
        "square_size_m": 0.025,
        "data_source": "synthetic",
    }
    save_yaml(intrinsics, calibration)
    return list(records.items()), intrinsics, calibration, expected, fixed_board


def robot_pose(record, frame="fixture_end"):
    return {
        "T_base_from_end": np.asarray(record["T_base_from_end"]).tolist(),
        "translation_unit": "metre",
        "end_frame": frame,
        "source": "synthetic ground truth",
    }


def test_complete_image_pairing_and_hand_eye_recovery(board_data, tmp_path):
    records, intrinsics, _, expected, fixed_board = board_data
    create_session(tmp_path, intrinsics)
    for index, (image, record) in enumerate(records):
        append_pair(tmp_path, image, robot_pose(record), validation=index >= 12)
    status = session_status(tmp_path)
    assert status["training_pairs"] == 12
    assert status["validation_pairs"] == 4
    assert status["ready_to_solve"]
    result = solve_session(tmp_path, tmp_path / "result", fixed_board[:3, 3])
    estimated = np.asarray(result["T_end_from_camera"])
    assert np.linalg.norm(estimated[:3, 3] - expected[:3, 3]) < 0.01
    assert rotation_error_rad(estimated[:3, :3], expected[:3, :3]) < np.deg2rad(2)
    assert len(result["validation"]) == 4
    assert result["independent_board_measurement"]["each_axis_within_5cm"]
    assert (tmp_path / "result" / "手眼标定报告.txt").is_file()
    np.testing.assert_allclose(estimated @ result["T_camera_from_end"], np.eye(4), atol=1e-8)


def test_duplicate_photo_cannot_be_paired_with_another_robot_pose(board_data, tmp_path):
    records, intrinsics, *_ = board_data
    create_session(tmp_path, intrinsics)
    append_pair(tmp_path, records[0][0], robot_pose(records[0][1]))
    with pytest.raises(ValueError, match="photograph"):
        append_pair(tmp_path, records[0][0], robot_pose(records[1][1]))
    assert len(load_session(tmp_path)["pairs"]) == 1


def test_missing_pose_and_wrong_units_are_rejected_before_recording(board_data, tmp_path):
    records, intrinsics, *_ = board_data
    create_session(tmp_path, intrinsics)
    pose = robot_pose(records[0][1])
    pose["translation_unit"] = "millimetre"
    with pytest.raises(ValueError, match="metres"):
        append_pair(tmp_path, records[0][0], pose)
    template = tmp_path / "robot_pose.template.json"
    with pytest.raises(ValueError, match="end_frame"):
        read_robot_pose(template)
    assert not load_session(tmp_path)["pairs"]


def test_resolution_and_end_frame_must_remain_consistent(board_data, tmp_path):
    records, intrinsics, *_ = board_data
    create_session(tmp_path, intrinsics)
    cropped = tmp_path / "wrong-resolution.png"
    write_image(cropped, read_image(records[0][0])[:360, :640])
    with pytest.raises(ValueError, match="resolution"):
        append_pair(tmp_path, cropped, robot_pose(records[0][1]))
    append_pair(tmp_path, records[0][0], robot_pose(records[0][1]))
    with pytest.raises(ValueError, match="same FK end_frame"):
        append_pair(tmp_path, records[1][0], robot_pose(records[1][1], frame="another_TCP"))


def test_reversing_corners_changes_only_the_board_coordinate_origin(board_data):
    records, _, calibration, *_ = board_data
    image = read_image(records[0][0])
    forward, _ = estimate_board_pose(image, calibration)
    backward, _ = estimate_board_pose(image, calibration, reverse_corners=True)
    old_from_new = transform(np.diag([-1.0, -1.0, 1.0]), [8 * 0.025, 5 * 0.025, 0])
    np.testing.assert_allclose(
        backward["T_camera_from_board"], np.asarray(forward["T_camera_from_board"]) @ old_from_new, atol=1e-5
    )


def test_translation_only_training_cannot_solve_hand_eye(board_data, tmp_path):
    records, intrinsics, *_ = board_data
    create_session(tmp_path, intrinsics)
    for index, (image, _) in enumerate(records[:12]):
        pose = {
            "T_base_from_end": transform(translation=[0.01 * index, 0, 0]).tolist(),
            "translation_unit": "metre",
            "end_frame": "fixture_end",
        }
        append_pair(tmp_path, image, pose)
    assert session_status(tmp_path)["rotation_axis_rank"] == 0
    assert not session_status(tmp_path)["ready_to_solve"]
    with pytest.raises(ValueError, match="distinct axes"):
        solve_session(tmp_path, tmp_path / "result")


def test_normalized_joint_values_need_measured_degree_mapping():
    observation = {"joint_b.pos": 0.75, "joint_a.pos": 0.5}
    with pytest.raises(ValueError, match="calibration"):
        joint_positions_in_degrees(observation, ["joint_a", "joint_b"], "normalized_0_1")
    mapping = {
        "joint_a": {"q_at_0_deg": -90, "q_at_1_deg": 90},
        "joint_b": {"q_at_0_deg": 80, "q_at_1_deg": -40},
    }
    np.testing.assert_array_equal(
        joint_positions_in_degrees(observation, ["joint_a", "joint_b"], "normalized_0_1", mapping), [0, -10]
    )
    mapping["joint_a"]["q_at_0_deg"] = None
    with pytest.raises(ValueError, match="measured angular endpoints"):
        joint_positions_in_degrees(observation, ["joint_a"], "normalized_0_1", mapping)


def test_repository_fk_adapter_uses_explicit_order_and_degrees():
    class RecordingKinematics:
        target_frame_name = "fixture_end"

        def forward_kinematics(self, angles):
            self.received = angles.copy()
            return transform(translation=[0.1, 0.2, 0.3])

    kinematics = RecordingKinematics()
    result = pose_from_observation(
        kinematics, {"joint_a.pos": 15, "joint_b.pos": -20}, ["joint_b", "joint_a"], "degrees"
    )
    np.testing.assert_array_equal(kinematics.received, [-20, 15])
    assert result["translation_unit"] == "metre"
    assert result["end_frame"] == "fixture_end"
    np.testing.assert_allclose(np.asarray(result["T_base_from_end"])[:3, 3], [0.1, 0.2, 0.3])


def test_unconfigured_session_is_empty_and_not_ready(board_data, tmp_path):
    _, intrinsics, *_ = board_data
    create_session(tmp_path, intrinsics)
    status = session_status(tmp_path)
    assert status["training_pairs"] == 0
    assert not status["ready_to_solve"]
    assert json.loads((tmp_path / "robot_pose.template.json").read_text())["T_base_from_end"] is None
    with pytest.raises(FileExistsError):
        create_session(tmp_path, intrinsics)
