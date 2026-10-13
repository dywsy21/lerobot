import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from visual_servo_week6.geometry import transform
from visual_servo_week6.robot_model import RobotModel, UrdfKinematics

URDF = """<robot name="test_fixture">
<link name="base"/><link name="first"/><link name="second"/><link name="camera_mount"/>
<joint name="pan" type="revolute"><parent link="base"/><child link="first"/>
<origin xyz="0.1 0.2 0.3" rpy="0.1 -0.2 0.3"/><axis xyz="0 0 1"/>
<limit lower="-1.5" upper="1.5" effort="1" velocity="1"/></joint>
<joint name="bend" type="revolute"><parent link="first"/><child link="second"/>
<origin xyz="0.2 0 0"/><axis xyz="0 1 0"/>
<limit lower="-1.5" upper="1.5" effort="1" velocity="1"/></joint>
<joint name="mount" type="fixed"><parent link="second"/><child link="camera_mount"/>
<origin xyz="0.15 0 0.03" rpy="0 0.1 0"/></joint></robot>"""


@pytest.fixture
def robot_model_files(tmp_path):
    (tmp_path / "fixture.urdf").write_text(URDF)
    config = {
        "urdf_path": "fixture.urdf",
        "base_frame": "base",
        "end_frame": "camera_mount",
        "joint_names": ["bend", "pan"],
        "backend": "portable_urdf",
        "observation_units": "normalized_0_1",
        "normalized_joint_calibration": {
            "pan": {"q_at_0_deg": -80, "q_at_1_deg": 80},
            "bend": {"q_at_0_deg": 70, "q_at_1_deg": -70},
        },
        "joint_limits_deg": {"pan": [-70, 70], "bend": [-60, 60]},
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(config))
    return tmp_path, path, config


def test_urdf_fk_applies_origins_axes_and_explicit_degree_order(robot_model_files):
    root, _, _ = robot_model_files
    kinematics = UrdfKinematics(root / "fixture.urdf", "camera_mount", ["bend", "pan"])
    expected = (
        transform(
            Rotation.from_euler("z", 0.3).as_matrix()
            @ Rotation.from_euler("y", -0.2).as_matrix()
            @ Rotation.from_euler("x", 0.1).as_matrix(),
            [0.1, 0.2, 0.3],
        )
        @ transform(Rotation.from_euler("z", 30, degrees=True).as_matrix())
        @ transform(translation=[0.2, 0, 0])
        @ transform(Rotation.from_euler("y", -20, degrees=True).as_matrix())
        @ transform(Rotation.from_euler("y", 0.1).as_matrix(), [0.15, 0, 0.03])
    )
    np.testing.assert_allclose(kinematics.forward_kinematics([-20, 30]), expected, atol=1e-12)
    assert kinematics.base_frame == "base"


def test_normalized_observation_and_action_are_inverse_even_with_reversed_sign(robot_model_files):
    root, path, _ = robot_model_files
    model = RobotModel(path, root)
    observation = {"pan.pos": 0.625, "bend.pos": 0.75}
    q = model.angles(observation)
    np.testing.assert_allclose(np.rad2deg(q), [-35, 20])
    assert model.action(q) == {"bend": 0.75, "pan": 0.625}
    assert model.pose(observation)["base_frame"] == "base"
    with pytest.raises(ValueError, match="limits"):
        model.angles({"pan.pos": 1, "bend.pos": 0.5})
    with pytest.raises(ValueError, match="limits"):
        model.action(np.deg2rad([0, 80]))


def test_bounded_ik_recovers_reachable_pose_and_rejects_unreachable_target(robot_model_files):
    root, _, _ = robot_model_files
    kinematics = UrdfKinematics(root / "fixture.urdf", "camera_mount", ["bend", "pan"])
    target = kinematics.forward_kinematics([-20, 30])
    result = kinematics.inverse_kinematics([0, 0], target, [[-60, 60], [-70, 70]])
    np.testing.assert_allclose(result, [-20, 30], atol=1e-4)
    with pytest.raises(ValueError, match="unreachable"):
        kinematics.inverse_kinematics([0, 0], transform(translation=[5, 0, 0]), [[-60, 60], [-70, 70]])


def test_urdf_rejects_wrong_chain_missing_frames_and_mimic_joints(robot_model_files):
    root, _, _ = robot_model_files
    with pytest.raises(ValueError, match="movable"):
        UrdfKinematics(root / "fixture.urdf", "camera_mount", ["pan"])
    with pytest.raises(ValueError, match="end_frame"):
        UrdfKinematics(root / "fixture.urdf", "missing", ["pan", "bend"])
    (root / "mimic.urdf").write_text(
        URDF.replace('<axis xyz="0 1 0"/>', '<mimic joint="pan"/><axis xyz="0 1 0"/>')
    )
    with pytest.raises(ValueError, match="Unsupported"):
        UrdfKinematics(root / "mimic.urdf", "camera_mount", ["pan", "bend"])


def test_real_configuration_cannot_use_unmeasured_endpoints_or_empty_limits(robot_model_files):
    root, path, config = robot_model_files
    config["normalized_joint_calibration"]["pan"]["q_at_0_deg"] = None
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="measured"):
        RobotModel(path, root)
    config["normalized_joint_calibration"]["pan"]["q_at_0_deg"] = -80
    config["joint_limits_deg"]["pan"] = None
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="joint_limits"):
        RobotModel(path, root)
