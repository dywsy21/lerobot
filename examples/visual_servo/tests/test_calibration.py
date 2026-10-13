import numpy as np
from visual_servo_week6.calibration import calibrate_images
from visual_servo_week6.synthetic import generate_board_views
from visual_servo_week6.web_calibration import is_distinct, pose_descriptor


def test_distinct_poses_reject_duplicates_but_allow_translation_and_scale():
    points = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=float)
    first = pose_descriptor(points, (1280, 720))
    assert not is_distinct(first + 0.005, [first])
    assert not is_distinct(first[::-1], [first])
    assert is_distinct(pose_descriptor(points + [100, 0], (1280, 720)), [first])
    assert is_distinct(pose_descriptor(points * 1.5, (1280, 720)), [first])


def test_calibration_from_unicode_images_recovers_known_intrinsics(tmp_path):
    records, camera_matrix, _, _ = generate_board_views(tmp_path / "中文标定图像", count=15)
    result, _ = calibrate_images(list(records))
    assert result["valid_images"] == 15
    assert result["rms_px"] < 1
    recovered = np.asarray(result["K"])
    np.testing.assert_allclose(
        [recovered[0, 0], recovered[1, 1]], [camera_matrix[0, 0], camera_matrix[1, 1]], rtol=0.03
    )
    np.testing.assert_allclose(recovered[:2, 2], camera_matrix[:2, 2], atol=10)
