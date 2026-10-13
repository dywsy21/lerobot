import numpy as np
from visual_servo_week6.image_io import read_image, write_image


def test_unicode_png_round_trip(tmp_path):
    path = tmp_path / "相机标定" / "棋盘格.png"
    image = np.random.default_rng(1).integers(0, 256, (30, 40, 3), dtype=np.uint8)
    write_image(path, image)
    np.testing.assert_array_equal(read_image(path), image)


def test_unreadable_image_returns_none(tmp_path):
    path = tmp_path / "invalid.png"
    path.write_bytes(b"not an image")
    assert read_image(path) is None
    assert read_image(tmp_path / "missing.png") is None
