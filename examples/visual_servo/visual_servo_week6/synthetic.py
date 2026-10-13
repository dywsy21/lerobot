from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .calibration import board_points
from .geometry import inverse, transform
from .image_io import write_image
from .markers import marker_points


def project(points, pose, camera_matrix, distortion=None):
    if distortion is None:
        distortion = np.zeros(5)
    rvec = cv2.Rodrigues(pose[:3, :3])[0]
    pixels, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64), rvec, pose[:3, 3], camera_matrix, distortion
    )
    return pixels.reshape(-1, 2)


def warp_texture(texture, world_corners, pose, camera_matrix, image_size):
    height, width = texture.shape[:2]
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    target = project(world_corners, pose, camera_matrix).astype(np.float32)
    homography = cv2.getPerspectiveTransform(source, target)
    image = cv2.warpPerspective(texture, homography, image_size, flags=cv2.INTER_LINEAR, borderValue=210)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def generate_board_views(directory, inner_corners=(9, 6), square_size_m=0.025, count=20):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nx, ny = inner_corners
    cell = 120
    texture = np.full(((ny + 3) * cell, (nx + 3) * cell), 255, np.uint8)
    for y in range(ny + 1):
        for x in range(nx + 1):
            if (x + y) % 2 == 0:
                texture[(y + 1) * cell : (y + 2) * cell, (x + 1) * cell : (x + 2) * cell] = 0
    corners = (
        np.array([[-2, -2, 0], [nx + 1, -2, 0], [nx + 1, ny + 1, 0], [-2, ny + 1, 0]], float) * square_size_m
    )
    image_size = (1280, 720)
    camera_matrix = np.array([[900.0, 0, 640.0], [0, 920.0, 360.0], [0, 0, 1]])
    hand_eye_matrix = transform(
        Rotation.from_euler("xyz", [0.02, -0.08, 0.03]).as_matrix(), [0.035, -0.018, 0.055]
    )
    fixed_board = transform(Rotation.from_euler("xyz", [0.1, 0.0, 0.2]).as_matrix(), [0.32, 0.08, 0.2])
    center = board_points(inner_corners, square_size_m).mean(axis=0)
    rng = np.random.default_rng(336)
    records = {}
    for index in range(count):
        rotation = Rotation.from_euler("xyz", rng.uniform([-0.4, -0.4, -0.15], [0.4, 0.4, 0.15])).as_matrix()
        translation = (
            np.array([rng.uniform(-0.07, 0.07), rng.uniform(-0.045, 0.045), rng.uniform(0.6, 0.9)])
            - rotation @ center
        )
        board_pose = transform(rotation, translation)
        image = warp_texture(texture, corners, board_pose, camera_matrix, image_size)
        image = cv2.GaussianBlur(image, (3, 3), 0.5)
        path = directory / f"view_{index:02d}.png"
        write_image(path, image)
        records[str(path)] = {
            "T_camera_from_board": board_pose,
            "T_base_from_end": fixed_board @ inverse(board_pose) @ inverse(hand_eye_matrix),
        }
    return records, camera_matrix, hand_eye_matrix, fixed_board


def render_marker(pose, camera_matrix, side_length_m=0.06, image_size=(1280, 720)):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    texture = cv2.aruco.generateImageMarker(dictionary, 0, 400)
    return warp_texture(texture, marker_points(side_length_m), pose, camera_matrix, image_size)
