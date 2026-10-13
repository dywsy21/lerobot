import cv2
import numpy as np

from .geometry import rotation_error_rad, transform, validate_transform


def calibrate_hand_eye(gripper_to_base, target_to_camera, minimum_pairs=12):
    """Eye-in-hand: inputs are B_T_E and C_T_M; output is E_T_C."""
    if len(gripper_to_base) != len(target_to_camera) or len(gripper_to_base) < minimum_pairs:
        raise ValueError(f"Need at least {minimum_pairs} paired robot/board poses")
    gripper_to_base = [validate_transform(pose) for pose in gripper_to_base]
    target_to_camera = [validate_transform(pose) for pose in target_to_camera]
    relative_vectors = [
        cv2.Rodrigues(gripper_to_base[0][:3, :3].T @ pose[:3, :3])[0].reshape(3)
        for pose in gripper_to_base[1:]
    ]
    if np.linalg.matrix_rank(np.array(relative_vectors), tol=1e-3) < 2:
        raise ValueError("Hand-eye poses need rotations about at least two distinct axes")
    rotation, translation = cv2.calibrateHandEye(
        [pose[:3, :3] for pose in gripper_to_base],
        [pose[:3, 3] for pose in gripper_to_base],
        [pose[:3, :3] for pose in target_to_camera],
        [pose[:3, 3] for pose in target_to_camera],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    result = validate_transform(transform(rotation, translation))
    board_in_base = [
        gripper @ result @ board for gripper, board in zip(gripper_to_base, target_to_camera, strict=True)
    ]
    positions = np.array([pose[:3, 3] for pose in board_in_base])
    mean_position = positions.mean(axis=0)
    return {
        "T_end_from_camera": result.tolist(),
        "translation_unit": "metre",
        "pose_pairs": len(board_in_base),
        "board_origin_mean_m": mean_position.tolist(),
        "board_origin_spread_rms_m": float(
            np.sqrt(np.mean(np.sum((positions - mean_position) ** 2, axis=1)))
        ),
        "board_rotation_spread_max_deg": float(
            np.degrees(
                max(rotation_error_rad(pose[:3, :3], board_in_base[0][:3, :3]) for pose in board_in_base)
            )
        ),
        "note": "Pose consistency is not a substitute for comparison with an independently measured board origin.",
    }
