import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .calibration import calibrate_images, save_yaml
from .camera import capture_images
from .demo import run_demo
from .hand_eye import calibrate_hand_eye
from .ibvs import compute_ibvs, simple_camera_step
from .image_io import read_image
from .markers import detect_marker, marker_points


def main():
    parser = argparse.ArgumentParser(
        description="Week 6 camera-only tools and offline IBVS. No robot commands are sent."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run synthetic calibration, hand-eye and IBVS checks")
    demo.add_argument("--output", default=None)
    capture = commands.add_parser("capture", help="Save images from an explicitly selected camera")
    capture.add_argument("--camera", type=int, required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--width", type=int, default=1280)
    capture.add_argument("--height", type=int, default=720)
    capture.add_argument("--count", type=int, default=1)
    capture.add_argument("--interval", type=float, default=3.0)
    capture.add_argument("--board", nargs=2, type=int, default=None, metavar=("NX", "NY"))
    calibration = commands.add_parser(
        "calibrate", help="Estimate intrinsics from at least 15 valid board images"
    )
    calibration.add_argument("--images", required=True)
    calibration.add_argument("--board", nargs=2, type=int, required=True, metavar=("NX", "NY"))
    calibration.add_argument("--square-size-m", type=float, default=0.025)
    calibration.add_argument("--output", required=True)
    hand_eye = commands.add_parser("hand-eye", help="Estimate E_T_C from paired B_T_E and C_T_M matrices")
    hand_eye.add_argument("--poses", required=True)
    hand_eye.add_argument("--output", required=True)
    marker = commands.add_parser("marker", help="Detect an ArUco marker and estimate its camera-frame pose")
    marker.add_argument("--image", required=True)
    marker.add_argument("--intrinsics", required=True)
    marker.add_argument("--side-length-m", type=float, required=True)
    marker.add_argument("--id", type=int, default=0)
    control = commands.add_parser(
        "ibvs-step", help="Compute a camera-only dry-run advisory, without moving devices"
    )
    control.add_argument("--image", required=True)
    control.add_argument("--intrinsics", required=True)
    control.add_argument("--side-length-m", type=float, required=True)
    control.add_argument("--id", type=int, default=0)
    control.add_argument("--goal", required=True, help="JSON file containing four [u,v] target points")
    args = parser.parse_args()

    if args.command == "demo":
        directory = args.output or f"outputs/demo-{datetime.now():%Y%m%d-%H%M%S}"
        run_demo(directory)
    elif args.command == "capture":
        capture_images(
            args.camera, args.output, args.width, args.height, args.count, args.interval, args.board
        )
    elif args.command == "calibrate":
        paths = sorted(
            path
            for path in Path(args.images).iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        result, _ = calibrate_images(paths, tuple(args.board), args.square_size_m)
        result["data_source"] = "input_images; verify the camera, focus and image size before reuse"
        save_yaml(args.output, result)
        print(
            json.dumps(
                {"rms_px": result["rms_px"], "valid_images": result["valid_images"], "K": result["K"]},
                indent=2,
            )
        )
    elif args.command == "hand-eye":
        source = json.loads(Path(args.poses).read_text(encoding="utf-8"))
        pairs = source["pairs"]
        result = calibrate_hand_eye(
            [np.asarray(pair["T_base_from_end"]) for pair in pairs],
            [np.asarray(pair["T_camera_from_board"]) for pair in pairs],
        )
        result["data_source"] = source.get("data_source", "input_pose_pairs")
        save_yaml(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        image = read_image(args.image)
        if image is None:
            raise ValueError("Image could not be read")
        calibration = yaml.safe_load(Path(args.intrinsics).read_text(encoding="utf-8"))
        if list(image.shape[1::-1]) != calibration["image_size"]:
            raise ValueError("Image size differs from the calibrated resolution")
        camera_matrix, distortion = np.asarray(calibration["K"]), np.asarray(calibration["distortion"])
        detected = detect_marker(image, camera_matrix, distortion, args.side_length_m, args.id)
        if detected is None:
            print(
                json.dumps(
                    {"status": "target_lost", "velocity_camera": [0.0] * 6, "hardware_commands_sent": False}
                )
            )
            return
        points, pose = detected
        if args.command == "marker":
            print(
                json.dumps(
                    {
                        "id": args.id,
                        "corners_px": points.tolist(),
                        "T_camera_from_marker": pose.tolist(),
                        "translation_unit": "metre",
                    },
                    indent=2,
                )
            )
        else:
            goal = json.loads(Path(args.goal).read_text(encoding="utf-8"))
            depths = (marker_points(args.side_length_m) @ pose[:3, :3].T + pose[:3, 3])[:, 2]
            result = compute_ibvs(points, goal, camera_matrix, distortion, depths)
            simple = simple_camera_step(points, tuple(calibration["image_size"]), float(depths.mean()))
            print(
                json.dumps(
                    {
                        "status": result.status,
                        "rmse_px": result.rmse_px,
                        "velocity_camera": result.velocity_camera.tolist(),
                        "simple_step_camera_m": simple.tolist(),
                        "hardware_commands_sent": False,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
