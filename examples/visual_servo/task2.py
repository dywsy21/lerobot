r"""任务2：固定棋盘，记录照片和同一静止姿态的FK，再求解手眼矩阵。

在仓库根目录运行 uv run --no-sync python examples/visual_servo/task2.py --help。
该入口读取照片和位姿数据；机械臂运动由现有机器人控制接口完成。
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from visual_servo_week6.camera import capture_images
from visual_servo_week6.geometry import rotation_error_rad
from visual_servo_week6.hand_eye_workflow import (
    append_pair,
    create_session,
    estimate_board_pose,
    joint_positions_in_degrees,
    load_session,
    read_robot_pose,
    session_status,
    solve_session,
)
from visual_servo_week6.image_io import read_image, write_image
from visual_servo_week6.robot_model import RobotModel

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
SO_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def resolve(path):
    return ROOT / path


def model_configuration(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    urdf = data.get("urdf_path")
    if not isinstance(urdf, str) or not urdf.strip():
        raise ValueError("请在 robot_model.json 中填写与实物对应的 urdf_path")
    urdf = REPOSITORY / urdf
    if not urdf.is_file():
        raise ValueError(f"URDF 文件不存在：{urdf}")
    if not data.get("joint_names") or not data.get("end_frame"):
        raise ValueError("请填写URDF关节顺序和FK末端坐标系名称")
    data["urdf_path"] = str(urdf)
    return data


def main():
    parser = argparse.ArgumentParser(description="任务2：眼在手上的手眼标定")
    parser.add_argument("--directory", default="data/task2", help="实验数据目录，相对本实验目录")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="读取任务1真实内参，建立空配对数据集")
    initialize.add_argument("--intrinsics", default="results/task1/intrinsics.yaml")
    commands.add_parser("status", help="检查已保存的配对数量与旋转姿态多样性")
    preflight = commands.add_parser("preflight", help="检查FK的URDF、依赖和角度标定配置")
    preflight.add_argument("--model", default="configs/robot_model.json")
    board = commands.add_parser("board-pose", help="检测棋盘并标出固定板原点和坐标轴")
    board.add_argument("--image", required=True)
    board.add_argument("--output", default="outputs/task2/board-pose.png")
    add = commands.add_parser("add", help="导入同一静止姿态的一张照片与FK文件")
    add.add_argument("--image", required=True)
    capture = commands.add_parser("capture", help="在给定FK对应的静止姿态下拍照并保存配对")
    capture.add_argument("--camera", type=int, required=True)
    for command in (add, capture):
        command.add_argument("--robot-pose", required=True, help="含T_base_from_end的实际FK结果JSON")
        command.add_argument("--validation", action="store_true", help="仅用于验证，不参与手眼拟合")
    for command in (board, add, capture):
        command.add_argument(
            "--reverse-corners", action="store_true", help="检测起点与固定物理原点相反时反转角点顺序"
        )
        command.add_argument("--max-rmse-px", type=float, default=2.0)
    solve = commands.add_parser("solve", help="由至少12组训练配对计算手眼矩阵")
    solve.add_argument("--output", default="results/task2")
    solve.add_argument(
        "--board-origin-m",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="板原点在基座系中的独立实测位置，单位米",
    )
    fk = commands.add_parser("fk", help="使用配置的FK后端，将实测关节状态转成末端位姿")
    fk.add_argument("--model", default="configs/robot_model.json")
    fk.add_argument("--observation", required=True, help="包含*.pos的实测关节观测JSON")
    fk.add_argument("--output", default="data/task2/robot_pose.json")
    args = parser.parse_args()
    directory = resolve(args.directory)

    if args.command == "init":
        data = create_session(directory, resolve(args.intrinsics))
        # SO关节名称是仓库中的模板；URDF、零位和符号必须按实物填写。
        model = {
            "backend": "portable_urdf",
            "urdf_path": None,
            "joint_names": SO_JOINTS,
            "end_frame": "gripper_frame_link",
            "base_frame": None,
            "joint_limits_deg": dict.fromkeys(SO_JOINTS),
            "motor_ids": dict(zip(SO_JOINTS, range(1, 6), strict=True)),
            "motor_model": "scs215",
            "motor_calibration_path": None,
            "port": "COM3",
            "baudrate": 1000000,
            "observation_units": "normalized_0_1",
            "normalized_joint_calibration": {
                name: {"q_at_0_deg": None, "q_at_1_deg": None} for name in SO_JOINTS
            },
            "note": "模板需按实际URDF和关节标定填写；夹爪是否进入运动链依据相机安装和URDF帧决定",
        }
        (directory / "robot_model.json").write_text(
            json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "directory": str(directory),
                    "board": data["intrinsics"]["board_inner_corners"],
                    "image_size": data["intrinsics"]["image_size"],
                    "pose_pairs": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "preflight":
        model_path = resolve(args.model)
        raw_model = json.loads(model_path.read_text(encoding="utf-8"))
        problems = []
        try:
            RobotModel(model_path, REPOSITORY)
        except (ValueError, KeyError, ImportError) as error:
            problems.append(str(error))
        if raw_model.get("observation_units") == "normalized_0_1":
            try:
                joint_positions_in_degrees(
                    {f"{name}.pos": 0.5 for name in raw_model["joint_names"]},
                    raw_model["joint_names"],
                    "normalized_0_1",
                    raw_model.get("normalized_joint_calibration"),
                )
            except (KeyError, ValueError, TypeError) as error:
                problems.append(str(error))
        elif raw_model.get("observation_units") != "degrees":
            problems.append("observation_units 必须明确为degrees或normalized_0_1")
        print(
            json.dumps(
                {
                    "ready_for_repository_fk": not problems,
                    "missing_configuration": problems,
                    "fk_source": "src/lerobot/model/kinematics.py",
                    **session_status(directory),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "fk":
        model = RobotModel(resolve(args.model), REPOSITORY)
        observation = json.loads(resolve(args.observation).read_text(encoding="utf-8"))
        observation = observation.get("observation", observation)
        pose = model.pose(observation)
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(pose, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(pose, ensure_ascii=False, indent=2))
    elif args.command == "board-pose":
        data = load_session(directory)
        estimate, overlay = estimate_board_pose(
            read_image(resolve(args.image)), data["intrinsics"], args.reverse_corners, args.max_rmse_px
        )
        write_image(resolve(args.output), overlay)
        print(json.dumps({**estimate, "overlay": str(resolve(args.output))}, ensure_ascii=False, indent=2))
    elif args.command in {"add", "capture"}:
        pose_path = resolve(args.robot_pose)
        pose = read_robot_pose(pose_path)
        metadata = None
        if args.command == "capture":
            calibration = load_session(directory)["intrinsics"]
            started = datetime.now(UTC).isoformat()
            images = capture_images(
                args.camera, directory / "staging", *calibration["image_size"], count=1, interval_s=0
            )
            after = read_robot_pose(pose_path)
            first, second = np.asarray(pose["T_base_from_end"]), np.asarray(after["T_base_from_end"])
            if np.linalg.norm(first[:3, 3] - second[:3, 3]) > 0.001 or rotation_error_rad(
                first[:3, :3], second[:3, :3]
            ) > np.deg2rad(0.5):
                raise ValueError("拍照前后FK文件显示位姿变化，请停稳后重拍")
            metadata = {
                "started_utc": started,
                "finished_utc": datetime.now(UTC).isoformat(),
                "pose_file_matrix_unchanged_within_tolerance": True,
                "automatic_live_robot_feedback": False,
                "note": "FK文件检查不能证明实物未运动；拍照期间需保持机械臂静止",
            }
            image = images[0]
        else:
            image = resolve(args.image)
        pair = append_pair(
            directory,
            image,
            pose,
            validation=args.validation,
            reverse_corners=args.reverse_corners,
            max_rmse_px=args.max_rmse_px,
            capture_metadata=metadata,
        )
        print(
            json.dumps(
                {
                    "saved_pair": pair["id"],
                    "board_rmse_px": pair["board_rmse_px"],
                    "overlay": str(directory / pair["overlay"]),
                    **session_status(directory),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "solve":
        result = solve_session(directory, resolve(args.output), args.board_origin_m)
        print(
            json.dumps(
                {
                    "T_end_from_camera": result["T_end_from_camera"],
                    "camera_origin_in_end_m": result["camera_origin_in_end_m"],
                    "board_origin_spread_rms_m": result["board_origin_spread_rms_m"],
                    "independent_board_measurement": result["independent_board_measurement"],
                    "validation_pairs": len(result["validation"]),
                    "output": str(resolve(args.output)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(json.dumps(session_status(directory), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, KeyError, ImportError) as error:
        print(f"任务2执行失败：{error}")
        if "could not be opened" in str(error):
            print("相机可能仍被任务1预览占用；独立拍照需要先退出该采集进程。")
        raise SystemExit(2) from None
