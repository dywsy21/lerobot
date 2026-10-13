r"""任务 1 的教学入口：拍照、采集棋盘格、检查角点、计算相机内参。

使用本项目的 .venv\Scripts\python.exe 运行。实际拍照与标定算法分别在
src/visual_servo_week6/camera.py 和 calibration.py，入口只负责组织步骤。
"""

import argparse
import json
from pathlib import Path

import cv2
from visual_servo_week6.calibration import calibrate_images, find_board, save_yaml
from visual_servo_week6.camera import capture_images
from visual_servo_week6.image_io import read_image, write_image

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="任务1：相机拍照与棋盘格内参标定")
    commands = parser.add_subparsers(dest="command", required=True)

    photo = commands.add_parser("photo", help="先拍一张普通照片，确认相机与视野")
    collect = commands.add_parser("collect", help="只保存完整检测到棋盘格的照片")
    for command in (photo, collect):
        command.add_argument(
            "--camera", type=int, required=True, help="本次验证外接相机为2；重插后需重新核对"
        )
        command.add_argument("--width", type=int, default=1280)
        command.add_argument("--height", type=int, default=720)
    photo.add_argument("--output", default="outputs/task1/photo-check")
    collect.add_argument("--output", default="data/task1/train")
    collect.add_argument(
        "--board",
        nargs=2,
        type=int,
        required=True,
        metavar=("NX", "NY"),
        help="横向和纵向内角点数量，不是格子数量",
    )
    collect.add_argument("--count", type=int, default=20)
    collect.add_argument("--interval", type=float, default=5.0, help="保存后留给你改变棋盘姿态的秒数")

    check = commands.add_parser("check", help="在一张照片上画出检测到的内角点")
    check.add_argument("--image", required=True)
    check.add_argument("--board", nargs=2, type=int, required=True, metavar=("NX", "NY"))
    check.add_argument("--output", default="outputs/task1/corners-check.png")

    calibration = commands.add_parser("calibrate", help="由至少15张有效棋盘格照片计算K与畸变")
    calibration.add_argument("--images", default="data/task1/train")
    calibration.add_argument("--board", nargs=2, type=int, required=True, metavar=("NX", "NY"))
    calibration.add_argument(
        "--square-size-m", type=float, default=0.025, help="课程单格边长25mm，转换为0.025米"
    )
    calibration.add_argument("--output", default="outputs/task1/intrinsics.yaml")
    args = parser.parse_args()

    if args.command in {"photo", "collect"}:
        # 普通拍照不检查棋盘；collect只有完整检测到指定内角点时才保存。
        is_collecting = args.command == "collect"
        paths = capture_images(
            args.camera,
            ROOT / args.output,
            width=args.width,
            height=args.height,
            count=args.count if is_collecting else 1,
            interval_s=args.interval if is_collecting else 0,
            board=tuple(args.board) if is_collecting else None,
        )
        print(json.dumps({"saved_images": paths}, ensure_ascii=False, indent=2))
    elif args.command == "check":
        # 角点的顺序必须和标定板上的三维点一一对应。
        image = read_image(ROOT / args.image)
        if image is None:
            raise ValueError("照片读取失败，请检查 --image 路径")
        corners = find_board(image, tuple(args.board))
        if corners is None:
            print("没有找到完整棋盘格。检查内角点数量、视野、反光与清晰度。")
            return 2
        cv2.drawChessboardCorners(image, tuple(args.board), corners, True)
        destination = ROOT / args.output
        write_image(destination, image)
        print(f"找到 {len(corners)} 个内角点；检查图保存到 {destination}")
    else:
        directory = ROOT / args.images
        paths = sorted(
            path for path in directory.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        # calibrate_images内部完成：三维点生成 → 角点检测 → calibrateCamera → 重投影误差。
        result, _ = calibrate_images(paths, tuple(args.board), args.square_size_m)
        result["data_source"] = (
            "provided_images; use photographs of the actual camera for physical experiments"
        )
        destination = ROOT / args.output
        save_yaml(destination, result)
        print(
            json.dumps(
                {
                    "valid_images": result["valid_images"],
                    "rms_px": result["rms_px"],
                    "K": result["K"],
                    "distortion": result["distortion"],
                    "saved_to": str(destination),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if result["rms_px"] > 1.0:
            print("重投影RMS超过1像素，请查看逐张误差并补充清晰、不同位置和倾角的照片。")
        else:
            print("训练照片重投影RMS不超过1像素；仍需确认姿态多样性，并用新照片检查结果。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"执行失败：{error}")
        if "could not be opened" in str(error):
            print("确认相机编号，并退出占用相机的采集程序后重试。暂停网页采集或关闭网页标签仍会占用相机。")
        raise SystemExit(2) from None
