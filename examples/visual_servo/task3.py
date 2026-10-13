"""Task 3 entry: offline simulation, photograph analysis or shared web console."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from visual_servo_week6.experiment_runtime import load_configuration
from visual_servo_week6.experiment_web import serve
from visual_servo_week6.image_io import read_image, write_image
from visual_servo_week6.task3_core import VisualServo, load_intrinsics
from visual_servo_week6.task3_simulation import annotate, run_task3_simulation

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="任务3：ArUco 视觉伺服")
    parser.add_argument("--config", default=str(ROOT / "configs/experiment.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    simulation = commands.add_parser("simulate", help="仿真闭环、遮挡测试并导出视频和报告")
    simulation.add_argument("--output", default=None)
    image = commands.add_parser("image", help="从照片检测标记并计算相机速度建议")
    image.add_argument("--image", required=True)
    image.add_argument("--output", default=str(ROOT / "outputs/task3/image-analysis"))
    web = commands.add_parser("web", help="启动任务2/3共享网页")
    web.add_argument("--source", choices=["simulation", "camera", "broker"], default="simulation")
    web.add_argument("--camera", type=int, default=2)
    web.add_argument("--broker", default="http://127.0.0.1:8766")
    web.add_argument("--port", type=int, default=8767)
    web.add_argument("--allow-hardware", action="store_true")
    args = parser.parse_args()
    config = load_configuration(args.config)
    if args.command == "simulate":
        output = args.output or ROOT / "outputs/task3" / datetime.now().strftime("%Y%m%d-%H%M%S")
        run_task3_simulation(config["intrinsics"], output)
    elif args.command == "image":
        matrix, distortion, size, _ = load_intrinsics(config["intrinsics"])
        servo = VisualServo(matrix, distortion, size, config["settings"])
        frame = read_image(args.image)
        if frame is None:
            raise ValueError("图像无法读取")
        advice = servo.evaluate(frame)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        write_image(output / "overlay.png", annotate(frame, advice, size, "ADVISORY ONLY"))
        (output / "advice.json").write_text(json.dumps(advice, indent=2), encoding="utf-8")
        print(json.dumps(advice, ensure_ascii=False, indent=2))
    else:
        serve(args.config, args.source, args.port, args.camera, args.broker, args.allow_hardware)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
        print(f"任务3执行失败：{error}")
        raise SystemExit(2) from None
