"""Unified Week 6 entry; every path resolves independently of the shell cwd."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from visual_servo_week6.experiment_runtime import load_configuration, preflight
from visual_servo_week6.experiment_web import serve
from visual_servo_week6.verification import verify_software

ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) > 1 and sys.argv[1] in {"task1", "task2", "task3"}:
        task = sys.argv.pop(1)
        module = __import__(task)
        return module.main()
    parser = argparse.ArgumentParser(description="课堂任务1/2/3统一软件入口")
    parser.add_argument("--config", default=str(ROOT / "configs/experiment.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="检查实物参数是否齐全，不连接或移动硬件")
    verify = commands.add_parser("verify", help="测试＋任务2合成配对＋任务3视频仿真")
    verify.add_argument("--output", default=None)
    web = commands.add_parser("web", help="启动统一网页控制台")
    web.add_argument("--source", choices=["simulation", "camera", "broker"], default="simulation")
    web.add_argument("--camera", type=int, default=2)
    web.add_argument("--broker", default="http://127.0.0.1:8766")
    web.add_argument("--port", type=int, default=8767)
    web.add_argument("--allow-hardware", action="store_true")
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(preflight(args.config), ensure_ascii=False, indent=2))
    elif args.command == "verify":
        config = load_configuration(args.config)
        output = args.output or ROOT / "outputs/verification" / datetime.now().strftime("%Y%m%d-%H%M%S")
        print(json.dumps(verify_software(ROOT, config["intrinsics"], output), ensure_ascii=False, indent=2))
    else:
        serve(args.config, args.source, args.port, args.camera, args.broker, args.allow_hardware)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
        print(f"实验启动失败：{error}")
        raise SystemExit(2) from None
