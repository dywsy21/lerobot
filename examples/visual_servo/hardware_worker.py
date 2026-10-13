"""Isolated serial owner with a watchdog; newline JSON on stdin/stdout."""

import argparse
import json
import queue
import sys
import threading
import time

from visual_servo_week6.hardware import FeetechAdapter
from visual_servo_week6.robot_model import RobotModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--repository")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        import scservo_sdk

        print(
            json.dumps(
                {
                    "ok": True,
                    "sdk_loaded": bool(scservo_sdk),
                    "torch_loaded": "torch" in sys.modules,
                    "hardware_opened": False,
                }
            ),
            flush=True,
        )
        return 0
    model = RobotModel(args.model, args.repository)
    adapter = FeetechAdapter(model, args.repository)
    inbox = queue.Queue()

    def reader():
        for line in sys.stdin:
            inbox.put(line)
        inbox.put(None)

    threading.Thread(target=reader, daemon=True).start()
    active_at = None
    try:
        while True:
            try:
                line = inbox.get(timeout=0.1)
            except queue.Empty:
                if active_at is not None and time.monotonic() - active_at > 1.5:
                    adapter.hold()
                    active_at = None
                continue
            if line is None:
                break
            try:
                data = json.loads(line)
                command, arguments = data["command"], data.get("arguments", [])
                if command not in {"connect", "observe", "arm", "move", "hold", "teach", "close"}:
                    raise ValueError("Unknown hardware operation")
                result = getattr(adapter, command)(*arguments)
                if command in {"arm", "move"}:
                    active_at = time.monotonic()
                elif command in {"hold", "teach", "close"}:
                    active_at = None
                print(
                    json.dumps({"ok": True, "result": result, "armed": adapter.armed}, allow_nan=False),
                    flush=True,
                )
                if command == "close":
                    break
            except Exception as error:
                try:
                    adapter.hold()
                except Exception as hold_error:
                    print(f"Hold failed: {hold_error}", file=sys.stderr, flush=True)
                active_at = None
                print(json.dumps({"ok": False, "error": str(error), "armed": adapter.armed}), flush=True)
    finally:
        if adapter.bus is not None:
            adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
