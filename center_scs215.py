#!/usr/bin/env python
"""Move one SCS215 to raw position 512, without calibration or seam bridging."""

import argparse
import math
import sys
import time
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _feetech_sdk_available, require_package

if TYPE_CHECKING or _feetech_sdk_available:
    import scservo_sdk as scs
else:
    scs = None

TARGET = 512
TOLERANCE = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a single unicast ID and connection options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, required=True, help="Servo ID (0..253); only this ID is moved")
    parser.add_argument("--port", default="COM5", help="Serial port (default: COM5)")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--speed", type=int, default=200, help="Native speed setting, 1..1000 (default: 200)")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Arrival timeout in seconds (default: 10)"
    )
    parser.add_argument(
        "--release", action="store_true", help="Release torque after arrival instead of holding"
    )
    args = parser.parse_args(argv)
    if not 0 <= args.id <= 253:
        parser.error("--id must be 0..253; broadcast IDs are not allowed")
    if not 1 <= args.speed <= 1000:
        parser.error("--speed must be 1..1000")
    if args.baudrate <= 0:
        parser.error("--baudrate must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    return args


class Servo:
    """Checked, single-ID access to the SCS protocol-1 registers."""

    def __init__(self, port, packet, motor_id: int):
        """Use an already-open port and a protocol-1 packet handler."""
        self.port = port
        self.packet = packet
        self.motor_id = motor_id

    def _check(self, comm: int, error: int) -> None:
        if comm != scs.COMM_SUCCESS:
            raise RuntimeError(self.packet.getTxRxResult(comm))
        if error:
            raise RuntimeError(f"ID {self.motor_id}: servo error 0x{error:02x}")

    def read(self, address: int, size: int = 2) -> int:
        """Read one register and reject communication or device errors."""
        read = self.packet.read1ByteTxRx if size == 1 else self.packet.read2ByteTxRx
        value, comm, error = read(self.port, self.motor_id, address)
        self._check(comm, error)
        return value

    def write(self, address: int, value: int, size: int = 2) -> None:
        """Write one register without broadcasting or using normalization."""
        write = self.packet.write1ByteTxRx if size == 1 else self.packet.write2ByteTxRx
        comm, error = write(self.port, self.motor_id, address, value)
        self._check(comm, error)

    def position(self) -> int:
        """Read a valid raw encoder sample."""
        raw = self.read(56)
        if not 0 <= raw <= 1023:
            raise RuntimeError(f"Invalid SCS215 position: {raw}")
        return raw


def center_servo(servo: Servo, speed: int, timeout: float, release: bool) -> int:
    """Command raw 512, verify feedback, and keep holding unless release is set."""
    model = servo.read(3)
    if model != 1315:
        raise RuntimeError(f"ID {servo.motor_id}: model {model}, expected SCS215 (1315)")
    voltage = servo.read(62, 1) / 10
    initial = servo.position()
    low, high = servo.read(9), servo.read(11)
    motor_mode = low == high == 0
    if not motor_mode and not low <= TARGET <= high:
        raise RuntimeError(f"Position 512 is outside the device's limits [{low}, {high}]")
    print(f"ID {servo.motor_id}: position={initial}, voltage={voltage:.1f} V, target={TARGET}")

    try:
        # Prime the target with output disabled, so an old goal cannot run first.
        servo.write(40, 0, 1)
        servo.write(44, 0)
        servo.write(46, speed)
        servo.write(42, TARGET)
        if motor_mode:
            # Recover from an earlier PWM session. Lock=1 keeps limits in RAM;
            # this does not rewrite the servo's persistent EEPROM calibration.
            lock = servo.read(48, 1)
            try:
                servo.write(48, 1, 1)
                servo.write(11, 1023)
                if servo.read(11) != 1023:
                    raise RuntimeError("Could not restore native position mode")
            finally:
                servo.write(48, lock, 1)
        servo.write(40, 1, 1)

        deadline = time.monotonic() + timeout
        stable = 0
        position = initial
        while time.monotonic() < deadline:
            position = servo.position()
            print(f"\rID {servo.motor_id}: {position:4d} -> {TARGET}", end="", flush=True)
            stable = stable + 1 if abs(position - TARGET) <= TOLERANCE else 0
            if stable >= 3:
                if release:
                    servo.write(40, 0, 1)
                state = "released" if release else "holding"
                print(
                    f"\nDone: target={TARGET}, actual={position}, tolerance=+/-{TOLERANCE}; torque {state}."
                )
                return position
            time.sleep(0.05)
        raise TimeoutError(f"Did not reach 512 +/- {TOLERANCE} in {timeout:g}s; last position={position}")
    except BaseException:
        # Address only this motor, including when Ctrl+C interrupts a movement.
        for address, value, size in ((40, 0, 1), (44, 0, 2)):
            try:
                servo.write(address, value, size)
            except Exception as exc:
                print(f"\nCould not stop ID {servo.motor_id}, register {address}: {exc}", file=sys.stderr)
        raise


def main(argv: list[str] | None = None) -> int:
    """Open the requested port, run the command, and always close the port."""
    args = parse_args(argv)
    require_package("feetech-servo-sdk", extra="feetech", import_name="scservo_sdk")
    port = scs.PortHandler(args.port)
    try:
        if not port.openPort() or not port.setBaudRate(args.baudrate):
            raise RuntimeError(f"Could not open {args.port} at {args.baudrate} baud")
        # Allow for Windows USB serial latency without unbounded packet waits.
        port.setPacketTimeout = lambda _length: port.setPacketTimeoutMillis(100)
        servo = Servo(port, scs.PacketHandler(1), args.id)
        center_servo(servo, args.speed, args.timeout, args.release)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted; stop requested for the selected servo.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        print("If the port is busy, disconnect the browser's serial connection first.", file=sys.stderr)
        return 1
    finally:
        port.closePort()


if __name__ == "__main__":
    raise SystemExit(main())
