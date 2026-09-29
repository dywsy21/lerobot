"""Continuous calibrated travel over the SCS215 potentiometer discontinuity.

The native position loop does not wrap. A crossing therefore approaches the
edge in position mode, traverses it with directional PWM, then hands back to
the native loop once feedback has entered the other recorded interval.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SeamCommand:
    kind: Literal["position", "enter_pwm", "pwm", "exit_pwm"]
    value: int


class Scs215SeamController:
    def __init__(self, segments: list[tuple[int, int]], pwm: int = 180, timeout: float = 3.0):
        self.segments = segments
        self.base_pwm = pwm
        self.timeout = timeout
        self.phase = "position"
        self.direction = 0
        self.destination = 0
        self.started = 0.0
        self.last_raw = 0
        self.last_goal: int | None = None
        self.last_pwm: int | None = None
        self.crossed = False
        self.stable = 0
        self.reverse_travel = 0

    def segment(self, raw: int, nearest: bool = False) -> int:
        for index, (start, end) in enumerate(self.segments):
            if start <= raw <= end:
                return index
        if nearest:
            return min(
                range(len(self.segments)),
                key=lambda index: min(abs(raw - endpoint) for endpoint in self.segments[index]),
            )
        raise ValueError(f"Position {raw} is outside the recorded SCS215 intervals.")

    def _pwm(self, elapsed: float) -> int:
        magnitude = min(400, self.base_pwm + 40 * int(elapsed / 0.4))
        # SCS215 raw position increases clockwise. In PWM mode bit 10 selects CW.
        return magnitude | (1024 if self.direction > 0 else 0)

    def step(self, raw: int, target: int, now: float) -> SeamCommand | None:
        if not 0 <= raw <= 1023 or not 0 <= target <= 1023:
            raise ValueError("SCS215 positions must be within 0..1023.")
        target_segment = self.segment(target)

        if self.phase == "bridge":
            delta = raw - self.last_raw
            if delta * self.direction < -512:
                self.crossed = True
                self.reverse_travel = 0
            elif delta * self.direction > 512:
                # A boundary sample can bounce back once; require re-entry.
                self.crossed = False
                self.stable = 0
            elif delta * self.direction < -2:
                self.reverse_travel -= delta * self.direction
            elif delta * self.direction > 0:
                self.reverse_travel = max(0, self.reverse_travel - delta * self.direction)
            self.last_raw = raw
            if self.reverse_travel > 24:
                raise RuntimeError("SCS215 PWM direction disagrees with position feedback.")

            start, end = self.segments[self.destination]
            margin = min(8, (end - start) // 2)
            inside = start <= raw <= end and (
                raw >= start + margin if self.direction > 0 else raw <= end - margin
            )
            self.stable = self.stable + 1 if self.crossed and inside else 0
            if self.stable >= 2:
                self.phase = "position"
                # A new target may arrive during the crossing. Finish this
                # handover first; the next step can plan a crossing back.
                goal = target if target_segment == self.destination else raw
                self.last_goal = goal
                return SeamCommand("exit_pwm", goal)

            elapsed = now - self.started
            if elapsed >= self.timeout:
                raise TimeoutError("SCS215 did not regain position feedback after crossing.")
            pwm = self._pwm(elapsed)
            if pwm != self.last_pwm:
                self.last_pwm = pwm
                return SeamCommand("pwm", pwm)
            return None

        source = self.segment(raw, nearest=True)
        if source == target_segment:
            self.phase = "position"
            if target != self.last_goal:
                self.last_goal = target
                return SeamCommand("position", target)
            return None

        direction = 1 if source < target_segment else -1
        if self.phase != "approach" or direction != self.direction:
            self.phase = "approach"
            self.started = now
        self.direction = direction
        self.destination = target_segment
        entry = max(self.segments[source][0], 1011) if direction > 0 else min(self.segments[source][1], 12)
        # The native position loop can settle several counts short of its goal.
        # PWM takes over in this small entry window.
        if (raw - entry) * direction >= -12:
            self.phase = "bridge"
            self.started = now
            self.last_raw = raw
            self.crossed = False
            self.stable = 0
            self.reverse_travel = 0
            self.last_pwm = self._pwm(0)
            return SeamCommand("enter_pwm", self.last_pwm)
        if now - self.started >= 10:
            raise TimeoutError("SCS215 did not reach the crossing entry position.")
        if self.last_goal != entry:
            self.last_goal = entry
            return SeamCommand("position", entry)
        return None
