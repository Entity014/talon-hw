"""Wireless remote: velocity command from the sticks and edge-triggered button combos."""

import math
import time

import numpy as np

from . import contract

# xKeySwitchUnion bits (same as _a1_driver.KEYS; kept here so the module imports without the driver)
KEYS = {
    "R1": 1 << 0, "L1": 1 << 1, "START": 1 << 2, "SELECT": 1 << 3, "R2": 1 << 4, "L2": 1 << 5,
    "F1": 1 << 6, "F2": 1 << 7, "A": 1 << 8, "B": 1 << 9, "X": 1 << 10, "Y": 1 << 11,
    "UP": 1 << 12, "RIGHT": 1 << 13, "DOWN": 1 << 14, "LEFT": 1 << 15,
}


def combo(*names):
    mask = 0
    for n in names:
        mask |= KEYS[n]
    return mask


class Remote(object):
    def __init__(self, vx_max=0.5, vy_max=0.3, wz_max=0.5, deadzone=0.08):
        lim = contract.COMMAND_LIMITS
        self.max = np.array(
            [min(vx_max, lim[0][1]), min(vy_max, lim[1][1]), min(wz_max, lim[2][1])], dtype=np.float32
        )
        self.deadzone = deadzone
        self.prev_keys = 0
        self.keys = 0

    def update(self, state):
        self.prev_keys, self.keys = self.keys, int(state.keys)

    def pressed(self, mask):
        """True on the frame every key in mask is down and at least one of them was not."""
        return (self.keys & mask) == mask and (self.prev_keys & mask) != mask

    def _axis(self, v):
        v = float(v)
        if abs(v) < self.deadzone:
            return 0.0
        return max(-1.0, min(1.0, v))

    def command(self, state):
        """(vx, vy, wz): left stick forward/left, right stick yaw left. Stick right is positive."""
        return self.max * np.array(
            [self._axis(state.ly), -self._axis(state.lx), -self._axis(state.rx)], dtype=np.float32
        )


def _trapezoid_profile(dist, vx, ramp):
    """Return (t_up, t_cruise, t_down, v_peak) for a trapezoidal (or triangular) velocity profile covering `dist`."""
    dist = max(0.0, float(dist))
    vx = float(vx)
    ramp = max(0.0, float(ramp))
    if dist == 0.0 or vx == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    sign = 1.0 if vx > 0 else -1.0
    speed = min(abs(vx), abs(float(contract.COMMAND_LIMITS[0][1])))
    if ramp == 0.0:
        return 0.0, dist / speed, 0.0, sign * speed
    if dist >= speed * ramp:
        t_cruise = (dist - speed * ramp) / speed
        return ramp, t_cruise, ramp, sign * speed
    # Triangular profile when distance is shorter than a full ramp-up + ramp-down
    v_peak = math.sqrt(dist * speed / ramp)
    t_half = dist / v_peak
    return t_half, 0.0, t_half, sign * v_peak


class MacroRemote(Remote):
    """Remote with a joystick-triggered full sequence macro (default: R1+X = stand -> walk 3.0 m @ vx=0.6 m/s -> stop -> sit).

    While the macro is idle, behaves identically to Remote (manual sticks respect vx_max/vy_max/wz_max).
    Pressing `start_combo` (R1+X) runs:
        STAND_UP (L1+A) -> wait stand_time -> START_WALK (L1+X) -> soft_start settle
        -> walk macro_dist @ macro_vx -> settle -> STOP_WALK (L1+Y) -> wait 1.2 s -> TO_PASSIVE (L1+B).
    Pressing `stop_combo` (R1+Y), L1+Y, L1+B, moving any stick outside deadzone, or a fault cancels immediately.
    """

    _IDLE = "IDLE"
    _STAND_UP = "STAND_UP"
    _START_WALK = "START_WALK"
    _WALK = "WALK"
    _SETTLE = "SETTLE"
    _STOP_WALK = "STOP_WALK"

    def __init__(
        self,
        vx_max=0.5,
        vy_max=0.3,
        wz_max=0.5,
        deadzone=0.08,
        macro_vx=0.6,
        macro_dist=3.0,
        macro_ramp=0.5,
        stand_time=3.0,
        soft_start=1.0,
        settle_time=0.5,
        start_combo=None,
        stop_combo=None,
        clock=None,
    ):
        super().__init__(vx_max=vx_max, vy_max=vy_max, wz_max=wz_max, deadzone=deadzone)
        self.macro_vx = float(macro_vx)
        self.macro_dist = float(macro_dist)
        self.macro_ramp = float(macro_ramp)
        self.stand_time = float(stand_time)
        self.soft_start = float(soft_start)
        self.settle_time = float(settle_time)
        self.start_combo = combo("R1", "X") if start_combo is None else start_combo
        self.stop_combo = combo("R1", "Y") if stop_combo is None else stop_combo
        self._cancel_combos = (self.stop_combo, combo("L1", "Y"), combo("L1", "B"))
        self._clock = time.monotonic if clock is None else clock
        self._t_up, self._t_cruise, self._t_down, self._v_peak = _trapezoid_profile(
            self.macro_dist, self.macro_vx, self.macro_ramp
        )
        self.macro_duration = self._t_up + self._t_cruise + self._t_down
        self._phase = self._IDLE
        self._phase_t0 = 0.0
        self._raw_prev_keys = 0
        self._raw_keys = 0

    @property
    def macro_active(self):
        return self._phase != self._IDLE

    def _raw_pressed(self, mask):
        return (self._raw_keys & mask) == mask and (self._raw_prev_keys & mask) != mask

    def _stick_moved(self, state):
        # Use a 0.25 threshold so minor physical stick drift does not accidentally cancel the macro.
        thresh = max(0.25, self.deadzone)
        return (
            abs(float(state.ly)) >= thresh
            or abs(float(state.lx)) >= thresh
            or abs(float(state.rx)) >= thresh
        )

    def _macro_vx_at(self, t):
        if t < 0.0 or t >= self.macro_duration:
            return 0.0
        if self._t_up > 0.0 and t < self._t_up:
            return self._v_peak * (t / self._t_up)
        t_after_up = t - self._t_up
        if t_after_up < self._t_cruise:
            return self._v_peak
        t_in_down = t_after_up - self._t_cruise
        if self._t_down > 0.0 and t_in_down < self._t_down:
            return self._v_peak * (1.0 - t_in_down / self._t_down)
        return 0.0

    def _enter(self, phase, now):
        self._phase = phase
        self._phase_t0 = now

    def update(self, state):
        raw = int(state.keys)
        self._raw_prev_keys, self._raw_keys = self._raw_keys, raw
        now = self._clock()
        inj = 0

        if self._phase != self._IDLE:
            faults = int(getattr(state, "faults", 0))
            if faults or any(self._raw_pressed(c) for c in self._cancel_combos) or self._stick_moved(state):
                self._phase = self._IDLE
                print("[talon_a1] macro cancelled")
                self.prev_keys, self.keys = self.keys, raw
                return

            dt_phase = now - self._phase_t0
            if self._phase == self._STAND_UP and dt_phase >= self.stand_time + 0.25 - 1e-6:
                inj = combo("L1", "X")
                self._enter(self._START_WALK, now)
            elif self._phase == self._START_WALK and dt_phase >= max(0.2, self.soft_start) - 1e-6:
                self._enter(self._WALK, now)
            elif self._phase == self._WALK and dt_phase >= self.macro_duration - 1e-6:
                self._enter(self._SETTLE, now)
            elif self._phase == self._SETTLE and dt_phase >= self.settle_time - 1e-6:
                inj = combo("L1", "Y")
                self._enter(self._STOP_WALK, now)
            elif self._phase == self._STOP_WALK and dt_phase >= 1.2 - 1e-6:
                inj = combo("L1", "B")
                self._phase = self._IDLE
                print("[talon_a1] macro complete (sitting down)")
        elif self._raw_pressed(self.start_combo) and self.macro_duration > 0.0 and not int(getattr(state, "faults", 0)):
            inj = combo("L1", "A")
            self._enter(self._STAND_UP, now)
            print(
                "[talon_a1] macro start: stand -> walk %.2f m @ vx=%.2f m/s (%.2f s) -> stop -> sit"
                % (self.macro_dist, self._v_peak, self.macro_duration)
            )

        self.prev_keys, self.keys = self.keys, raw | inj

    def command(self, state):
        if self._phase == self._WALK:
            t = self._clock() - self._phase_t0
            vx = self._macro_vx_at(t)
            lo, hi = contract.COMMAND_LIMITS[0]
            vx = max(float(lo), min(float(hi), vx))
            return np.array([vx, 0.0, 0.0], dtype=np.float32)
        if self._phase != self._IDLE:
            return np.zeros(3, dtype=np.float32)
        return super().command(state)
