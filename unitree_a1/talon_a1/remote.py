"""Wireless remote: velocity command from the sticks and edge-triggered button combos."""

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
