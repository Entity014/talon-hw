"""Tests for action EMA smoothing in Controller."""

import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from talon_a1 import contract
from talon_a1.controller import Controller, STAND, WALK, START_WALK


class DummyDriver:
    def __init__(self):
        self.target = None
        self.kp = 0.0
        self.kd = 0.0

    def set_target(self, q, kp, kd):
        self.target = q
        self.kp = kp
        self.kd = kd
        return True

    def disarm(self):
        pass

    def clear_faults(self):
        pass


class DummyPolicy:
    obs_dim = 48
    act_dim = 12
    hist_len = 5
    action_scale = 0.25
    kp = 55.0
    meta = {}

    def __init__(self, fixed_action=None):
        self.fixed_action = fixed_action if fixed_action is not None else np.ones(12, dtype=np.float32)

    def action_mean(self, obs, hist, w=None):
        return self.fixed_action.copy(), np.zeros(3, dtype=np.float32), np.zeros(8, dtype=np.float32)


class DummyRemote:
    def __init__(self):
        self._keys = 0

    def update(self, state):
        pass

    def pressed(self, key):
        return (self._keys & key) != 0

    def command(self, state):
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)


class DummyState:
    faults = 0
    tick = 1
    gyro = np.zeros(3, dtype=np.float32)
    acc = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rpy = np.zeros(3, dtype=np.float32)
    q = contract.DEFAULT_JOINT_POS.copy()
    dq = np.zeros(12, dtype=np.float32)
    tau_est = np.zeros(12, dtype=np.float32)
    keys = 0
    lx = ly = rx = ry = 0.0


def test_action_smooth_zero_preserves_raw_action():
    drv = DummyDriver()
    pol = DummyPolicy(fixed_action=np.ones(12, dtype=np.float32) * 2.0)
    rem = DummyRemote()
    ctl = Controller(drv, pol, remote=rem, soft_start=0.0, action_clip=10.0, action_smooth=0.0)

    # Put in WALK mode
    ctl.mode = WALK
    state = DummyState()
    rec = ctl.step(state)

    # With smooth=0.0 and soft_start=0.0, action should be exactly 2.0
    assert np.allclose(rec["action"], 2.0)
    assert np.allclose(ctl.last_action, 2.0)


def test_action_smooth_ema_step_response():
    drv = DummyDriver()
    target_action = np.ones(12, dtype=np.float32) * 2.0
    pol = DummyPolicy(fixed_action=target_action)
    rem = DummyRemote()
    alpha = 0.3
    ctl = Controller(drv, pol, remote=rem, soft_start=0.0, action_clip=10.0, action_smooth=alpha)

    ctl.mode = WALK
    ctl.last_action[:] = 0.0
    state = DummyState()

    # Step 1: last_action was 0, new action is 2.0
    # Expected: (1 - 0.3) * 2.0 + 0.3 * 0.0 = 1.4
    rec1 = ctl.step(state)
    assert np.allclose(rec1["action"], 1.4)
    assert np.allclose(ctl.last_action, 1.4)

    # Step 2: last_action is 1.4, new action is 2.0
    # Expected: (1 - 0.3) * 2.0 + 0.3 * 1.4 = 1.4 + 0.42 = 1.82
    rec2 = ctl.step(state)
    assert np.allclose(rec2["action"], 1.82)
    assert np.allclose(ctl.last_action, 1.82)
