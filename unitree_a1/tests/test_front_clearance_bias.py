"""Tests for pitch_bias and front_lift_bias compensation in Controller."""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from talon_a1 import contract
from talon_a1.controller import Controller, WALK


class DummyDriver:
    def __init__(self):
        self.target = None

    def set_target(self, q, kp, kd):
        self.target = np.asarray(q, dtype=np.float32)
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

    def action_mean(self, obs, hist, w=None):
        return np.zeros(12, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(8, dtype=np.float32)


class DummyRemote:
    def __init__(self, cmd):
        self.cmd = np.asarray(cmd, dtype=np.float32)
        self.vx_max = 0.5

    def update(self, state):
        pass

    def pressed(self, key):
        return False

    def command(self, state):
        return self.cmd


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


def test_pitch_bias_flexes_rear_calves():
    drv = DummyDriver()
    ctl = Controller(drv, DummyPolicy(), remote=DummyRemote([0.0, 0.0, 0.0]),
                     soft_start=0.0, pitch_bias=0.10)
    ctl.mode = WALK
    rec = ctl.step(DummyState())
    # Policy order indices 10, 11 are RL_calf, RR_calf
    assert np.allclose(rec["q_target"][10:12], contract.DEFAULT_JOINT_POS[10:12] - 0.10)
    assert np.allclose(rec["q_target"][:10], contract.DEFAULT_JOINT_POS[:10])


def test_front_lift_bias_only_active_on_forward_cmd():
    drv = DummyDriver()
    # Backward command: front_lift_bias must not apply
    ctl_bwd = Controller(drv, DummyPolicy(), remote=DummyRemote([-0.5, 0.0, 0.0]),
                         soft_start=0.0, front_lift_bias=0.10)
    ctl_bwd.mode = WALK
    rec_bwd = ctl_bwd.step(DummyState())
    assert np.allclose(rec_bwd["q_target"], contract.DEFAULT_JOINT_POS)

    # Full forward command: FL/FR thigh -= 0.05, FL/FR calf -= 0.10
    ctl_fwd = Controller(drv, DummyPolicy(), remote=DummyRemote([0.5, 0.0, 0.0]),
                         soft_start=0.0, front_lift_bias=0.10)
    ctl_fwd.mode = WALK
    rec_fwd = ctl_fwd.step(DummyState())
    assert np.allclose(rec_fwd["q_target"][4:6], contract.DEFAULT_JOINT_POS[4:6] - 0.05)
    assert np.allclose(rec_fwd["q_target"][8:10], contract.DEFAULT_JOINT_POS[8:10] - 0.10)
    assert np.allclose(rec_fwd["last_action"] if "last_action" in rec_fwd else rec_fwd["action"], 0.0)
