"""Tests for IMU roll/pitch bias correction and --imu-zero calibration."""

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from talon_a1 import contract
from talon_a1.controller import Controller, STAND, START_WALK, WALK
from talon_a1.obs import PromoHistory, correct_projected_gravity


def test_correct_projected_gravity_zeros_static_roll_and_pitch():
    roll_bias = 0.07
    pitch_bias = 0.056
    # Level robot with positive roll/pitch mounting offset produces g_x = -sin(pitch), g_y = -sin(roll)*cos(pitch)
    g_raw = np.array([-np.sin(pitch_bias), -np.sin(roll_bias) * np.cos(pitch_bias), -np.cos(roll_bias) * np.cos(pitch_bias)], dtype=np.float32)
    g_fixed = correct_projected_gravity(g_raw, roll_bias=roll_bias, pitch_bias=pitch_bias)
    assert np.allclose(g_fixed, [0.0, 0.0, -1.0], atol=1e-4)


class DummyDriver:
    def set_target(self, q, kp, kd):
        return True

    def disarm(self):
        pass

    def clear_faults(self):
        pass


class DummyPolicy:
    obs_dim = 45
    act_dim = 12
    hist_len = 10
    action_scale = 0.25
    kp = 55.0
    meta = {}

    def action_mean(self, obs, hist, w=None):
        return np.zeros(12, dtype=np.float32)


class DummyRemote:
    def __init__(self):
        self.keys = 0
        self.vx_max = 0.5

    def update(self, state):
        pass

    def pressed(self, key):
        return (self.keys & key) == key

    def command(self, state):
        return np.zeros(3, dtype=np.float32)


class DummyState:
    faults = 0
    tick = 1
    gyro = np.zeros(3, dtype=np.float32)
    acc = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    # Pure roll = +0.07 rad quaternion [cos(r/2), sin(r/2), 0, 0]
    quat = np.array([np.cos(0.035), np.sin(0.035), 0.0, 0.0], dtype=np.float32)
    rpy = np.array([0.07, 0.0, 0.0], dtype=np.float32)
    q = contract.DEFAULT_JOINT_POS_PROMO_V2.copy()
    dq = np.zeros(12, dtype=np.float32)
    tau_est = np.zeros(12, dtype=np.float32)
    keys = 0
    lx = ly = rx = ry = 0.0


def test_imu_zero_calibrates_on_stand_to_walk():
    drv = DummyDriver()
    rem = DummyRemote()
    ctl = Controller(drv, DummyPolicy(), remote=rem, soft_start=0.0, imu_zero=True)
    ctl.is_promo = True
    ctl.hist = PromoHistory(length=10, dim=45)
    ctl.mode = STAND
    ctl.ramp_from = contract.DEFAULT_JOINT_POS_PROMO_V2.copy()
    ctl.ramp_legs = [(contract.DEFAULT_JOINT_POS_PROMO_V2.copy(), 0.0)]
    ctl.ramp_t = 0.1

    s = DummyState()
    ctl.step(s)
    rem.keys = START_WALK
    rec = ctl.step(s)
    assert ctl.mode == WALK
    assert np.isclose(ctl.imu_roll_bias, 0.07, atol=1e-4)
    # First WALK tick should see corrected projected_gravity ~ [0, 0, -1]
    rem.keys = 0
    rec2 = ctl.step(s)
    assert np.allclose(rec2["obs"][3:6], [0.0, 0.0, -1.0], atol=1e-4)
