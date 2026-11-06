"""Stand-up path and gains of the controller against a fake driver."""

import numpy as np

from talon_a1 import contract
from talon_a1.controller import STAND, WALK, Controller
from talon_a1.remote import combo


class FakeState(object):
    def __init__(self, q_policy, keys=0):
        self.q = contract.policy_to_sdk(q_policy)
        self.dq = np.zeros(12)
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.gyro = (0.0, 0.0, 0.0)
        self.faults = 0
        self.keys = keys
        self.lx = self.ly = self.rx = self.ry = 0.0


class FakeDriver(object):
    def __init__(self):
        self.targets = []

    def set_target(self, q, kp, kd):
        self.targets.append((contract.sdk_to_policy(np.array(q)), kp, kd))
        return True

    def arm(self):
        return True

    def disarm(self):
        pass


class FakePolicy(object):
    def action_mean(self, x, a_prev, hist):
        return np.ones(12, np.float32), np.zeros(3), np.zeros(8)


LYING = np.array([0.13, -0.29, 0.21, -0.24, 1.16, 1.13, 1.1, 1.11, -2.68, -2.69, -2.65, -2.7], dtype=np.float32)


def stand_up(q0, ticks=170, **kwargs):
    drv = FakeDriver()
    ctl = Controller(drv, FakePolicy(), stand_time=3.0, **kwargs)
    ctl.step(FakeState(q0))  # keys were up
    ctl.step(FakeState(q0, combo("L1", "A")))
    assert ctl.mode == STAND
    for _ in range(ticks):
        ctl.step(FakeState(q0))
    return drv, ctl


def test_from_lying_goes_through_the_crouch_to_the_default_pose():
    drv, _ = stand_up(LYING)
    path = np.array([t[0] for t in drv.targets])
    np.testing.assert_allclose(path[0], LYING, atol=1e-5)
    np.testing.assert_allclose(path[75], contract.CROUCH_JOINT_POS, atol=1e-3)  # halfway through 3 s at 50 Hz
    np.testing.assert_allclose(path[-1], contract.DEFAULT_JOINT_POS, atol=1e-5)
    assert np.abs(np.diff(path, axis=0)).max() < 0.05  # no jump between 20 ms ticks


def test_standing_robot_does_not_crouch():
    q0 = contract.DEFAULT_JOINT_POS + 0.05
    drv, _ = stand_up(q0)
    path = np.array([t[0] for t in drv.targets])
    assert path[:, 8:12].min() > -1.5


def test_bring_up_limits_reach_the_motor_and_the_history():
    q0 = contract.DEFAULT_JOINT_POS
    drv, ctl = stand_up(q0, action_clip=0.1, policy_kp=20.0)
    assert all(t[1:] == (contract.STAND_KP, contract.STAND_KD) for t in drv.targets)  # standing is unchanged
    ctl.step(FakeState(q0, combo("L1", "X")))
    ctl.step(FakeState(q0))
    assert drv.targets[-1][1:] == (20.0, contract.KD)
    np.testing.assert_allclose(drv.targets[-1][0], q0 + contract.ACTION_SCALE * 0.1, atol=1e-5)
    np.testing.assert_allclose(ctl.last_action, 0.1, atol=1e-6)  # the policy is fed what was applied
    np.testing.assert_allclose(ctl.hist.buf[-1, contract.OBS_DIM:], 0.1, atol=1e-6)


def test_stand_only_never_starts_the_policy():
    q0 = contract.DEFAULT_JOINT_POS
    drv, ctl = stand_up(q0, allow_walk=False)
    ctl.step(FakeState(q0, combo("L1", "X")))
    ctl.step(FakeState(q0))
    assert ctl.mode == STAND
    assert all(t[1:] == (contract.STAND_KP, contract.STAND_KD) for t in drv.targets)


def test_stand_gains_then_policy_gains():
    q0 = contract.DEFAULT_JOINT_POS
    drv, ctl = stand_up(q0)
    assert all(t[1:] == (contract.STAND_KP, contract.STAND_KD) for t in drv.targets)
    ctl.step(FakeState(q0, combo("L1", "X")))
    assert ctl.mode == WALK
    ctl.step(FakeState(q0))
    assert drv.targets[-1][1:] == (contract.KP, contract.KD)
    np.testing.assert_allclose(drv.targets[-1][0], contract.DEFAULT_JOINT_POS + contract.ACTION_SCALE, atol=1e-5)
    ctl.step(FakeState(q0, combo("L1", "Y")))
    ctl.step(FakeState(q0))
    assert ctl.mode == STAND and drv.targets[-1][1:] == (contract.STAND_KP, contract.STAND_KD)
