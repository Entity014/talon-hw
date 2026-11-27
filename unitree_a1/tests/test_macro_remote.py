"""Tests for MacroRemote (joystick-triggered stand -> walk 3 m @ vx=0.6 m/s -> stop -> sit)."""

import numpy as np

from talon_a1 import contract
from talon_a1.controller import CROUCH_DOWN, PASSIVE, STAND, WALK, Controller
from talon_a1.remote import MacroRemote, combo


class FakeState:
    def __init__(self, q_policy=None, keys=0, ly=0.0, lx=0.0, rx=0.0, faults=0):
        q_p = contract.DEFAULT_JOINT_POS if q_policy is None else q_policy
        self.q = contract.policy_to_sdk(q_p)
        self.dq = np.zeros(12)
        self.quat = (1.0, 0.0, 0.0, 0.0)
        self.gyro = (0.0, 0.0, 0.0)
        self.faults = faults
        self.keys = keys
        self.ly = ly
        self.lx = lx
        self.rx = rx


class FakeDriver:
    def set_target(self, q, kp, kd):
        return True

    def arm(self):
        return True

    def disarm(self):
        pass


class FakePolicy:
    def action_mean(self, x, a_prev, hist):
        return np.zeros(12, np.float32), np.zeros(3), np.zeros(8)


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def test_full_sequence_stand_walk_3m_stop_sit():
    clk = FakeClock(0.0)
    dt = contract.POLICY_DT
    remote = MacroRemote(
        vx_max=0.5,
        macro_vx=0.6,
        macro_dist=3.0,
        macro_ramp=0.5,
        stand_time=3.0,
        soft_start=1.0,
        settle_time=0.5,
        clock=clk,
    )
    ctl = Controller(FakeDriver(), FakePolicy(), remote=remote, stand_time=3.0, soft_start=1.0)
    assert ctl.mode == PASSIVE

    # Initial idle tick
    ctl.step(FakeState())
    clk.t += dt

    # Press R1+X in PASSIVE -> triggers STAND_UP (L1+A)
    rec = ctl.step(FakeState(keys=combo("R1", "X")))
    assert ctl.mode == STAND
    assert remote.macro_active

    modes_seen = [ctl.mode]
    walk_vxs = []
    # Run for enough ticks to complete stand (3.25) + soft_start (1.0) + walk (5.5) + settle (0.5) + stop (1.2) + crouch (1.5)
    for _ in range(int(round(14.0 / dt))):
        clk.t += dt
        rec = ctl.step(FakeState())
        if ctl.mode != modes_seen[-1]:
            modes_seen.append(ctl.mode)
        if "command" in rec:
            walk_vxs.append(float(rec["command"][0]))

    assert modes_seen == [STAND, WALK, STAND, CROUCH_DOWN, PASSIVE]
    assert not remote.macro_active
    walk_vxs = np.array(walk_vxs)
    assert np.isclose(walk_vxs.max(), 0.6, atol=1e-5)
    dist = float(np.sum(walk_vxs) * dt)
    assert np.isclose(dist, 3.0, atol=0.02)


def test_manual_stick_still_clamped_to_vx_max():
    clk = FakeClock(0.0)
    r = MacroRemote(vx_max=0.5, macro_vx=0.6, macro_dist=3.0, clock=clk)
    r.update(FakeState(keys=0, ly=1.0))
    cmd = r.command(FakeState(keys=0, ly=1.0))
    assert np.isclose(cmd[0], 0.5)


def test_macro_cancel_by_r1_y_and_stick():
    clk = FakeClock(0.0)
    r = MacroRemote(clock=clk)
    r.update(FakeState(keys=0))
    r.update(FakeState(keys=combo("R1", "X")))
    assert r.macro_active

    clk.t += 1.0
    r.update(FakeState(keys=combo("R1", "Y")))
    assert not r.macro_active
    np.testing.assert_array_equal(r.command(FakeState(keys=0)), np.zeros(3, dtype=np.float32))

    # Re-trigger and cancel with stick movement
    r.update(FakeState(keys=combo("R1", "X")))
    assert r.macro_active
    clk.t += 0.5
    r.update(FakeState(keys=0, ly=0.3))
    assert not r.macro_active
    assert np.isclose(r.command(FakeState(keys=0, ly=0.3))[0], 0.5 * 0.3)


def test_short_distance_triangular_profile():
    clk = FakeClock(0.0)
    r = MacroRemote(macro_vx=0.6, macro_dist=0.15, macro_ramp=0.5, stand_time=0.1, soft_start=0.1, settle_time=0.1, clock=clk)
    ctl = Controller(FakeDriver(), FakePolicy(), remote=r, stand_time=0.1, soft_start=0.1)
    ctl.step(FakeState())
    clk.t += contract.POLICY_DT
    ctl.step(FakeState(keys=combo("R1", "X")))
    vxs = []
    for _ in range(int(round(5.0 / contract.POLICY_DT))):
        clk.t += contract.POLICY_DT
        rec = ctl.step(FakeState())
        if "command" in rec:
            vxs.append(float(rec["command"][0]))
    dist = float(np.sum(vxs) * contract.POLICY_DT)
    assert np.isclose(dist, 0.15, atol=0.01)
    assert ctl.mode == PASSIVE
