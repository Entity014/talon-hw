"""Full stack on the loopback desk plant: C++ motor loop + safety core + controller + NumPy policy.

Needs _a1_driver built with -DTALON_A1_WITH_SDK=OFF; skipped otherwise.
"""

import time

import numpy as np
import pytest

from talon_a1 import contract
from talon_a1.controller import PASSIVE, STAND, WALK, Controller
from talon_a1.policy import RmaPolicy
from talon_a1.remote import combo

_a1_driver = pytest.importorskip("talon_a1._a1_driver")
if _a1_driver.WITH_SDK:
    pytest.skip("driver was built against the real SDK", allow_module_level=True)

FAULTS = dict(_a1_driver.FAULTS)


@pytest.fixture
def rig(policy_file):
    drv = _a1_driver.Driver(_a1_driver.SafetyConfig(), 0.002)
    drv.start()
    time.sleep(0.05)
    ctl = Controller(drv, RmaPolicy(policy_file),
                     fault_table=FAULTS, stand_time=0.4, action_clip=0.5)
    yield drv, ctl
    drv.stop()


def run(ctl, n, keys=None, drv=None, ly=0.0):
    """n policy ticks at 50 Hz. keys are held for the first tick only (one press)."""
    for i in range(n):
        if drv is not None:
            drv.sim_set_remote(keys if (keys and i == 0) else 0, 0.0, ly, 0.0, 0.0)
            time.sleep(0.004)  # let the motor loop pick up the new remote frame
        ctl.step(drv.state() if drv is not None else ctl.drv.state())
        time.sleep(contract.POLICY_DT)


def stand_up(drv, ctl):
    run(ctl, 1, combo("L1", "A"), drv)
    assert ctl.mode == STAND and drv.state().mode == 1
    run(ctl, 30, None, drv)  # ramp done (0.4 s) and settled


def test_stand_walk_passive(rig):
    drv, ctl = rig
    assert ctl.mode == PASSIVE and drv.state().mode == 0
    stand_up(drv, ctl)
    q = contract.sdk_to_policy(drv.state().q)
    assert np.abs(q - contract.DEFAULT_JOINT_POS).max() < 0.05
    run(ctl, 1, combo("L1", "X"), drv)
    assert ctl.mode == WALK
    run(ctl, 25, None, drv, ly=0.8)
    s = drv.state()
    assert ctl.mode == WALK and s.mode == 1 and s.faults == 0
    run(ctl, 1, combo("L1", "B"), drv)
    assert ctl.mode == PASSIVE and drv.state().mode == 0


def test_walk_needs_finished_ramp(rig):
    drv, ctl = rig
    run(ctl, 1, combo("L1", "A"), drv)
    run(ctl, 1, combo("L1", "X"), drv)  # ramp still running
    assert ctl.mode == STAND


def test_python_stall_latches_stale_target(rig):
    drv, ctl = rig
    stand_up(drv, ctl)
    time.sleep(0.1)  # no set_target for 100 ms
    s = drv.state()
    assert s.mode == 0 and s.faults & FAULTS["stale_target"]
    run(ctl, 1, None, drv)
    assert ctl.mode == PASSIVE
    run(ctl, 1, combo("L1", "A"), drv)  # refused while latched
    assert ctl.mode == PASSIVE
    run(ctl, 1, combo("SELECT"), drv)
    assert drv.state().faults == 0
    stand_up(drv, ctl)


def test_estop_button_in_any_mode(rig):
    drv, ctl = rig
    drv.sim_set_remote(combo("L2", "B"))
    time.sleep(0.01)
    assert drv.state().faults & FAULTS["estop_button"]
    drv.sim_set_remote(0)


def test_tilt_latches_while_armed(rig):
    drv, ctl = rig
    stand_up(drv, ctl)
    drv.sim_set_tilt(0.0, 1.0)
    run(ctl, 1, None, drv)
    assert drv.state().faults & FAULTS["tilt"] and ctl.mode == PASSIVE


def test_overtorque_latches_while_armed(rig):
    drv, ctl = rig
    drv.sim_set_tau_est(5, 31.0)  # ignored while damped
    time.sleep(0.01)
    assert drv.state().faults == 0
    drv.sim_set_tau_est(5, 0.0)
    stand_up(drv, ctl)
    drv.sim_set_tau_est(5, -31.0)
    run(ctl, 1, None, drv)
    assert drv.state().faults & FAULTS["overtorque"] and ctl.mode == PASSIVE


def test_frozen_robot_state_latches(rig):
    drv, ctl = rig
    stand_up(drv, ctl)
    drv.sim_freeze_state(True)
    run(ctl, 5, None, drv)
    assert drv.state().faults & FAULTS["stale_state"]


def test_arm_refused_without_robot(policy_file):
    drv = _a1_driver.Driver(_a1_driver.SafetyConfig(), 0.002)
    drv.start()
    drv.sim_freeze_state(True)  # tick never changes: no robot talking
    time.sleep(0.05)
    assert drv.set_target(list(contract.policy_to_sdk(contract.DEFAULT_JOINT_POS).astype(float)), 25.0, 0.5)
    assert not drv.arm()
    drv.stop()
