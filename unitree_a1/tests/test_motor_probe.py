"""motor_probe.py: the verdict rule, and a full run on the loopback desk plant."""

import importlib.util
import os
import time

import numpy as np
import pytest

from talon_a1 import contract

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("motor_probe", os.path.join(ROOT, "scripts", "motor_probe.py"))
_a1_driver = pytest.importorskip("talon_a1._a1_driver")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize(
    "sign, dq, dtau, derr, others, want",
    [
        (1, 0.08, 0.1, 0.02, 0.0, "PASS"),  # moved the right way
        (-1, -0.08, -0.1, -0.02, 0.0, "PASS"),
        (1, 0.0, 0.5, 0.1, 0.0, "PASS"),  # held back by friction, torque is right at Kp * error
        (1, -0.08, -0.1, 0.18, 0.0, "WRONG"),  # moved against the command
        (1, 0.0, -0.5, 0.1, 0.0, "WRONG"),  # pushed against the command
        (1, 0.08, 0.1, 0.02, 0.05, "WRONG"),  # a different joint answered
        (1, 0.0, 2.5, 0.1, 0.0, "WRONG"),  # torque 5 x Kp * error: gain scale off
        (1, 0.0, 0.0, 0.1, 0.0, "INCONCLUSIVE"),
    ],
)
def test_judge(sign, dq, dtau, derr, others, want):
    assert probe.judge(sign, dq, dtau, derr, others)[0] == want


@pytest.mark.parametrize(
    "dq, dtau, others, want",
    [
        (0.08, -0.9, 0.0, "PASS"),  # moved the right way; torque follows gravity on a free leg
        (0.005, 0.5, 0.0, "INCONCLUSIVE"),  # torque alone is not motion
        (-0.08, 0.5, 0.0, "WRONG"),
        (0.08, 0.1, 0.05, "WRONG"),
    ],
)
def test_judge_needs_motion_on_the_box(dq, dtau, others, want):
    assert probe.judge(1, dq, dtau, 0.02, others, need_motion=True)[0] == want


def test_limits_are_small():
    assert probe.KP * probe.STEP <= 0.5
    assert probe.BOX_KP * probe.STEP <= 1.5


def test_box_leaves_out_every_hip():
    names = [contract.POLICY_JOINTS[i] for i in probe.joints_to_probe(True)]
    assert len(names) == 8 and not any("hip" in n for n in names)
    assert len(probe.joints_to_probe(False)) == 12


@pytest.mark.skipif(_a1_driver.WITH_SDK, reason="needs the loopback build")
def test_box_run_never_moves_a_hip_target():
    cfg = _a1_driver.SafetyConfig()
    drv = _a1_driver.Driver(cfg, 0.002, 0)
    drv.start()
    time.sleep(0.05)
    rows, results = [], {}
    try:
        probe.run(drv, cfg, dict(_a1_driver.FAULTS), rows, results, auto=True, box=True)
    finally:
        drv.stop()
    assert set(results.values()) == {"PASS"} and len(results) == 8
    hips = np.asarray(rows)[:, 1:5]  # target columns, policy order
    assert np.ptp(hips, axis=0).max() == 0.0


def test_directions_leave_the_folded_calf_limit():
    for name, sign in probe.DIRECTION.items():
        if "calf" in name:
            assert sign > 0


@pytest.mark.skipif(_a1_driver.WITH_SDK, reason="needs the loopback build")
def test_full_run_on_desk():
    cfg = _a1_driver.SafetyConfig()
    drv = _a1_driver.Driver(cfg, 0.002, 0)
    drv.start()
    time.sleep(0.05)
    rows, results = [], {}
    try:
        probe.run(drv, cfg, dict(_a1_driver.FAULTS), rows, results, auto=True)
    finally:
        drv.stop()
    rows = np.asarray(rows)
    assert set(results) == set(contract.POLICY_JOINTS)
    assert set(results.values()) == {"PASS"}
    assert drv.state().mode == 0
    q0 = rows[0, 1:13]
    assert np.abs(rows[:, 1:13] - q0).max() <= probe.STEP + 0.06  # targets stay at the step, plus the first clamp


@pytest.mark.skipif(_a1_driver.WITH_SDK, reason="needs the loopback build")
def test_fault_stops_in_damping():
    cfg = _a1_driver.SafetyConfig()
    drv = _a1_driver.Driver(cfg, 0.002, 0)
    drv.start()
    time.sleep(0.05)
    drv.sim_set_remote(0)
    try:
        import threading

        threading.Timer(0.5, lambda: drv.sim_set_tilt(0.0, 1.0)).start()
        rows, results = [], {}
        with pytest.raises(probe.Abort, match="tilt"):
            probe.run(drv, cfg, dict(_a1_driver.FAULTS), rows, results, auto=True)
        assert drv.state().mode == 0
        assert rows  # the log survives the abort
    finally:
        drv.stop()
