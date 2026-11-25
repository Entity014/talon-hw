"""Scripted hardware evaluation (scripts/eval_protocol.py) against a fake driver whose state mirrors the pybind RobotState of talon-hw (driver/a1_driver.cpp: q dq tau_est quat gyro acc rpy foot_force tick keys lx ly rx ry mode faults target_age loops overruns max_period)
and whose methods mirror Driver (state, set_target, arm, disarm, estop, clear_faults)."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HW)
sys.path.insert(0, os.path.join(HW, "scripts"))
pytest.importorskip("talon_a1.controller")

import eval_protocol as ep  # noqa: E402
from talon_a1 import contract  # noqa: E402
from talon_a1.remote import combo  # noqa: E402


class FakeState(object):
    def __init__(self, q_sdk, tick, keys=0, faults=0):
        self.q = list(q_sdk); self.dq = [0.0] * 12; self.tau_est = [0.0] * 12
        self.quat = (1.0, 0.0, 0.0, 0.0); self.gyro = (0.0, 0.0, 0.0); self.acc = (0.0, 0.0, 9.8); self.rpy = (0.0, 0.0, 0.0)
        self.foot_force = [100.0] * 4; self.tick = tick; self.keys = keys
        self.lx = self.ly = self.rx = self.ry = 0.0
        self.mode = 0; self.faults = faults; self.target_age = 0.0; self.loops = tick; self.overruns = 0; self.max_period = 0.002


class FakeDriver(object):
    def __init__(self, fault_at=None, keys_at=None):
        self.q = contract.policy_to_sdk(contract.CROUCH_JOINT_POS)  # starts lying down
        self.tick = 0; self.armed = False; self.faults = 0; self.fault_at = fault_at; self.keys_at = keys_at or {}

    def state(self):
        self.tick += 1
        if self.fault_at is not None and self.tick == self.fault_at:
            self.faults = 1
        return FakeState(self.q, self.tick, self.keys_at.get(self.tick, 0), self.faults)

    def set_target(self, q, kp, kd):
        self.q = list(q); return True

    def arm(self):
        self.armed = True; return True

    def disarm(self):
        self.armed = False

    def estop(self):
        self.faults = 1

    def clear_faults(self):
        self.faults = 0


class FakePolicy(object):  # the RMA-style call signature of Controller (a_prev, history) -> (action, v_hat, z_hat)
    def action_mean(self, x, a_prev, hist):
        return np.zeros(12, np.float32), np.zeros(3), np.zeros(8)


ARGS = SimpleNamespace(vx_max=0.5, vy_max=0.3, wz_max=0.5, stand_time=3.0, action_clip=10.0, policy_kp=55.0, soft_start=2.0, action_scale=0.25)
FAULTS = {"TILT": 1}


def run(tmp_path, regime, **kw):
    log = str(tmp_path / "t.npz")
    entry = ep.run_trial(FakeDriver(**kw), FakePolicy(), ep.REGIMES[regime], ARGS, np.array([0.8, 0.1, 0.1], np.float32), log, {"regime": regime}, sleep=False, fault_table=FAULTS)
    return entry, np.load(log, allow_pickle=True)


def test_profile_is_zero_before_ramped_between_segments_and_zero_after():
    seg = [(2.0, (0, 0, 0)), (4.0, (0.5, 0, 0)), (2.0, (0, 0, 0))]
    assert ep.profile_command(0.0, seg).tolist() == [0, 0, 0] and ep.profile_command(100.0, seg).tolist() == [0, 0, 0]
    assert ep.profile_command(2.25, seg)[0] == pytest.approx(0.25)  # half of the 0.5 s ramp
    assert ep.profile_command(4.0, seg)[0] == pytest.approx(0.5)
    assert ep.profile_command(6.25, seg)[0] == pytest.approx(0.25)
    assert ep.profile_duration(ep.REGIMES["move_fast"]) == 14.0


def test_schedule_is_complete_reproducible_and_shuffled():
    a = ep.make_schedule(list(ep.PREFERENCES), list(ep.REGIMES), 5, 0)
    assert len(a) == 60 and len(set(a)) == 60
    assert a == ep.make_schedule(list(ep.PREFERENCES), list(ep.REGIMES), 5, 0)
    assert a != ep.make_schedule(list(ep.PREFERENCES), list(ep.REGIMES), 5, 1)
    assert a != sorted(a)  # not in condition order


def test_trial_goes_passive_stand_walk_stand_passive_and_applies_the_profile(tmp_path):
    entry, d = run(tmp_path, "move_fast")
    assert entry["outcome"] == "OK" and entry["faults"] == []
    modes = [str(m) for m in d["mode"]]
    order = [m for i, m in enumerate(modes) if i == 0 or m != modes[i - 1]]
    assert order == ["PASSIVE", "STAND", "WALK", "STAND", "CROUCH_DOWN"]  # the log stores the mode at the START of each tick, the final PASSIVE tick ends the trial
    walk = np.array(modes) == "WALK"
    cx = d["command"][walk][:, 0]
    assert np.nanmax(cx) == pytest.approx(0.5) and np.nanmax(np.abs(d["command"][walk][:, 1:])) == 0.0
    assert entry["walk_time_s"] == pytest.approx(ep.profile_duration(ep.REGIMES["move_fast"]), abs=0.1)


def test_command_is_clipped_to_the_remote_limits(tmp_path):
    entry, d = run(tmp_path, "replay")
    walk = np.array([str(m) for m in d["mode"]]) == "WALK"
    c = d["command"][walk]
    assert np.nanmax(c[:, 0]) <= 0.5 + 1e-6 and np.nanmin(c[:, 0]) <= -0.29 and np.nanmax(np.abs(c[:, 1])) <= 0.3 + 1e-6 and np.nanmax(np.abs(c[:, 2])) <= 0.5 + 1e-6


def test_a_fault_ends_the_trial_as_fault_and_the_log_is_saved(tmp_path):
    entry, d = run(tmp_path, "move_fast", fault_at=700)  # during the walk
    assert entry["outcome"] == "FAULT" and entry["faults"] == ["TILT"]
    assert os.path.exists(str(tmp_path / "t.npz"))


def test_the_real_remote_still_stops_the_robot_during_the_walk(tmp_path):
    entry, d = run(tmp_path, "move_fast", keys_at={700: combo("L1", "B")})
    assert entry["outcome"] == "ABORT"
    assert "CROUCH_DOWN" in [str(m) for m in d["mode"]]  # the trial waited for the crouch to finish (not left half way)


def test_analysis_metrics_are_finite_and_energy_needs_the_measured_distance(tmp_path):
    sys.path.insert(0, os.path.join(HW, "scripts"))
    import analyze_protocol as ap

    entry, d = run(tmp_path, "move_fast")
    m = ap.log_metrics(d, None, skip_s=0.0)  # the fake loop does not sleep: the wall clock t is much shorter than the 20 ms ticks
    assert np.isnan(m["energy_j_per_m"]) and np.isnan(m["speed_mps"]) and np.isfinite(m["power_w"]) and np.isfinite(m["attitude_deg"]) and np.isfinite(m["action_rate"])
    m = ap.log_metrics(d, 4.0, mass_kg=12.0, skip_s=0.0)
    assert np.isfinite(m["energy_j_per_m"]) and m["speed_mps"] == pytest.approx(4.0 / m["walk_time_s"])
