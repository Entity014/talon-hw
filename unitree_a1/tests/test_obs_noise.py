"""Observation-noise injection of the controller (the dq A/B test on the robot): channels, amplitudes, reproducibility, and that zero noise changes nothing."""
import numpy as np

from talon_a1 import contract
from talon_a1.controller import STAND, TRAIN_OBS_NOISE, WALK, Controller, obs_noise_amplitude
from talon_a1.obs import build_observation_promo_v2
from talon_a1.policy import PromoNumpyPolicy
from talon_a1.remote import combo

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_controller import FakeDriver, FakeState  # noqa: E402

ARTIFACT = os.environ.get("TALON_OURSA_NPZ", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policies", "oursA", "policy.npz"))


def test_amplitudes_follow_the_training_noise_layout():
    assert TRAIN_OBS_NOISE.shape == (45,)
    a = obs_noise_amplitude(1.0)
    np.testing.assert_allclose(a[0:3], 0.2); np.testing.assert_allclose(a[3:6], 0.05); np.testing.assert_allclose(a[6:9], 0.0)
    np.testing.assert_allclose(a[9:21], 0.01); np.testing.assert_allclose(a[21:33], 1.5); np.testing.assert_allclose(a[33:45], 0.0)
    assert obs_noise_amplitude(0.0) is None and obs_noise_amplitude(0.0, None) is None
    d = obs_noise_amplitude(0.0, 1.5)  # the dq-only test: only joint_vel
    assert np.count_nonzero(d) == 12 and np.all(d[21:33] == 1.5)
    d = obs_noise_amplitude(0.5, 0.3)
    np.testing.assert_allclose(d[21:33], 0.3); np.testing.assert_allclose(d[0:3], 0.1)  # dq overrides only its own channels


class Probe(object):
    """Policy-like object that records the observations it is given (RMA-style call signature so no model file is needed); the noise path of Controller is the PROMO one,
    so the test drives Controller._noisy and the observation builder directly with a PROMO-type controller."""


def _controller(obs_noise, seed=0):
    pol = _promo_policy()
    return Controller(FakeDriver(), pol, stand_time=3.0, action_clip=10.0, soft_start=0.0, obs_noise=obs_noise, noise_seed=seed, w=[0.8, 0.1, 0.1])


def _promo_policy():
    if not os.path.exists(ARTIFACT):
        import pytest

        pytest.skip("policies/oursA/policy.npz is not in this checkout (policies/ is git-ignored)")
    return PromoNumpyPolicy(ARTIFACT)


def test_noise_is_uniform_bounded_per_channel_and_reproducible():
    ctl = _controller(obs_noise_amplitude(1.0), seed=3)
    x = build_observation_promo_v2(FakeState(contract.DEFAULT_JOINT_POS), (0.5, 0, 0), np.zeros(12, np.float32), ctl.default_joint_pos)
    xs = np.array([ctl._noisy(x) - x for _ in range(4000)])
    assert np.all(np.abs(xs) <= TRAIN_OBS_NOISE + 1e-6)
    assert np.all(xs[:, 6:9] == 0) and np.all(xs[:, 33:45] == 0)  # command and last action stay clean
    assert abs(float(xs[:, 21:33].std()) - 1.5 / np.sqrt(3)) < 0.03 and abs(float(xs[:, 0:3].mean())) < 0.02
    a, b = _controller(obs_noise_amplitude(1.0), 3), _controller(obs_noise_amplitude(1.0), 3)
    np.testing.assert_array_equal(a._noisy(x), b._noisy(x))


def test_no_noise_returns_the_same_observation_object_values():
    ctl = _controller(None)
    x = build_observation_promo_v2(FakeState(contract.DEFAULT_JOINT_POS), (0.5, 0, 0), np.zeros(12, np.float32), ctl.default_joint_pos)
    assert ctl._noisy(x) is x


def test_walk_pushes_the_noisy_observation_into_policy_and_history():
    ctl = _controller(obs_noise_amplitude(0.0, 1.5), seed=1)
    q0 = contract.DEFAULT_JOINT_POS
    ctl.step(FakeState(q0)); ctl.step(FakeState(q0, combo("L1", "A")))
    for _ in range(170):
        ctl.step(FakeState(q0))
    assert ctl.mode == STAND
    ctl.step(FakeState(q0, combo("L1", "X"))); assert ctl.mode == WALK
    ctl.step(FakeState(q0))
    last = ctl.hist.buf[-1][: contract.OBS_DIM_PROMO_V2]
    clean = build_observation_promo_v2(FakeState(q0), (0.0, 0.0, 0.0), np.zeros(12, np.float32), ctl.default_joint_pos)
    dev = np.abs(last - clean)
    assert np.all(dev[21:33] > 0) and np.all(dev[21:33] <= 1.5 + 1e-6)  # the fake robot has dq = 0: the whole joint_vel channel is noise, inside +-1.5
    assert np.all(dev[0:21] < 1e-5) and np.all(dev[33:45] < 1e-5)  # dq-only test: no other channel is touched


def test_lead_compensation_shifts_only_joint_pos_by_joint_vel_times_lead_and_is_off_by_default():
    pol = _promo_policy()
    q0 = contract.DEFAULT_JOINT_POS
    st = FakeState(q0)
    st.dq = np.full(12, 2.0)
    x = build_observation_promo_v2(st, (0.5, 0, 0), np.zeros(12, np.float32), pol.default_q)
    off = Controller(FakeDriver(), pol, stand_time=3.0, w=[0.8, 0.1, 0.1])
    assert off._noisy(x) is x
    lead = Controller(FakeDriver(), pol, stand_time=3.0, w=[0.8, 0.1, 0.1], lead_ms=10.0)
    y = lead._noisy(x)
    np.testing.assert_allclose(y[9:21] - x[9:21], 2.0 * 0.010, atol=1e-6)
    np.testing.assert_array_equal(np.delete(y, np.arange(9, 21)), np.delete(x, np.arange(9, 21)))
    assert not np.shares_memory(x, y)  # the observation built from the state is not modified in place
