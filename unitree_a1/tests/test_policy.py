import json

import numpy as np
import pytest

from talon_a1 import contract
from talon_a1.policy import Conv1dStack, RmaPolicy, leaky_relu


def test_leaky_relu():
    np.testing.assert_allclose(leaky_relu(np.array([-1.0, 0.0, 2.0])), [-0.01, 0.0, 2.0])


def test_conv_is_correlation_not_convolution():
    # One channel, kernel [1, 2], stride 2, no bias: y_t = x_t + 2 x_{t+1}, with no kernel flip.
    conv = Conv1dStack([np.array([[[1.0, 2.0]]])], [np.zeros(1)], strides=(2,))
    out = conv(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]))
    np.testing.assert_allclose(out, [[5.0, 11.0]])


def test_shapes_and_determinism(policy_file):
    p = RmaPolicy(policy_file)
    rng = np.random.RandomState(1)
    hist = rng.randn(contract.HIST, contract.HIST_DIM).astype(np.float32)
    x = rng.randn(contract.OBS_DIM).astype(np.float32)
    a_prev = np.zeros(contract.ACT_DIM, np.float32)
    a, v, z = p.action_mean(x, a_prev, hist)
    assert a.shape == (contract.ACT_DIM,) and v.shape == (contract.VEL_DIM,) and z.shape == (contract.LATENT,)
    assert a.dtype == np.float32
    assert np.abs(a).max() <= 1.0  # tanh output
    a2, _, _ = p.action_mean(x, a_prev, hist)
    assert np.array_equal(a, a2)


def test_first_call_happens_while_loading(policy_file, monkeypatch):
    calls = []
    original = RmaPolicy.action_mean
    monkeypatch.setattr(RmaPolicy, "action_mean", lambda self, *a: calls.append(1) or original(self, *a))
    RmaPolicy(policy_file)
    assert calls


def test_history_changes_action(policy_file):
    p = RmaPolicy(policy_file)
    x = np.ones(contract.OBS_DIM, np.float32)
    a_prev = np.zeros(contract.ACT_DIM, np.float32)
    a1, _, _ = p.action_mean(x, a_prev, np.zeros((contract.HIST, contract.HIST_DIM), np.float32))
    a2, _, _ = p.action_mean(x, a_prev, np.ones((contract.HIST, contract.HIST_DIM), np.float32))
    assert not np.allclose(a1, a2)


def test_rejects_contract_mismatch(policy_file, tmp_path):
    data = dict(np.load(policy_file))
    meta = json.loads(str(data["meta"]))
    meta["contract"]["action_scale"] = 0.5
    data["meta"] = np.array(json.dumps(meta))
    bad = str(tmp_path / "bad.npz")
    np.savez(bad, **data)
    with pytest.raises(ValueError, match="action_scale"):
        RmaPolicy(bad)


def test_onnx_policy_loading_and_inference():
    import os
    from talon_a1.policy import OnnxPolicy, load_policy
    onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policies", "base_v2", "policy.onnx")
    if not os.path.exists(onnx_path):
        pytest.skip("base_v2/policy.onnx not found")
    p = load_policy(onnx_path)
    assert isinstance(p, OnnxPolicy)
    assert p.obs_dim == 45
    assert p.hist_len == 10
    assert p.action_scale == 0.25

    obs = np.zeros(p.obs_dim, dtype=np.float32)
    hist = np.zeros((p.hist_len, p.obs_dim), dtype=np.float32)
    act = p.action_mean(obs, hist)
    assert act.shape == (12,)
    assert act.dtype == np.float32

