import json

import numpy as np
import pytest

from talon_a1 import contract
from talon_a1.policy import PromoPolicy, elu


def test_elu():
    np.testing.assert_allclose(elu(np.array([-1.0, 0.0, 2.0])), [np.expm1(-1.0), 0.0, 2.0])


def test_shapes_and_determinism(policy_file):
    p = PromoPolicy(policy_file, contract.PREFERENCE_ANCHORS["balanced"])
    rng = np.random.RandomState(1)
    hist = rng.randn(contract.HIST, contract.OBS_DIM).astype(np.float32)
    a, v = p.action_mean(hist[-1], hist)
    assert a.shape == (contract.ACT_DIM,) and v.shape == (3,)
    assert a.dtype == np.float32
    a2, _ = p.action_mean(hist[-1], hist)
    assert np.array_equal(a, a2)


def test_preference_changes_action(policy_file):
    p = PromoPolicy(policy_file, contract.PREFERENCE_ANCHORS["tracking"])
    hist = np.ones((contract.HIST, contract.OBS_DIM), np.float32)
    a1, _ = p.action_mean(hist[-1], hist)
    p.set_preference(contract.PREFERENCE_ANCHORS["stability"])
    a2, _ = p.action_mean(hist[-1], hist)
    assert not np.allclose(a1, a2)


def test_rejects_bad_preference(policy_file):
    with pytest.raises(ValueError):
        PromoPolicy(policy_file, (0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        PromoPolicy(policy_file, (1.2, -0.1, -0.1))


def test_rejects_contract_mismatch(policy_file, tmp_path):
    data = dict(np.load(policy_file))
    meta = json.loads(str(data["meta"]))
    meta["contract"]["action_scale"] = 0.5
    data["meta"] = np.array(json.dumps(meta))
    bad = str(tmp_path / "bad.npz")
    np.savez(bad, **data)
    with pytest.raises(ValueError, match="action_scale"):
        PromoPolicy(bad, contract.PREFERENCE_ANCHORS["balanced"])
