import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talon_a1 import contract  # noqa: E402
from talon_a1.policy import save_policy  # noqa: E402


def random_layers(seed=0):
    """Random weights in the talon_rl.rma.networks deployable-path shapes."""
    rng = np.random.RandomState(seed)
    c = contract
    sizes = {
        "phi_embed": [c.HIST_DIM, 32, 32],
        "phi_proj": [32 * 3, c.LATENT],
        "velocity": [c.LATENT + c.VEL_HIST * c.OBS_DIM, 128, 64, c.VEL_DIM],
        "actor": [c.OBS_DIM + c.ACT_DIM + c.LATENT + c.VEL_DIM, 128, 128, c.ACT_DIM],
    }
    layers = {
        net: [(rng.randn(o, i) / np.sqrt(i), 0.1 * rng.randn(o)) for i, o in zip(s[:-1], s[1:])]
        for net, s in sizes.items()
    }
    layers["phi_conv"] = [(rng.randn(32, 32, k) / np.sqrt(32 * k), 0.1 * rng.randn(32)) for k in (8, 5, 5)]
    return layers


@pytest.fixture
def policy_file(tmp_path):
    path = str(tmp_path / "policy.npz")
    save_policy(path, random_layers(), {"format": 2, "test": True})
    return path
