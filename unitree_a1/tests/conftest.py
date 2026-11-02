import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talon_a1 import contract  # noqa: E402
from talon_a1.policy import save_policy  # noqa: E402


def random_layers(seed=0):
    """Random weights in PromoActorCritic's deployable-path shapes (enc_hidden 256,128; net_hidden 512,256,128)."""
    rng = np.random.RandomState(seed)
    c = contract
    sizes = {
        "hist_enc": [c.HIST * c.OBS_DIM + c.PREF_DIM, 256, 128, c.LATENT],
        "vel_est": [c.LATENT + c.VEL_HIST * c.OBS_DIM, 256, 128, 3],
        "actor": [c.OBS_DIM + c.LATENT + 3 + c.PREF_DIM, 512, 256, 128, c.ACT_DIM],
    }
    return {
        net: [(rng.randn(o, i) / np.sqrt(i), 0.1 * rng.randn(o)) for i, o in zip(s[:-1], s[1:])]
        for net, s in sizes.items()
    }


@pytest.fixture
def policy_file(tmp_path):
    path = str(tmp_path / "policy.npz")
    save_policy(path, random_layers(), {"format": 1, "test": True})
    return path
