"""NumPy port of PROMO's deployable path: E_H (history encoder) -> E_V (velocity estimator) -> actor.

Mirrors talon_rl.promo.networks.PromoActorCritic.encode_history / action_mean. The weights come from
scripts/export_policy.py; its parity report gives the max action error against the PyTorch model.
"""

import json

import numpy as np

from . import contract

NETS = ("hist_enc", "vel_est", "actor")


def elu(x):
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


class Mlp(object):
    """Linear, ELU, ..., Linear (talon_rl.promo.networks.mlp with last_act=False)."""

    def __init__(self, weights, biases):
        self.weights = [np.ascontiguousarray(w, dtype=np.float32) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32) for b in biases]

    def __call__(self, x):
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x.dot(w.T) + b
            if i < last:
                x = elu(x)
        return x


def save_policy(path, layers, meta):
    """layers: {net: [(weight (out, in), bias (out,)), ...]} for every net in NETS."""
    arrays = {}
    meta = dict(meta, contract=contract.as_dict(), layers={net: len(layers[net]) for net in NETS})
    for net in NETS:
        for i, (w, b) in enumerate(layers[net]):
            arrays["%s.%d.weight" % (net, i)] = np.asarray(w, dtype=np.float32)
            arrays["%s.%d.bias" % (net, i)] = np.asarray(b, dtype=np.float32)
    arrays["meta"] = np.array(json.dumps(meta, default=str))
    np.savez(path, **arrays)


def load_mlps(path):
    """Returns ({net: Mlp}, meta) from an export_policy.py file."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    mlps = {}
    for net in NETS:
        n = meta["layers"][net]
        mlps[net] = Mlp(
            [data["%s.%d.weight" % (net, i)] for i in range(n)], [data["%s.%d.bias" % (net, i)] for i in range(n)]
        )
    return mlps, meta


class PromoPolicy(object):
    def __init__(self, path, preference):
        self.mlps, self.meta = load_mlps(path)
        expected = contract.as_dict()
        mismatched = [k for k, v in expected.items() if self.meta["contract"].get(k) != v]
        if mismatched:
            raise ValueError("policy file %s disagrees with contract.py on %s" % (path, ", ".join(mismatched)))
        self.set_preference(preference)

    def set_preference(self, preference):
        w = np.asarray(preference, dtype=np.float32).reshape(contract.PREF_DIM)
        if np.any(w < 0.0) or abs(float(w.sum()) - 1.0) > 1e-4:
            raise ValueError("preference must be non-negative and sum to 1, got %s" % (w,))
        self.w = w

    def encode_history(self, hist):
        """hist: (HIST, OBS_DIM) oldest first. Returns z_H (LATENT,), v_hat (3,)."""
        z_h = self.mlps["hist_enc"](np.concatenate([hist.reshape(-1), self.w]))
        recent = hist[-contract.VEL_HIST:].reshape(-1)
        v_hat = self.mlps["vel_est"](np.concatenate([z_h, recent]))
        return z_h, v_hat

    def action_mean(self, obs, hist):
        z_h, v_hat = self.encode_history(hist)
        return self.mlps["actor"](np.concatenate([obs, z_h, v_hat, self.w])), v_hat
