"""NumPy port of deployment RMA: phi (history -> z_hat) -> velocity estimator -> actor.

Mirrors talon_rl.rma.networks AdaptationModule, VelocityEstimator and BasePolicy. The weights come from
scripts/export_policy.py; its parity report gives the max action error against the PyTorch model.
"""

import json

import numpy as np

from . import contract

NETS = ("phi_embed", "phi_conv", "phi_proj", "velocity", "actor")
CONV_STRIDES = (4, 1, 1)  # AdaptationModule.cnn


def leaky_relu(x):
    """nn.LeakyReLU default slope."""
    return np.where(x >= 0.0, x, 0.01 * x)


class Mlp(object):
    """Linear, LeakyReLU, ..., Linear (talon_rl.rma.networks.mlp). Works on the last axis."""

    def __init__(self, weights, biases):
        self.weights = [np.ascontiguousarray(w, dtype=np.float32) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32) for b in biases]

    def __call__(self, x):
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x.dot(w.T) + b
            if i < last:
                x = leaky_relu(x)
        return x


class Conv1dStack(object):
    """Conv1d, LeakyReLU after every layer, on (channels, length). Windows are precomputed per input length."""

    def __init__(self, weights, biases, strides=CONV_STRIDES):
        self.weights = [np.ascontiguousarray(w, dtype=np.float32) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float32) for b in biases]
        self.strides = strides

    def __call__(self, x):
        for w, b, stride in zip(self.weights, self.biases, self.strides):
            kernel = w.shape[2]
            starts = np.arange(0, x.shape[1] - kernel + 1, stride)
            windows = x[:, starts[:, None] + np.arange(kernel)]  # (in, out_length, kernel)
            x = leaky_relu(np.tensordot(w, windows, axes=([1, 2], [0, 2])) + b[:, None])
        return x


def save_policy(path, layers, meta):
    """layers: {net: [(weight, bias), ...]} for every net in NETS."""
    arrays = {}
    meta = dict(meta, contract=contract.as_dict(), layers={net: len(layers[net]) for net in NETS})
    for net in NETS:
        for i, (w, b) in enumerate(layers[net]):
            arrays["%s.%d.weight" % (net, i)] = np.asarray(w, dtype=np.float32)
            arrays["%s.%d.bias" % (net, i)] = np.asarray(b, dtype=np.float32)
    arrays["meta"] = np.array(json.dumps(meta, default=str))
    np.savez(path, **arrays)


def load_layers(path):
    """Returns ({net: (weights, biases)}, meta) from an export_policy.py file."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    layers = {}
    for net in NETS:
        n = meta["layers"][net]
        layers[net] = ([data["%s.%d.weight" % (net, i)] for i in range(n)],
                       [data["%s.%d.bias" % (net, i)] for i in range(n)])
    return layers, meta


class RmaPolicy(object):
    def __init__(self, path):
        layers, self.meta = load_layers(path)
        expected = contract.as_dict()
        mismatched = [k for k, v in expected.items() if self.meta["contract"].get(k) != v]
        if mismatched:
            raise ValueError("policy file %s disagrees with contract.py on %s" % (path, ", ".join(mismatched)))
        self.phi_embed = Mlp(*layers["phi_embed"])
        self.phi_conv = Conv1dStack(*layers["phi_conv"])
        self.phi_proj = Mlp(*layers["phi_proj"])
        self.velocity = Mlp(*layers["velocity"])
        self.actor = Mlp(*layers["actor"])

    def encode_history(self, hist):
        """hist: (HIST, HIST_DIM) oldest first. Returns z_hat (LATENT,), v_hat (VEL_DIM,)."""
        z_hat = self.phi_proj(self.phi_conv(self.phi_embed(hist).T).reshape(-1))
        recent_x = hist[-contract.VEL_HIST:, :contract.OBS_DIM].reshape(-1)
        v_hat = self.velocity(np.concatenate([z_hat, recent_x]))
        return z_hat, v_hat

    def action_mean(self, x, a_prev, hist):
        """x: current observation, a_prev: last applied action, hist: pairs up to t-1."""
        z_hat, v_hat = self.encode_history(hist)
        a = np.tanh(self.actor(np.concatenate([x, a_prev, z_hat, v_hat])))  # BasePolicy out_act=nn.Tanh
        return a, v_hat, z_hat
