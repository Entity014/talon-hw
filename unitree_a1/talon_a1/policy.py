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
        # The first call took 22 ms on the robot's Atom, then 0.2 ms. Pay it here: at L1+X it adds to the 20 ms tick
        # and trips the driver's 40 ms stale-target fault before the first action reaches the motors.
        self.action_mean(np.zeros(contract.OBS_DIM, np.float32), np.zeros(contract.ACT_DIM, np.float32),
                         np.zeros((contract.HIST, contract.HIST_DIM), np.float32))

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


class OnnxPolicy(object):
    """ONNX Runtime wrapper for deployable policies (PROMO v2: base_v2, oursA).

    Expected inputs to ONNX model:
      - 'obs':  (1, obs_dim)
      - 'hist': (1, hist_len, obs_dim)
      - 'w':    (1, 3) (if model expects 'w')
    Output:
      - 'action': (1, 12)
    """

    def __init__(self, onnx_path, meta_path=None, w=None):
        import onnxruntime as ort

        self.onnx_path = str(onnx_path)
        self.meta = {}
        if meta_path is not None:
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
        else:
            cand = onnx_path.replace(".onnx", "_meta.json") if onnx_path.endswith(".onnx") else None
            if cand and np.os.path.exists(cand):
                with open(cand, "r") as f:
                    self.meta = json.load(f)

        self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        self.obs_dim = self.meta.get("obs_dim", contract.OBS_DIM_PROMO_V2)
        self.hist_len = self.meta.get("hist_len", contract.HIST_LEN_PROMO_V2)
        self.action_scale = float(self.meta.get("action_scale", contract.ACTION_SCALE_PROMO_V2))
        self.kp = float(self.meta.get("kp", contract.KP))
        self.kd = float(self.meta.get("kd", contract.KD))

        # Default q
        if "default_q" in self.meta and isinstance(self.meta["default_q"], dict):
            self.default_q = contract.DEFAULT_JOINT_POS_PROMO_V2.copy()
        else:
            self.default_q = contract.DEFAULT_JOINT_POS_PROMO_V2.copy()

        # Preference vector w
        if w is None:
            self.w = contract.DEFAULT_W_PROMO_V2.copy()
        else:
            self.w = np.asarray(w, dtype=np.float32)

        # Warm up session to eliminate first-call jitter
        dummy_obs = np.zeros((1, self.obs_dim), dtype=np.float32)
        dummy_hist = np.zeros((1, self.hist_len, self.obs_dim), dtype=np.float32)
        feed = {"obs": dummy_obs, "hist": dummy_hist}
        if "w" in self.input_names:
            feed["w"] = self.w.reshape(1, -1)
        self.session.run(None, feed)

    def set_preference(self, w):
        self.w = np.asarray(w, dtype=np.float32)

    def action_mean(self, obs, hist, w=None):
        """Run policy inference.

        obs: (obs_dim,) float32
        hist: (hist_len, obs_dim) float32
        w: (3,) float32 (optional override)
        Returns: action (12,)
        """
        pref = np.asarray(w if w is not None else self.w, dtype=np.float32).reshape(1, -1)
        feed = {
            "obs": np.asarray(obs, dtype=np.float32).reshape(1, -1),
            "hist": np.asarray(hist, dtype=np.float32).reshape(1, self.hist_len, self.obs_dim),
        }
        if "w" in self.input_names:
            feed["w"] = pref
        action = self.session.run(None, feed)[0]
        return action.reshape(-1)


def elu(x):
    return np.where(x > 0.0, x, np.exp(x) - 1.0)


class EluMlp(object):
    """Linear, ELU, ..., Linear for PROMO networks."""

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


class PromoNumpyPolicy(object):
    """Pure NumPy inference engine for PROMO v2 models (base_v2, oursA).
    Runs with ZERO external dependencies (no PyTorch, no ONNX Runtime), perfect for Python 3.5 on A1.
    """

    def __init__(self, npz_path, w=None):
        data = np.load(npz_path, allow_pickle=False)
        self.meta = json.loads(str(data["meta"]))
        self.obs_dim = self.meta.get("obs_dim", contract.OBS_DIM_PROMO_V2)
        self.hist_len = self.meta.get("hist_len", contract.HIST_LEN_PROMO_V2)
        self.vel_hist = self.meta.get("vel_hist", 5)
        self.latent = self.meta.get("latent", 45)
        self.film_actor = bool(self.meta.get("film_actor", False))
        self.action_scale = float(self.meta.get("action_scale", contract.ACTION_SCALE_PROMO_V2))
        self.kp = float(self.meta.get("kp", contract.KP))
        self.kd = float(self.meta.get("kd", contract.KD))
        self.default_q = np.array(self.meta.get("default_q", contract.DEFAULT_JOINT_POS_PROMO_V2), dtype=np.float32)

        # Build hist_enc (3 layers)
        self.hist_enc = EluMlp(
            [data["hist_enc.0.weight"], data["hist_enc.1.weight"], data["hist_enc.2.weight"]],
            [data["hist_enc.0.bias"], data["hist_enc.1.bias"], data["hist_enc.2.bias"]],
        )

        # Build vel_est (3 layers)
        self.vel_est = EluMlp(
            [data["vel_est.0.weight"], data["vel_est.1.weight"], data["vel_est.2.weight"]],
            [data["vel_est.0.bias"], data["vel_est.1.bias"], data["vel_est.2.bias"]],
        )

        # Build actor
        if self.film_actor:
            self.film_net = EluMlp(
                [data["film.0.weight"], data["film.1.weight"]],
                [data["film.0.bias"], data["film.1.bias"]],
            )
            self.actor_layers = [
                (data["actor_layer.0.weight"], data["actor_layer.0.bias"]),
                (data["actor_layer.1.weight"], data["actor_layer.1.bias"]),
                (data["actor_layer.2.weight"], data["actor_layer.2.bias"]),
            ]
            self.actor_out = (data["actor_out.weight"], data["actor_out.bias"])
        else:
            self.actor = EluMlp(
                [data["actor.0.weight"], data["actor.1.weight"], data["actor.2.weight"], data["actor.3.weight"]],
                [data["actor.0.bias"], data["actor.1.bias"], data["actor.2.bias"], data["actor.3.bias"]],
            )

        self.w = np.asarray(w if w is not None else contract.DEFAULT_W_PROMO_V2, dtype=np.float32)

        # Warmup
        self.action_mean(np.zeros(self.obs_dim, dtype=np.float32),
                         np.zeros((self.hist_len, self.obs_dim), dtype=np.float32),
                         self.w)

    def set_preference(self, w):
        self.w = np.asarray(w, dtype=np.float32)

    def encode_history(self, hist, w):
        """hist: (H, obs_dim), w: (3,). Returns z_H (latent,), v_hat (3,)."""
        hist_flat = hist.reshape(-1)
        z_H = self.hist_enc(np.concatenate([hist_flat, w]))
        recent_obs = hist[-self.vel_hist:].reshape(-1)
        v_hat = self.vel_est(np.concatenate([z_H, recent_obs]))
        return z_H, v_hat

    def action_mean(self, obs, hist, w=None):
        """obs: (obs_dim,), hist: (hist_len, obs_dim), w: (3,). Returns action (12,)."""
        pref = np.asarray(w if w is not None else self.w, dtype=np.float32)
        z_H, v_hat = self.encode_history(hist, pref)

        if self.film_actor:
            gb = self.film_net(pref)
            hidden_dims = [l[0].shape[0] for l in self.actor_layers]
            x = np.concatenate([obs, z_H, v_hat])
            pos = 0
            for (weight, bias), n in zip(self.actor_layers, hidden_dims):
                g = gb[pos:pos + n]
                b = gb[pos + n:pos + 2 * n]
                pos += 2 * n
                h = x.dot(weight.T) + bias
                x = elu(h) * (1.0 + g) + b
            w_out, b_out = self.actor_out
            action = x.dot(w_out.T) + b_out
        else:
            x = np.concatenate([obs, z_H, v_hat, pref])
            action = self.actor(x)

        return action.astype(np.float32)


def load_policy(path, w=None):
    """Factory function: returns OnnxPolicy for .onnx, PromoNumpyPolicy for promo .npz, or RmaPolicy for rma .npz."""
    if path.endswith(".onnx"):
        import os
        meta_cand = os.path.join(os.path.dirname(path), "policy_meta.json")
        if not os.path.exists(meta_cand):
            meta_cand = None
        return OnnxPolicy(path, meta_path=meta_cand, w=w)
    
    # Check if npz is promo numpy format
    data = np.load(path, allow_pickle=False)
    if "meta" in data:
        meta_str = str(data["meta"])
        if "promo" in meta_str or "hist_enc.0.weight" in data:
            return PromoNumpyPolicy(path, w=w)
    return RmaPolicy(path)

