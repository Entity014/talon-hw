"""Export deployable PROMO policy (base_v2, oursA) to pure NumPy .npz file.
This allows running inference on the A1 onboard computer without onnxruntime or PyTorch.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from talon_rl.promo.networks import model_from_state

def export_promo_numpy(checkpoint_path: Path, out_path: Path, thigh_front: float = 0.8, thigh_rear: float = 1.0):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ac = model_from_state(state["model"]).eval()
    ac.load_state_dict(state["model"])
    d = ac.d

    weights = {}
    
    # History encoder: mlp([d.hist * d.obs + K, *h, d.latent]) -> Linear layers at 0, 2, 4
    for i, idx in enumerate([0, 2, 4]):
        weights[f"hist_enc.{i}.weight"] = ac.hist_enc[idx].weight.detach().numpy().astype(np.float32)
        weights[f"hist_enc.{i}.bias"] = ac.hist_enc[idx].bias.detach().numpy().astype(np.float32)

    # Velocity estimator: mlp([d.latent + d.vel_hist * d.obs, *h, 3]) -> Linear layers at 0, 2, 4
    for i, idx in enumerate([0, 2, 4]):
        weights[f"vel_est.{i}.weight"] = ac.vel_est[idx].weight.detach().numpy().astype(np.float32)
        weights[f"vel_est.{i}.bias"] = ac.vel_est[idx].bias.detach().numpy().astype(np.float32)

    # Actor: either FiLMActor or standard MLP
    if d.film:
        weights["film_actor"] = np.array(True)
        # film net: 0, 2
        weights["film.0.weight"] = ac.actor.film[0].weight.detach().numpy().astype(np.float32)
        weights["film.0.bias"] = ac.actor.film[0].bias.detach().numpy().astype(np.float32)
        weights["film.1.weight"] = ac.actor.film[2].weight.detach().numpy().astype(np.float32)
        weights["film.1.bias"] = ac.actor.film[2].bias.detach().numpy().astype(np.float32)
        # layers: 0, 1, 2
        for i, l in enumerate(ac.actor.layers):
            weights[f"actor_layer.{i}.weight"] = l.weight.detach().numpy().astype(np.float32)
            weights[f"actor_layer.{i}.bias"] = l.bias.detach().numpy().astype(np.float32)
        weights["actor_out.weight"] = ac.actor.out.weight.detach().numpy().astype(np.float32)
        weights["actor_out.bias"] = ac.actor.out.bias.detach().numpy().astype(np.float32)
    else:
        weights["film_actor"] = np.array(False)
        # mlp has linear layers at 0, 2, 4, 6
        for i, idx in enumerate([0, 2, 4, 6]):
            weights[f"actor.{i}.weight"] = ac.actor[idx].weight.detach().numpy().astype(np.float32)
            weights[f"actor.{i}.bias"] = ac.actor[idx].bias.detach().numpy().astype(np.float32)

    meta = {
        "format": "promo_numpy_v1",
        "checkpoint": str(checkpoint_path),
        "obs_dim": int(d.obs),
        "hist_len": int(d.hist),
        "vel_hist": int(d.vel_hist),
        "latent": int(d.latent),
        "film_actor": bool(d.film),
        "action_scale": 0.25,
        "kp": 55.0,
        "kd": 0.8,
        "default_q": [0.1, -0.1, 0.1, -0.1, thigh_front, thigh_front, thigh_rear, thigh_rear, -1.5, -1.5, -1.5, -1.5]  # the default pose the policy was trained with (oursA / base_v2: 0.8 / 1.0; oursA_v2: 0.9 / 0.9)
    }
    weights["meta"] = np.array(json.dumps(meta))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **weights)
    print(f"Exported pure NumPy policy to {out_path}")

    # Parity check vs PyTorch
    obs = torch.randn(1, d.obs)
    hist = torch.randn(1, d.hist, d.obs)
    w = torch.tensor([[0.333, 0.333, 0.334]])
    with torch.no_grad():
        z_pt, v_pt = ac.encode_history(hist, w, None)
        a_pt = ac.action_mean(obs, z_pt, v_pt, w).numpy().reshape(-1)

    # Evaluate numpy forward
    from talon_a1.policy import PromoNumpyPolicy
    p_np = PromoNumpyPolicy(str(out_path))
    a_np = p_np.action_mean(obs.numpy().reshape(-1), hist.numpy().reshape(d.hist, d.obs), w.numpy().reshape(-1))

    diff = np.abs(a_pt - a_np).max()
    print(f"Max abs parity error against PyTorch: {diff:.2e}")
    assert diff < 1e-5, f"Parity failed: {diff}"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--thigh-front", type=float, default=0.8, help="default front thigh angle used in training (oursA_v2: 0.9)")
    p.add_argument("--thigh-rear", type=float, default=1.0, help="default rear thigh angle used in training (oursA_v2: 0.9)")
    args = p.parse_args()
    export_promo_numpy(Path(args.checkpoint), Path(args.out), args.thigh_front, args.thigh_rear)
