"""Export PROMO's deployable path from a talon-rl research/v1 checkpoint to the NumPy policy file.

Runs on the training machine, inside the talon-rl environment (needs torch and talon_rl):

    uv run --project ../../talon-rl python scripts/export_policy.py \
        --checkpoint ../../talon-rl/runs/<run>/final.pt --out policies/promo_a1.npz

Writes <out> and <out minus .npz>.parity.json. The parity check feeds the same random inputs through the
PyTorch model and the NumPy port; the export fails if their action means differ by more than --tol.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talon_a1 import contract  # noqa: E402
from talon_a1.policy import NETS, PromoPolicy, save_policy  # noqa: E402
from talon_rl.promo.networks import PromoActorCritic, PromoDims  # noqa: E402
from talon_rl.promo.preference import K, sample_preferences  # noqa: E402


def linear_layers(seq: torch.nn.Sequential) -> list[torch.nn.Linear]:
    """Checks the Linear, ELU, ..., Linear pattern the NumPy Mlp assumes."""
    mods = list(seq)
    for i, m in enumerate(mods):
        expected = torch.nn.Linear if i % 2 == 0 else torch.nn.ELU
        if not isinstance(m, expected):
            raise TypeError(f"layer {i} is {type(m).__name__}, expected {expected.__name__}")
        if isinstance(m, torch.nn.ELU) and m.alpha != 1.0:
            raise ValueError(f"ELU alpha {m.alpha} != 1")
    if not isinstance(mods[-1], torch.nn.Linear):
        raise TypeError("last layer must be Linear")
    return mods[::2]


def check_dims(d: PromoDims, manifest: dict | None) -> None:
    ours = {"obs": contract.OBS_DIM, "act": contract.ACT_DIM, "hist": contract.HIST,
            "vel_hist": contract.VEL_HIST, "latent": contract.LATENT}
    theirs = {k: getattr(d, k) for k in ours}
    if ours != theirs or K != contract.PREF_DIM:
        raise ValueError(f"PromoDims {theirs}, K={K} disagree with contract.py {ours}, {contract.PREF_DIM}")
    if manifest:
        for key, want in (("policy_observation_dim", contract.OBS_DIM), ("preference_dim", contract.PREF_DIM),
                          ("action_dim", contract.ACT_DIM)):
            if key in manifest and manifest[key] != want:
                raise ValueError(f"checkpoint manifest {key}={manifest[key]}, contract.py says {want}")
        if manifest.get("task") not in (None, "TalonRl-Promo-A1"):
            print(f"warning: checkpoint task is {manifest['task']!r}, not TalonRl-Promo-A1")


def parity(ac: PromoActorCritic, out: Path, n: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    anchors = torch.tensor(list(contract.PREFERENCE_ANCHORS.values()), dtype=torch.float32)
    prefs = torch.cat([anchors, sample_preferences(n - len(anchors), generator=g)])
    hist = torch.randn(n, contract.HIST, contract.OBS_DIM, generator=g)
    obs = hist[:, -1]  # o_t is the newest history entry, as at deployment
    with torch.inference_mode():
        z_h, v_t = ac.encode_history(hist, prefs)
        a_t = ac.action_mean(obs, z_h, v_t, prefs)
    a_err = v_err = 0.0
    policy = PromoPolicy(out, prefs[0].numpy())
    for i in range(n):
        policy.set_preference(prefs[i].numpy())
        a_np, v_np = policy.action_mean(obs[i].numpy(), hist[i].numpy())
        a_err = max(a_err, float(np.abs(a_np - a_t[i].numpy()).max()))
        v_err = max(v_err, float(np.abs(v_np - v_t[i].numpy()).max()))
    return {"samples": n, "seed": seed, "max_abs_action_error": a_err, "max_abs_v_hat_error": v_err}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    d = PromoDims()
    check_dims(d, state.get("manifest"))
    ac = PromoActorCritic(d)
    ac.load_state_dict(state["model"])
    ac.eval()

    layers = {net: [(m.weight.detach().numpy(), m.bias.detach().numpy()) for m in linear_layers(getattr(ac, net))]
              for net in NETS}
    meta = {"format": 1, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha,
            "iteration": state.get("iteration"), "manifest": state.get("manifest")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_policy(args.out, layers, meta)

    report = parity(ac, args.out, args.samples, args.seed)
    report.update(checkpoint_sha256=sha, iteration=state.get("iteration"), tol=args.tol,
                  policy_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                  verdict="PASS" if report["max_abs_action_error"] <= args.tol else "FAIL")
    report_path = args.out.with_suffix(".parity.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["verdict"] != "PASS":
        sys.exit(f"parity FAIL: max action error {report['max_abs_action_error']:.3g} > {args.tol}")


if __name__ == "__main__":
    main()
