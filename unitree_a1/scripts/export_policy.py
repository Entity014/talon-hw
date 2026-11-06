"""Export deployment RMA from a talon-rl Phase-2 checkpoint to the NumPy policy file.

Runs on the training machine, with torch and the talon_rl package from the talon-rl deployment branch:

    PYTHONPATH=../../talon-rl/src python3 scripts/export_policy.py \
        --checkpoint policies/phase2_500.pt --out policies/rma_a1.npz

Writes <out> and <out minus .npz>.parity.json. The parity check feeds the same random inputs through the
PyTorch modules and the NumPy port; the export fails if their actions differ by more than --tol.
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
from talon_a1.policy import RmaPolicy, save_policy  # noqa: E402
from talon_rl.rma.networks import ACT, K_HIST, AdaptationModule, BasePolicy, VelocityEstimator  # noqa: E402

TASK = "TalonRl-PromoRma-A1"


def load_checkpoint(path: Path) -> dict:
    """weights_only: the checkpoint is data, so an unexpected pickle global is refused, not run."""
    numpy_globals = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType]
    with torch.serialization.safe_globals(numpy_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def check_dims(state: dict) -> None:
    manifest = state["run_manifest"]
    want = {"task": TASK, "phase": 2, "state_dim": contract.OBS_DIM, "velocity_dim": contract.VEL_DIM}
    wrong = {k: manifest.get(k) for k, v in want.items() if manifest.get(k) != v}
    if wrong:
        raise ValueError(f"checkpoint manifest {wrong} disagrees with {want}")
    if state.get("observation_normalizer") is not None or manifest.get("observation_normalization") != "none":
        raise ValueError("checkpoint uses observation normalization; the NumPy port and obs.py do not apply it")
    ours = (contract.ACT_DIM, contract.HIST)
    if ours != (ACT, K_HIST):
        raise ValueError(f"talon_rl networks (ACT, K_HIST) = {(ACT, K_HIST)}, contract.py says {ours}")


def linears(sd: dict, prefix: str, indices) -> list:
    return [(sd[f"{prefix}{i}.weight"].cpu().numpy(), sd[f"{prefix}{i}.bias"].cpu().numpy()) for i in indices]


def deployable_layers(state: dict) -> dict:
    """The deployable path only: the critic and the privileged encoder mu never reach the robot."""
    return {
        "phi_embed": linears(state["phi"], "embed.", (0, 2)),
        "phi_conv": linears(state["phi"], "cnn.", (0, 2, 4)),
        "phi_proj": [(state["phi"]["proj.weight"].cpu().numpy(), state["phi"]["proj.bias"].cpu().numpy())],
        "velocity": linears(state["velocity_estimator"], "net.", (0, 2, 4)),
        "actor": linears(state["pi"], "actor.", (0, 2, 4)),
    }


def load_modules(state: dict) -> tuple:
    d = contract.OBS_DIM
    pi, phi, velocity = BasePolicy(state_dim=d, velocity_dim=contract.VEL_DIM), AdaptationModule(d), VelocityEstimator(d)
    pi.load_state_dict(state["pi"])
    phi.load_state_dict(state["phi"])
    velocity.load_state_dict(state["velocity_estimator"])
    return pi.eval(), phi.eval(), velocity.eval()


def parity(modules: tuple, out: Path, n: int, seed: int) -> dict:
    pi, phi, velocity = modules
    g = torch.Generator().manual_seed(seed)
    hist = torch.randn(n, contract.HIST, contract.HIST_DIM, generator=g)
    x = torch.randn(n, contract.OBS_DIM, generator=g)
    a_prev = torch.randn(n, contract.ACT_DIM, generator=g).clamp(-1.0, 1.0)
    with torch.inference_mode():
        z_t = phi(hist)
        v_t = velocity(hist, z_t)
        a_t = pi(x, a_prev, z_t, v_t)[0]
    a_err = v_err = z_err = 0.0
    policy = RmaPolicy(out)
    for i in range(n):
        a_np, v_np, z_np = policy.action_mean(x[i].numpy(), a_prev[i].numpy(), hist[i].numpy())
        a_err = max(a_err, float(np.abs(a_np - a_t[i].numpy()).max()))
        v_err = max(v_err, float(np.abs(v_np - v_t[i].numpy()).max()))
        z_err = max(z_err, float(np.abs(z_np - z_t[i].numpy()).max()))
    return {"samples": n, "seed": seed, "max_abs_action_error": a_err, "max_abs_v_hat_error": v_err,
            "max_abs_z_hat_error": z_err}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    state = load_checkpoint(args.checkpoint)
    check_dims(state)
    modules = load_modules(state)

    meta = {"format": 2, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha, "iteration": state.get("it"),
            "task": state["run_manifest"]["task"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_policy(args.out, deployable_layers(state), meta)

    report = parity(modules, args.out, args.samples, args.seed)
    report.update(checkpoint_sha256=sha, iteration=state.get("it"), tol=args.tol,
                  policy_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
                  verdict="PASS" if report["max_abs_action_error"] <= args.tol else "FAIL")
    report_path = args.out.with_suffix(".parity.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["verdict"] != "PASS":
        sys.exit(f"parity FAIL: max action error {report['max_abs_action_error']:.3g} > {args.tol}")


if __name__ == "__main__":
    main()
