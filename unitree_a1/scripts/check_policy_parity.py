"""Check parity and integrity across PyTorch (.pt), ONNX (.onnx), and NumPy (.npz) PROMO policies.

Implements 3 levels of checks:
  1. Static Check: zeros, ones, normal(0,1), uniform(-10,10) inputs -> Max Abs / Mean Abs / Max Rel diff.
  2. Layer-by-Layer Check (--layer-check, or auto on failure): compares hist_enc, vel_est, film, actor layers.
  3. Real-Log Replay Check (--replay-log): replays recorded obs/w sequence through .pt, .onnx, .npz.

Usage:
    python3 scripts/check_policy_parity.py --dir policies/base_v2 --layer-check
    python3 scripts/check_policy_parity.py --dir policies/oursA --layer-check
    python3 scripts/check_policy_parity.py --dir policies/oursA --replay-log logs/run_20261127_004201.npz
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
THESIS_ROOT = REPO_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))
if (THESIS_ROOT / "talon-rl" / "src").exists():
    sys.path.insert(0, str(THESIS_ROOT / "talon-rl" / "src"))

from talon_a1 import contract
from talon_a1.policy import OnnxPolicy, PromoNumpyPolicy, elu


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_single(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def load_pt_policy(pt_path: Path):
    import torch
    from talon_rl.promo.networks import model_from_state

    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    ac = model_from_state(state["model"]).eval()
    ac.load_state_dict(state["model"])
    return ac


def eval_pt(ac, obs: np.ndarray, hist: np.ndarray, w: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        hist_t = torch.from_numpy(hist.astype(np.float32)).unsqueeze(0)
        w_t = torch.from_numpy(w.astype(np.float32)).unsqueeze(0)
        z_pt, v_pt = ac.encode_history(hist_t, w_t, None)
        return ac.action_mean(obs_t, z_pt, v_pt, w_t).numpy().reshape(-1)


def check_metadata(meta_path: Path | None, npz_policy: PromoNumpyPolicy) -> list[str]:
    issues = []
    if meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for key, npz_val in [
            ("obs_dim", npz_policy.obs_dim),
            ("hist_len", npz_policy.hist_len),
            ("film_actor", npz_policy.film_actor),
        ]:
            if key in meta and (bool(meta[key]) != bool(npz_val) if key == "film_actor" else meta.get(key) != npz_val):
                issues.append(f"meta mismatch on {key}: policy_meta.json={meta.get(key)} vs npz={npz_val}")
    if npz_policy.obs_dim != contract.OBS_DIM_PROMO_V2:
        issues.append(f"obs_dim={npz_policy.obs_dim} != contract {contract.OBS_DIM_PROMO_V2}")
    if npz_policy.hist_len != contract.HIST_LEN_PROMO_V2:
        issues.append(f"hist_len={npz_policy.hist_len} != contract {contract.HIST_LEN_PROMO_V2}")
    if not np.allclose(npz_policy.default_q, contract.DEFAULT_JOINT_POS_PROMO_V2):
        issues.append(f"default_q={npz_policy.default_q.tolist()} != contract {contract.DEFAULT_JOINT_POS_PROMO_V2.tolist()}")
    return issues


def diff_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    abs_d = np.abs(a - b)
    rel_d = abs_d / (np.abs(a) + 1e-8)
    return float(np.max(abs_d)), float(np.mean(abs_d)), float(np.max(rel_d))


def layer_by_layer_check(ac, npz_p: PromoNumpyPolicy, obs: np.ndarray, hist: np.ndarray, w: np.ndarray, tol: float) -> bool:
    import torch

    print("[LAYER]  Comparing PyTorch vs NumPy submodule-by-submodule:")
    ok = True
    with torch.no_grad():
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        hist_t = torch.from_numpy(hist.astype(np.float32)).unsqueeze(0)
        w_t = torch.from_numpy(w.astype(np.float32)).unsqueeze(0)
        z_pt_t, v_pt_t = ac.encode_history(hist_t, w_t, None)
        z_pt = z_pt_t.numpy().reshape(-1)
        v_pt = v_pt_t.numpy().reshape(-1)

    z_np, v_np = npz_p.encode_history(hist, w)
    for label, a, b in [("hist_enc -> z_H", z_pt, z_np), ("vel_est  -> v_hat", v_pt, v_np)]:
        mx, mn, rl = diff_metrics(a, b)
        st = "OK" if mx <= tol else "FAIL"
        print(f"         {label:20s}: max_abs={mx:.2e}, mean_abs={mn:.2e}, max_rel={rl:.2e} [{st}]")
        if mx > tol:
            ok = False

    if npz_p.film_actor:
        with torch.no_grad():
            gb_pt = ac.actor.film(w_t).numpy().reshape(-1)
        gb_np = npz_p.film_net(w)
        mx, mn, rl = diff_metrics(gb_pt, gb_np)
        st = "OK" if mx <= tol else "FAIL"
        print(f"         {'film_net -> gb':20s}: max_abs={mx:.2e}, mean_abs={mn:.2e}, max_rel={rl:.2e} [{st}]")
        if mx > tol:
            ok = False

        with torch.no_grad():
            x_t = torch.cat([obs_t, z_pt_t, v_pt_t], dim=-1)
            x_np = np.concatenate([obs, z_np, v_np])
            pos = 0
            for i, (pt_layer, (w_np, b_np)) in enumerate(zip(ac.actor.layers, npz_p.actor_layers)):
                n = w_np.shape[0]
                g_t = ac.actor.film(w_t)[:, pos : pos + n]
                b_t = ac.actor.film(w_t)[:, pos + n : pos + 2 * n]
                x_t = torch.nn.functional.elu(pt_layer(x_t)) * (1.0 + g_t) + b_t

                g_np = gb_np[pos : pos + n]
                b_np_f = gb_np[pos + n : pos + 2 * n]
                pos += 2 * n
                x_np = elu(x_np.dot(w_np.T) + b_np) * (1.0 + g_np) + b_np_f

                mx, mn, rl = diff_metrics(x_t.numpy().reshape(-1), x_np)
                st = "OK" if mx <= tol else "FAIL"
                print(f"         {f'actor_layer.{i}':20s}: max_abs={mx:.2e}, mean_abs={mn:.2e}, max_rel={rl:.2e} [{st}]")
                if mx > tol:
                    ok = False

            a_pt = ac.actor.out(x_t).numpy().reshape(-1)
            w_out, b_out = npz_p.actor_out
            a_np = x_np.dot(w_out.T) + b_out
            mx, mn, rl = diff_metrics(a_pt, a_np)
            st = "OK" if mx <= tol else "FAIL"
            print(f"         {'actor_out':20s}: max_abs={mx:.2e}, mean_abs={mn:.2e}, max_rel={rl:.2e} [{st}]")
            if mx > tol:
                ok = False
    else:
        with torch.no_grad():
            x_t = torch.cat([obs_t, z_pt_t, v_pt_t, w_t], dim=-1)
            x_np = np.concatenate([obs, z_np, v_np, w])
            last = len(npz_p.actor.weights) - 1
            for i, (w_np, b_np) in enumerate(zip(npz_p.actor.weights, npz_p.actor.biases)):
                pt_l = ac.actor[i * 2]
                x_t = pt_l(x_t)
                x_np = x_np.dot(w_np.T) + b_np
                if i < last:
                    x_t = torch.nn.functional.elu(x_t)
                    x_np = elu(x_np)
                mx, mn, rl = diff_metrics(x_t.numpy().reshape(-1), x_np)
                st = "OK" if mx <= tol else "FAIL"
                print(f"         {f'actor.{i}':20s}: max_abs={mx:.2e}, mean_abs={mn:.2e}, max_rel={rl:.2e} [{st}]")
                if mx > tol:
                    ok = False

    return ok


def replay_log_check(log_path: Path, models: dict, hist_len: int, tol: float) -> bool:
    print(f"[REPLAY] Replaying real log: {log_path.name}")
    log = np.load(log_path, allow_pickle=False)
    obs_seq = log["obs"].astype(np.float32)
    w_seq = log["w"].astype(np.float32)
    act_logged = log["action"].astype(np.float32)
    mode_seq = log["mode"] if "mode" in log else None

    # Filter to active policy ticks (where action is finite and mode is WALK/STAND if available)
    valid = np.all(np.isfinite(obs_seq), axis=1) & np.all(np.isfinite(act_logged), axis=1)
    if mode_seq is not None:
        valid &= np.isin(mode_seq, ["WALK", "STAND"])
    idxs = np.where(valid)[0]
    if len(idxs) == 0:
        print("[REPLAY] WARN: no valid policy ticks found in log.")
        return True

    hist = np.repeat(obs_seq[idxs[0]][None, :], hist_len, axis=0)
    names = list(models.keys())
    outs = {n: [] for n in names}
    logged = []

    for step_i, t_idx in enumerate(idxs):
        o = obs_seq[t_idx]
        w = w_seq[t_idx] if w_seq.ndim == 2 else w_seq
        if step_i > 0:
            hist = np.concatenate([hist[1:], o[None, :]], axis=0)
        for n, fn in models.items():
            outs[n].append(fn(o, hist, w))
        logged.append(act_logged[t_idx])

    ok = True
    logged_arr = np.asarray(logged, dtype=np.float32)
    for i, a in enumerate(names):
        arr_a = np.asarray(outs[a], dtype=np.float32)
        for b in names[i + 1 :]:
            arr_b = np.asarray(outs[b], dtype=np.float32)
            mx, mn, rl = diff_metrics(arr_a, arr_b)
            st = "OK" if mx <= tol else "FAIL"
            print(f"         {a:4s} vs {b:4s} ({len(idxs)} ticks): max_abs={mx:.2e}, mean_abs={mn:.2e}, max_rel={rl:.2e} [{st}]")
            if mx > tol:
                ok = False
        mx_l, mn_l, _ = diff_metrics(arr_a, logged_arr)
        print(f"         {a:4s} vs logged action: max_abs={mx_l:.2e}, mean_abs={mn_l:.2e}")
    return ok


def mujoco_rollout_check(models: dict, xml_path: Path, steps: int, q_tol: float = 2e-3) -> bool:
    import mujoco

    if not xml_path.exists():
        print(f"[MUJOCO] FAIL: scene.xml not found at {xml_path}")
        return False

    print(f"[MUJOCO] Closed-loop {steps}-step rollout ({steps * 0.02:.1f}s @ 50Hz) across {list(models.keys())}:")
    joints = [f"{leg}_{kind}_joint" for kind in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]

    def def_q(n: str) -> float:
        if "hip" in n:
            return -0.1 if n.startswith(("FR", "RR")) else 0.1
        return (0.8 if n.startswith("F") else 1.0) if "thigh" in n else -1.5

    def one_rollout(fn):
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        m.opt.timestep = 0.005
        m.actuator_gainprm[:, 0] = 55.0
        m.actuator_biasprm[:, :3] = [0.0, -55.0, -0.8]
        m.dof_damping[6:] = 0.0
        m.dof_frictionloss[6:] = 0.0
        m.actuator_ctrllimited[:] = 0
        m.actuator_forcerange[:] = [-33.5, 33.5]
        m.actuator_forcelimited[:] = 1
        jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
        aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).replace("_joint", ""): i for i in range(m.nu)}
        qadr = np.array([m.jnt_qposadr[jid[n]] for n in joints])
        vadr = np.array([m.jnt_dofadr[jid[n]] for n in joints])
        aord = np.array([aid[n.replace("_joint", "")] for n in joints])
        q0 = np.array([def_q(n) for n in joints], dtype=np.float32)

        d = mujoco.MjData(m)
        d.qpos[2] = 0.30
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[qadr] = q0
        mujoco.mj_forward(m, d)

        cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        w = np.array([0.8, 0.1, 0.1], dtype=np.float32)
        last_act = np.zeros(12, dtype=np.float32)
        hist = None
        qs, acts, vxs = [], [], []
        for s in range(steps):
            R = d.xmat[1].reshape(3, 3)
            obs = np.concatenate([R.T @ d.qvel[3:6], R.T @ [0.0, 0.0, -1.0], cmd, d.qpos[qadr] - q0, d.qvel[vadr], last_act]).astype(np.float32)
            hist = np.repeat(obs[None, :], 10, axis=0) if hist is None else np.concatenate([hist[1:], obs[None, :]], axis=0)
            act = fn(obs, hist, w)
            last_act = act
            d.ctrl[aord] = q0 + 0.25 * act
            for _ in range(4):
                mujoco.mj_step(m, d)
            qs.append(d.qpos[qadr].copy())
            acts.append(act.copy())
            vxs.append((R.T @ d.qvel[:3])[0])
            if d.qpos[2] < 0.12:
                return np.array(qs), np.array(acts), np.array(vxs), False, s
        return np.array(qs), np.array(acts), np.array(vxs), True, steps

    results = {}
    ok = True
    for name, fn in models.items():
        qs, acts, vxs, survived, st = one_rollout(fn)
        results[name] = (qs, acts, vxs, survived, st)
        print(f"         {name:4s}: survived={survived} ({st}/{steps}), mean_vx={vxs.mean():+.3f} m/s")
        if not survived:
            ok = False

    names = list(models.keys())
    for i, a in enumerate(names):
        qa, aa = results[a][0], results[a][1]
        for b in names[i + 1 :]:
            qb, ab = results[b][0], results[b][1]
            n = min(len(qa), len(qb))
            n200 = min(200, n)
            q_diff_200 = float(np.abs(qa[:n200] - qb[:n200]).max())
            q_diff_all = float(np.abs(qa[:n] - qb[:n]).max())
            a_diff_all = float(np.abs(aa[:n] - ab[:n]).max())
            st = "OK" if q_diff_200 <= q_tol else "FAIL"
            print(f"         {a:4s} vs {b:4s}: q_max_200={q_diff_200:.2e}, q_max_{n}={q_diff_all:.2e}, act_max={a_diff_all:.2e} [{st}]")
            if q_diff_200 > q_tol:
                ok = False
    return ok


def run_check(
    policy_dir: Path,
    pt_path: Path | None,
    onnx_path: Path | None,
    npz_path: Path | None,
    meta_path: Path | None,
    ref_path: Path | None,
    n_normal: int,
    n_uniform: int,
    tol: float,
    do_layer_check: bool,
    replay_log: Path | None,
    mujoco_xml: Path | None = None,
    mujoco_steps: int = 0,
) -> bool:
    print(f"=== Checking policy directory: {policy_dir} ===")
    ok = True
    meta = json.loads(meta_path.read_text()) if meta_path and meta_path.exists() else {}

    # 1. SHA-256 checks
    if pt_path and pt_path.exists() and "checkpoint_sha256" in meta:
        actual = sha256_file(pt_path)
        expected = meta["checkpoint_sha256"]
        match = actual == expected
        print(f"[SHA256] .pt   ({pt_path.name}): {'MATCH' if match else f'MISMATCH ({actual[:12]} != {expected[:12]})'}")
        if not match:
            ok = False
    if onnx_path and onnx_path.exists() and "onnx_sha256" in meta:
        actual = sha256_file(onnx_path)
        expected = meta["onnx_sha256"]
        match = actual == expected
        print(f"[SHA256] .onnx ({onnx_path.name}): {'MATCH' if match else f'MISMATCH ({actual[:12]} != {expected[:12]})'}")
        if not match:
            ok = False

    # 2. Load models
    models = {}
    ac = None
    if pt_path and pt_path.exists():
        try:
            ac = load_pt_policy(pt_path)
            models["pt"] = lambda obs, hist, w, _ac=ac: eval_pt(_ac, obs, hist, w)
            print(f"[LOAD]   .pt   loaded: {pt_path.name}")
        except Exception as e:
            print(f"[LOAD]   .pt   FAILED: {e}")
            ok = False

    if onnx_path and onnx_path.exists():
        try:
            onnx_p = OnnxPolicy(str(onnx_path), meta_path=str(meta_path) if meta_path else None)
            models["onnx"] = lambda obs, hist, w, _p=onnx_p: _p.action_mean(obs, hist, w)
            print(f"[LOAD]   .onnx loaded: {onnx_path.name}")
        except Exception as e:
            print(f"[LOAD]   .onnx FAILED: {e}")
            ok = False

    npz_p = None
    if npz_path and npz_path.exists():
        try:
            npz_p = PromoNumpyPolicy(str(npz_path))
            models["npz"] = lambda obs, hist, w, _p=npz_p: _p.action_mean(obs, hist, w)
            print(f"[LOAD]   .npz  loaded: {npz_path.name} (film_actor={npz_p.film_actor})")
        except Exception as e:
            print(f"[LOAD]   .npz  FAILED: {e}")
            ok = False

    if npz_p is not None:
        meta_issues = check_metadata(meta_path, npz_p)
        if meta_issues:
            for issue in meta_issues:
                print(f"[META]   FAIL: {issue}")
            ok = False
        else:
            print("[META]   OK (matches policy_meta.json and contract.py)")

    if len(models) < 2:
        print("[WARN]   Need at least 2 model formats to compare parity.")
        return ok

    # 3. Reference I/O check
    if ref_path and ref_path.exists():
        ref = np.load(ref_path)
        obs0, hist0, w0, act_ref = ref["obs"][0], ref["hist"][0], ref["w"][0], ref["action"][0]
        for name, fn in models.items():
            act = fn(obs0, hist0, w0)
            mx, mn, rl = diff_metrics(act_ref, act)
            st = "OK" if mx <= tol else "FAIL"
            print(f"[REF_IO] {name:4s} vs reference_io.npz: max_abs={mx:.2e}, max_rel={rl:.2e} [{st}]")
            if mx > tol:
                ok = False

    # 4. Level 1 Static Check: zeros, ones, normal(0,1), uniform(-10, 10)
    rng = np.random.default_rng(42)
    obs_dim = npz_p.obs_dim if npz_p else contract.OBS_DIM_PROMO_V2
    hist_len = npz_p.hist_len if npz_p else contract.HIST_LEN_PROMO_V2

    obs_batch = np.vstack([
        np.zeros((1, obs_dim), dtype=np.float32),
        np.ones((1, obs_dim), dtype=np.float32),
        rng.standard_normal((n_normal, obs_dim)).astype(np.float32),
        rng.uniform(-10.0, 10.0, (n_uniform, obs_dim)).astype(np.float32),
    ])
    hist_batch = np.vstack([
        np.zeros((1, hist_len, obs_dim), dtype=np.float32),
        np.ones((1, hist_len, obs_dim), dtype=np.float32),
        rng.standard_normal((n_normal, hist_len, obs_dim)).astype(np.float32),
        rng.uniform(-10.0, 10.0, (n_uniform, hist_len, obs_dim)).astype(np.float32),
    ])
    w_presets = [
        np.array([0.333, 0.333, 0.334], dtype=np.float32),
        np.array([0.8, 0.1, 0.1], dtype=np.float32),
        np.array([0.1, 0.8, 0.1], dtype=np.float32),
        np.array([0.1, 0.1, 0.8], dtype=np.float32),
    ]
    n_total = obs_batch.shape[0]
    w_batch = np.vstack([
        w_presets[i % len(w_presets)] if i < 8 else rng.dirichlet([1.0, 1.0, 1.0]).astype(np.float32)
        for i in range(n_total)
    ])

    outs = {name: [] for name in models}
    for i in range(n_total):
        for name, fn in models.items():
            out = fn(obs_batch[i], hist_batch[i], w_batch[i])
            if not np.all(np.isfinite(out)):
                print(f"[NUM]    FAIL: {name} produced NaN/Inf at sample {i}")
                ok = False
            outs[name].append(out)
    outs = {k: np.asarray(v, dtype=np.float32) for k, v in outs.items()}

    static_ok = True
    names = list(models.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            mx, mn, rl = diff_metrics(outs[a], outs[b])
            st = "OK" if mx <= tol else "FAIL"
            print(f"[STATIC] {a:4s} vs {b:4s} ({n_total} samples): max_abs={mx:.2e}, mean_abs={mn:.2e}, max_rel={rl:.2e} [{st}]")
            if mx > tol:
                static_ok = False
                ok = False

    # 5. Level 2 Layer-by-Layer Check
    if (do_layer_check or not static_ok) and ac is not None and npz_p is not None:
        if not layer_by_layer_check(ac, npz_p, obs_batch[2], hist_batch[2], w_batch[0], tol):
            ok = False

    # 6. Level 3 Real-Log Replay Check
    if replay_log is not None:
        if not replay_log_check(replay_log, models, hist_len, tol):
            ok = False

    # 7. Level 3 Closed-Loop MuJoCo Rollout Check
    if mujoco_steps > 0 and mujoco_xml is not None:
        if not mujoco_rollout_check(models, mujoco_xml, mujoco_steps):
            ok = False

    print(f"=== RESULT: {'PASS' if ok else 'FAIL'} ===\n")
    return ok


def main():
    p = argparse.ArgumentParser(description="Verify parity between .pt, .onnx, and .npz policy files.")
    p.add_argument("--dir", type=Path, required=True, help="Policy directory (e.g. policies/oursA)")
    p.add_argument("--pt", type=Path, default=None, help="Optional explicit .pt checkpoint path")
    p.add_argument("--onnx", type=Path, default=None, help="Optional explicit .onnx path")
    p.add_argument("--npz", type=Path, default=None, help="Optional explicit .npz path")
    p.add_argument("--samples", type=int, default=50, help="Number of normal(0,1) samples (plus zeros, ones, uniform)")
    p.add_argument("--uniform-samples", type=int, default=48, help="Number of uniform(-10, 10) edge samples")
    p.add_argument("--tol", type=float, default=1e-4, help="Max absolute action error tolerance")
    p.add_argument("--layer-check", action="store_true", help="Run layer-by-layer PyTorch vs NumPy comparison")
    p.add_argument("--replay-log", type=Path, default=None, help="Optional logs/run_*.npz to replay")
    p.add_argument("--mujoco-rollout", action="store_true", help="Run 500-step closed-loop MuJoCo rollout comparing .pt, .onnx, and .npz")
    p.add_argument("--mujoco-steps", type=int, default=500, help="Steps for --mujoco-rollout (50 Hz)")
    p.add_argument("--mujoco-xml", type=Path, default=THESIS_ROOT / "talon-rl" / "assets" / "mujoco_a1" / "scene.xml", help="MuJoCo scene.xml path")
    args = p.parse_args()

    d = args.dir
    pt = args.pt or (d / "final.pt" if (d / "final.pt").exists() else find_single(d, "*.pt"))
    onnx = args.onnx or (d / "policy.onnx")
    npz = args.npz or (d / "policy.npz")
    meta = d / "policy_meta.json"
    ref = d / "reference_io.npz"

    passed = run_check(
        d, pt, onnx, npz, meta, ref,
        args.samples, args.uniform_samples, args.tol,
        args.layer_check, args.replay_log,
        args.mujoco_xml if args.mujoco_rollout else None,
        args.mujoco_steps if args.mujoco_rollout else 0,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
