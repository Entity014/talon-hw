"""Evaluate a PROMO policy (.npz) in MuJoCo using Unitree A1 hardware System-ID parameters.

Default SI parameters calibrated against real-robot logs (2026-11-26):
  - delay_ms     = 10.0  (from latency_probe.py: 9.4 ms torque delay -> 2 substeps @ 200 Hz)
  - kd           = 0.75  (from Unitree A1 wire truncation int16(0.8 * 16) / 16)
  - frictionloss = 1.0   (Nm, Coulomb joint friction matching interactive1/2 logs)
  - damping      = 0.2   (Nm*s/rad, viscous joint damping)

Usage:
    python3 scripts/mujoco_si_eval.py --policy policies/oursA/checkpoint_04900.npz
    python3 scripts/mujoco_si_eval.py --policy policies/base_v2/policy.npz --ideal
"""

import argparse
from collections import deque
from pathlib import Path
import sys

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_XML = REPO_ROOT.parent.parent / "talon-rl" / "assets" / "mujoco_a1" / "scene.xml"

from talon_a1.policy import load_policy  # noqa: E402

JOINTS = [f"{leg}_{kind}_joint" for kind in ("hip", "thigh", "calf") for leg in ("FL", "FR", "RL", "RR")]
PREFERENCES = {
    "tracking": (0.8, 0.1, 0.1),
    "stability": (0.1, 0.8, 0.1),
    "efficiency": (0.1, 0.1, 0.8),
    "balanced": (0.333, 0.333, 0.334),
}


def default_q(name: str) -> float:
    if "hip" in name:
        return -0.1 if name.startswith(("FR", "RR")) else 0.1
    if "thigh" in name:
        return 0.8 if name.startswith("F") else 1.0
    return -1.5


def run_rollout(policy, xml_path: Path, w, cmd_vx: float, kp: float, kd: float, delay_ms: float, frictionloss: float, damping: float, action_smooth: float, steps: int):
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    m.opt.timestep = 0.005
    m.actuator_gainprm[:, 0] = kp
    m.actuator_biasprm[:, :3] = [0.0, -kp, -kd]
    m.dof_damping[6:] = damping
    m.dof_frictionloss[6:] = frictionloss
    m.actuator_ctrllimited[:] = 0
    m.actuator_forcerange[:] = [-33.5, 33.5]
    m.actuator_forcelimited[:] = 1

    jid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(m.njnt)}
    aid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i).replace("_joint", ""): i for i in range(m.nu)}
    qadr = np.array([m.jnt_qposadr[jid[n]] for n in JOINTS])
    vadr = np.array([m.jnt_dofadr[jid[n]] for n in JOINTS])
    aord = np.array([aid[n.replace("_joint", "")] for n in JOINTS])
    q0 = np.array([default_q(n) for n in JOINTS], dtype=np.float32)

    d = mujoco.MjData(m)
    d.qpos[2] = 0.30
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qpos[qadr] = q0
    mujoco.mj_forward(m, d)

    delay_substeps = max(0, int(round(delay_ms / 5.0)))
    ctrl_buf = deque([q0.copy()] * (delay_substeps + 1), maxlen=delay_substeps + 1)

    cmd = np.array([cmd_vx, 0.0, 0.0], dtype=np.float32)
    w_arr = np.array(w, dtype=np.float32)
    last_act = np.zeros(12, dtype=np.float32)
    hist = None

    vxs, gyros, powers, acts = [], [], [], []
    warmup_steps = min(100, steps // 4)
    for s in range(steps):
        R = d.xmat[1].reshape(3, 3)
        ang = R.T @ d.qvel[3:6]
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        obs = np.concatenate([ang, grav, cmd, d.qpos[qadr] - q0, d.qvel[vadr], last_act]).astype(np.float32)
        hist = np.repeat(obs[None, :], 10, axis=0) if hist is None else np.concatenate([hist[1:], obs[None, :]], axis=0)
        raw_act = policy.action_mean(obs, hist, w_arr)
        act = (1.0 - action_smooth) * raw_act + action_smooth * last_act
        last_act = act
        tgt = q0 + 0.25 * act
        for _ in range(4):
            ctrl_buf.append(tgt.copy())
            d.ctrl[aord] = ctrl_buf[0]
            mujoco.mj_step(m, d)
            if s >= warmup_steps:
                powers.append(np.sum(np.maximum(d.qfrc_actuator[vadr] * d.qvel[vadr], 0.0)))
        if s >= warmup_steps:
            vxs.append((R.T @ d.qvel[:3])[0])
            gyros.append(ang[:2].copy())
            acts.append(act.copy())
        if d.qpos[2] < 0.12:
            return dict(survived=False, steps=s, vx=0.0, power=0.0, gyro_rp=0.0, act_rate=0.0)

    ar = float(np.sqrt(np.mean((np.diff(np.array(acts), axis=0) / 0.02) ** 2)))
    return dict(
        survived=True,
        steps=steps,
        vx=float(np.mean(vxs)),
        power=float(np.mean(powers)),
        gyro_rp=float(np.mean(np.std(gyros, axis=0))),
        act_rate=ar,
    )


def main():
    p = argparse.ArgumentParser(description="Evaluate policy in MuJoCo with Unitree A1 hardware SI parameters.")
    p.add_argument("--policy", required=True, help="Path to .npz or .onnx policy")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML, help="MuJoCo A1 scene.xml")
    p.add_argument("--kp", type=float, default=55.0)
    p.add_argument("--kd", type=float, default=0.75)
    p.add_argument("--delay-ms", type=float, default=10.0)
    p.add_argument("--frictionloss", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=0.2)
    p.add_argument("--action-smooth", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--ideal", action="store_true", help="Override to Ideal Sim (0 ms, Kd=0.80, friction=0, damping=0)")
    args = p.parse_args()

    if args.ideal:
        args.delay_ms, args.kd, args.frictionloss, args.damping = 0.0, 0.80, 0.0, 0.0

    policy = load_policy(args.policy)
    mode = "IDEAL SIM" if args.ideal else "HW-SI SIM"
    print(f"=== {mode}: delay={args.delay_ms}ms, Kp={args.kp}, Kd={args.kd}, frictionloss={args.frictionloss}, damping={args.damping}, smooth={args.action_smooth} ===")
    for cmd_name, cmd_vx in [("move_fast (0.50 m/s)", 0.50), ("move_slow (0.25 m/s)", 0.25)]:
        print(f"--- {cmd_name} ---")
        for pname in ("tracking", "stability", "efficiency", "balanced"):
            r = run_rollout(policy, args.xml, PREFERENCES[pname], cmd_vx, args.kp, args.kd, args.delay_ms, args.frictionloss, args.damping, args.action_smooth, args.steps)
            status = "OK" if r["survived"] else f"FALL@{r['steps']}"
            print(f"  {pname:10s}: {status:8s} | vx={r['vx']:+.3f} m/s | power={r['power']:5.1f} W | gyro_rp={r['gyro_rp']:.3f} | act_rate={r['act_rate']:5.2f}")


if __name__ == "__main__":
    main()
