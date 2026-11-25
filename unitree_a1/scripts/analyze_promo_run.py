#!/usr/bin/env python3
"""Analyze run log npz from hardware or loopback runs to evaluate PROMO metrics:
- Power / Energy / Cost of Transport (CoT)
- Attitude deviation (Roll/Pitch RMS and Peak)
- Action & Torque smoothness (Rate RMS)
- Tracking error (Velocity vs command)

Usage:
    python3 scripts/analyze_promo_run.py --log logs/run_20261125_125152.npz
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def analyze_log(log_path, mass_kg=12.0):
    data = np.load(log_path, allow_pickle=True)
    mode = data["mode"]
    walk_idx = np.where(mode == "WALK")[0]
    if len(walk_idx) == 0:
        print(f"Warning: No WALK steps found in {log_path}. Using all steps.")
        walk_idx = np.arange(len(mode))

    t = data["t"][walk_idx]
    dt = np.diff(t)
    dt = np.where(dt > 0, dt, 0.02)
    dt_mean = float(np.mean(dt)) if len(dt) > 0 else 0.02

    cmd = data["command"][walk_idx]  # (N, 3): vx, vy, wz
    q = data["q"][walk_idx]          # (N, 12)
    dq = data["dq"][walk_idx]        # (N, 12)
    tau = data["tau_est"][walk_idx]  # (N, 12)
    rpy = data["rpy"][walk_idx]      # (N, 3): roll, pitch, yaw
    action = data["action"][walk_idx]  # (N, 12)

    w_val = None
    if "w" in data and not np.isnan(data["w"][walk_idx[0]]).any():
        w_val = data["w"][walk_idx[0]].tolist()

    # 1. Power & Energy (Mechanical power sum |tau * dq|)
    mech_power = np.sum(np.abs(tau * dq), axis=1)  # (N,)
    mean_power_w = float(np.mean(mech_power))
    peak_power_w = float(np.max(mech_power)) if len(mech_power) > 0 else 0.0
    duration_s = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    total_energy_j = float(np.sum(mech_power[:-1] * dt)) if len(dt) > 0 else 0.0

    # Distance estimation from commanded or integrated speed
    # On physical robot without mocap, odometry estimate from command:
    cmd_speed = np.linalg.norm(cmd[:, :2], axis=1)
    est_distance_m = float(np.sum(cmd_speed[:-1] * dt)) if len(dt) > 0 else 0.0
    if est_distance_m > 0.05:
        specific_energy_j_per_m = total_energy_j / est_distance_m
        cot = specific_energy_j_per_m / (mass_kg * 9.81)
    else:
        specific_energy_j_per_m = np.nan
        cot = np.nan

    # 2. Attitude Deviation (Roll, Pitch)
    roll = rpy[:, 0]
    pitch = rpy[:, 1]
    att_dev = np.sqrt(roll**2 + pitch**2)  # (N,)
    roll_rms_rad = float(np.sqrt(np.mean(roll**2)))
    pitch_rms_rad = float(np.sqrt(np.mean(pitch**2)))
    peak_att_rad = float(np.max(att_dev)) if len(att_dev) > 0 else 0.0
    mean_att_rad = float(np.mean(att_dev)) if len(att_dev) > 0 else 0.0

    # 3. Action & Torque Smoothness (Rate RMS)
    valid_act = ~np.isnan(action).any(axis=1)
    if np.sum(valid_act) > 1:
        act_valid = action[valid_act]
        t_valid = t[valid_act]
        dt_act = np.diff(t_valid)
        dt_act = np.where(dt_act > 0, dt_act, dt_mean)
        act_diff = np.diff(act_valid, axis=0) / dt_act[:, None]
        act_rate_rms = float(np.sqrt(np.mean(act_diff**2)))
    else:
        act_rate_rms = 0.0

    valid_tau = ~np.isnan(tau).any(axis=1)
    if np.sum(valid_tau) > 1:
        tau_valid = tau[valid_tau]
        t_valid_tau = t[valid_tau]
        dt_tau = np.diff(t_valid_tau)
        dt_tau = np.where(dt_tau > 0, dt_tau, dt_mean)
        tau_diff = np.diff(tau_valid, axis=0) / dt_tau[:, None]
        tau_rate_rms = float(np.sqrt(np.mean(tau_diff**2)))
    else:
        tau_rate_rms = 0.0

    # 4. Timing & Health
    compute_s = data["compute_s"][walk_idx]
    compute_p50_ms = float(np.percentile(compute_s, 50) * 1000)
    compute_p99_ms = float(np.percentile(compute_s, 99) * 1000)
    overruns = int(data["loop_overruns"][-1]) if len(data["loop_overruns"]) > 0 else 0

    results = {
        "log_path": str(log_path),
        "walk_steps": len(walk_idx),
        "duration_s": round(duration_s, 2),
        "preference_w": w_val,
        "efficiency": {
            "mean_power_w": round(mean_power_w, 2),
            "peak_power_w": round(peak_power_w, 2),
            "total_energy_j": round(total_energy_j, 2),
            "est_distance_m": round(est_distance_m, 2),
            "specific_energy_j_per_m": round(specific_energy_j_per_m, 2) if not np.isnan(specific_energy_j_per_m) else None,
            "cot": round(cot, 3) if not np.isnan(cot) else None,
            "act_rate_rms_rad_per_s": round(act_rate_rms, 2),
            "trq_rate_rms_nm_per_s": round(tau_rate_rms, 2),
        },
        "stability": {
            "roll_rms_rad": round(roll_rms_rad, 4),
            "pitch_rms_rad": round(pitch_rms_rad, 4),
            "mean_att_rad": round(mean_att_rad, 4),
            "peak_att_rad": round(peak_att_rad, 4),
        },
        "performance": {
            "compute_p50_ms": round(compute_p50_ms, 2),
            "compute_p99_ms": round(compute_p99_ms, 2),
            "loop_overruns": overruns,
        },
    }
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, required=True, help="Path to .npz run log")
    p.add_argument("--json", action="store_true", help="Print as raw JSON")
    args = p.parse_args()

    res = analyze_log(args.log)
    if args.json:
        print(json.dumps(res, indent=2))
        return

    print("=" * 60)
    print(f"PROMO Run Evaluation: {args.log.name}")
    print("=" * 60)
    print(f"Duration: {res['duration_s']} s ({res['walk_steps']} WALK steps)")
    if res["preference_w"]:
        w = res["preference_w"]
        print(f"Preference w (track, stab, eff): [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}]")
    print("\n[Efficiency & Effort]")
    print(f"  Mean Power:       {res['efficiency']['mean_power_w']} W (Peak: {res['efficiency']['peak_power_w']} W)")
    print(f"  Total Energy:     {res['efficiency']['total_energy_j']} J")
    print(f"  Est Distance:     {res['efficiency']['est_distance_m']} m")
    print(f"  Specific Energy:  {res['efficiency']['specific_energy_j_per_m']} J/m")
    print(f"  CoT:              {res['efficiency']['cot']}")
    print(f"  Action Rate RMS:  {res['efficiency']['act_rate_rms_rad_per_s']} rad/s")
    print(f"  Torque Rate RMS:  {res['efficiency']['trq_rate_rms_nm_per_s']} N*m/s")
    print("\n[Stability / Body Attitude]")
    print(f"  Roll RMS:         {res['stability']['roll_rms_rad']} rad ({math.degrees(res['stability']['roll_rms_rad']):.2f} deg)")
    print(f"  Pitch RMS:        {res['stability']['pitch_rms_rad']} rad ({math.degrees(res['stability']['pitch_rms_rad']):.2f} deg)")
    print(f"  Peak Attitude:    {res['stability']['peak_att_rad']} rad ({math.degrees(res['stability']['peak_att_rad']):.2f} deg)")
    print("\n[Timing & Latency]")
    print(f"  Inference p50:    {res['performance']['compute_p50_ms']} ms")
    print(f"  Inference p99:    {res['performance']['compute_p99_ms']} ms")
    print(f"  Overruns:         {res['performance']['loop_overruns']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
