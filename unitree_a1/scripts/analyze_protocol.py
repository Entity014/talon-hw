"""Metrics and tables from a scripted evaluation (scripts/eval_protocol.py): one row per (preference, regime), mean +- sd over trials. (2026-11-25), runs on any machine with numpy; per-run log metrics only, scripts/analyze_promo_run.py is the older single-run report.
    python3 scripts/analyze_protocol.py --dir logs/protocol1 [--mass-kg 12.45]  ->  prints the table and writes summary.csv and per_trial.csv in --dir

Definitions (the PROMO paper does not state its formulas; these are ours and are printed with the table):
  window     = the WALK rows after the first --skip-s seconds (default 2.0 = the soft start) up to the end of the walk.
  power      = mean over the window of sum_j max(tau_est_j * dq_j, 0)  [W]   (positive mechanical power; the sum of |tau * dq| is also given as power_abs)
  energy J/m = power * walk time / distance walked; distance = the number typed by the operator (m) for that trial; blank -> NaN (never estimated from v_hat)
  CoT        = energy per metre / (mass * 9.81)
  speed      = distance / walk time (m/s) (the whole walk, including the pauses of the profile)
  attitude   = max over the window of sqrt(dr^2 + dp^2) in degrees, with dr, dp the IMU roll / pitch minus their mean in the last 1.5 s of STAND (the IMU has a constant offset; the loads show the robot stands level)
  gyro_rp    = std of the roll and pitch gyro rate over the window, mean of the two [rad/s]
  action_rate = RMS over the window of d(action)/dt [1/s]; torque_rate = RMS of d(tau_est)/dt over joints and time [Nm/s]
  fault      = the trial ended in FAULT (latched safety fault) or ABORT (operator/timeout)
Tracking error needs ground truth and is NOT computed here (add mocap).
"""
import argparse
import csv
import json
import os

import numpy as np


def window_rows(d, skip_s):
    mode = np.array([str(x) for x in d["mode"]])
    walk = np.where(mode == "WALK")[0]
    if len(walk) < 10:
        return None, None
    t = d["t"]
    start = walk[0]
    idx = walk[t[walk] - t[start] >= skip_s]
    return idx, walk


def log_metrics(d, distance_m=None, mass_kg=12.0, skip_s=2.0):
    idx, walk = window_rows(d, skip_s)
    out = dict.fromkeys(("walk_time_s", "power_w", "power_abs_w", "energy_j_per_m", "cot", "speed_mps", "attitude_deg", "gyro_rp", "action_rate", "torque_rate"), float("nan"))
    if idx is None or len(idx) < 25:
        return out
    t = d["t"]; dt = np.diff(t[idx]); dt = dt[(dt > 0) & (dt < 0.2)]
    step = float(np.median(dt)) if len(dt) else 0.02
    mode = np.array([str(x) for x in d["mode"]])
    stand = np.where(mode == "STAND")[0]
    base = d["rpy"][stand][-75:, :2].mean(0) if len(stand) >= 20 else np.zeros(2)
    tau, dq = d["tau_est"][idx], d["dq"][idx]
    p_pos = np.maximum(tau * dq, 0.0).sum(1); p_abs = np.abs(tau * dq).sum(1)
    out["walk_time_s"] = float(t[walk[-1]] - t[walk[0]])
    out["power_w"], out["power_abs_w"] = float(np.nanmean(p_pos)), float(np.nanmean(p_abs))
    drp = d["rpy"][idx][:, :2] - base
    out["attitude_deg"] = float(np.degrees(np.nanmax(np.sqrt((drp ** 2).sum(1)))))
    out["gyro_rp"] = float(np.nanmean(np.nanstd(d["gyro"][idx][:, :2], axis=0)))
    a = d["action"][idx]; ok = ~np.isnan(a).any(1)
    out["action_rate"] = float(np.sqrt(np.nanmean((np.diff(a[ok], axis=0) / step) ** 2)))
    out["torque_rate"] = float(np.sqrt(np.nanmean((np.diff(tau, axis=0) / step) ** 2)))
    if distance_m and distance_m > 0:
        out["energy_j_per_m"] = out["power_w"] * out["walk_time_s"] / distance_m
        out["cot"] = out["energy_j_per_m"] / (mass_kg * 9.81)
        out["speed_mps"] = distance_m / out["walk_time_s"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True)
    p.add_argument("--mass-kg", type=float, default=12.0)
    p.add_argument("--skip-s", type=float, default=2.0)
    a = p.parse_args()
    man = json.load(open(os.path.join(a.dir, "manifest.json")))
    rows = []
    for e in man:
        d = np.load(os.path.join(a.dir, e["log"]), allow_pickle=True)
        m = log_metrics(d, e.get("distance_m"), a.mass_kg, a.skip_s)
        rows.append(dict(index=e["index"], preference=e["preference"], regime=e["regime"], trial=e["trial"], outcome=e["outcome"], faults=",".join(e["faults"]), distance_m=e.get("distance_m", float("nan")), **m))
    keys = ["index", "preference", "regime", "trial", "outcome", "faults", "distance_m", "walk_time_s", "power_w", "power_abs_w", "energy_j_per_m", "cot", "speed_mps", "attitude_deg", "gyro_rp", "action_rate", "torque_rate"]
    with open(os.path.join(a.dir, "per_trial.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in rows]
    cells = sorted({(r["preference"], r["regime"]) for r in rows})
    cols = ["power_w", "energy_j_per_m", "cot", "speed_mps", "attitude_deg", "gyro_rp", "action_rate", "torque_rate"]
    summ = []
    print("%-11s %-9s %2s %s %s" % ("preference", "regime", "n", "ok/fault/abort", " ".join("%16s" % c for c in cols)))
    for pref, reg in cells:
        rr = [r for r in rows if r["preference"] == pref and r["regime"] == reg]
        n_ok = sum(r["outcome"] == "OK" for r in rr); n_f = sum(r["outcome"] == "FAULT" for r in rr); n_a = sum(r["outcome"] == "ABORT" for r in rr)
        cells_txt, rec = [], dict(preference=pref, regime=reg, n=len(rr), ok=n_ok, fault=n_f, abort=n_a)
        for c in cols:
            v = np.array([r[c] for r in rr if r["outcome"] == "OK"], dtype=float); v = v[~np.isnan(v)]
            rec[c + "_mean"], rec[c + "_sd"] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")) if len(v) else (float("nan"), float("nan"))
            cells_txt.append("%7.3g +-%6.2g" % (rec[c + "_mean"], rec[c + "_sd"]) if len(v) else "%16s" % "-")
        summ.append(rec)
        print("%-11s %-9s %2d %2d/%2d/%2d        %s" % (pref, reg, len(rr), n_ok, n_f, n_a, " ".join(cells_txt)))
    with open(os.path.join(a.dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summ[0].keys())); w.writeheader(); [w.writerow(r) for r in summ]
    print("written: per_trial.csv, summary.csv (only OK trials enter the mean +- sd; FAULT / ABORT are counted)")


if __name__ == "__main__":
    main()
