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
  vel_rmse_est = RMS over the window of |command_xy - v_hat_xy| [m/s]; v_hat is the policy's OWN velocity estimate (vel_est head), not ground truth; vx_est_mean = mean v_hat_x. Compare vx_est_mean with speed_mps (tape) to see how far the estimate can be trusted
  wz_rmse = RMS of command_wz - measured gyro z [rad/s] (a real measurement); q_track_rmse = RMS(q_target - q) [rad]; tau_peak [Nm], tau_sat_frac = share of joint-ticks above 80 % of 33.5 Nm (tau_est is the PD law, not a torque measurement)
  step_hz = dominant foot_force frequency 1-12 Hz; front_load_frac = front / total foot_force; foot_air_frac_max = worst foot's share of ticks with foot_force <= 5 (foot_force units are uncalibrated)
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
    out = dict.fromkeys(("walk_time_s", "power_w", "power_abs_w", "energy_j_per_m", "cot", "speed_mps", "attitude_deg", "gyro_rp", "action_rate", "torque_rate", "vel_rmse_est", "vx_est_mean", "wz_rmse", "q_track_rmse", "tau_peak", "tau_sat_frac", "roll_pitch_rms_deg", "step_hz", "front_load_frac", "foot_air_frac_max"), float("nan"))
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
    cmd = d["command"][idx].astype(float)
    gz = d["gyro"][idx][:, 2].astype(float)  # measured body yaw rate (the driver already applies --flip-imu-x)
    ok = np.isfinite(cmd[:, 2]) & np.isfinite(gz)
    if ok.sum() > 25:
        out["wz_rmse"] = float(np.sqrt(np.mean((cmd[ok, 2] - gz[ok]) ** 2)))
    qe = (d["q_target"][idx] - d["q"][idx]).astype(float)  # NOTE q_target is in POLICY joint order, q in SDK order
    try:
        from talon_a1 import contract
        qe = (contract.policy_to_sdk(d["q_target"][idx].T).T - d["q"][idx]).astype(float)
        out["q_track_rmse"] = float(np.sqrt(np.nanmean(qe ** 2)))
    except Exception:
        pass
    out["tau_peak"] = float(np.nanmax(np.abs(tau)))
    out["tau_sat_frac"] = float(np.nanmean(np.abs(tau) > 0.8 * 33.5))  # share of joint-ticks above 80 % of the A1 33.5 N*m limit
    out["roll_pitch_rms_deg"] = float(np.degrees(np.sqrt(np.nanmean((drp ** 2).sum(1)))))
    ff = d["foot_force"][idx].astype(float)
    if np.isfinite(ff).all() and len(ff) > 50:
        x = ff - ff.mean(0)
        spec = np.abs(np.fft.rfft(x, axis=0)).mean(1)
        fr = np.fft.rfftfreq(len(ff), step)
        band = (fr > 1.0) & (fr < 12.0)
        out["step_hz"] = float(fr[band][np.argmax(spec[band])]) if band.any() else float("nan")
        tot = ff.sum(1)
        out["front_load_frac"] = float(np.nanmean(ff[:, :2].sum(1)[tot > 0] / tot[tot > 0]))  # foot order FR, FL, RR, RL
        out["foot_air_frac_max"] = float(np.max(np.mean(ff <= 5, axis=0)))  # worst foot: share of ticks with foot_force <= 5 (raw units)
    vh = d["v_hat"][idx][:, :2].astype(float) if "v_hat" in d.files else None
    if vh is not None and np.isfinite(vh).all(1).sum() > 25:  # the policy's own velocity estimate vs the command (body frame); NOT ground truth
        ok = np.isfinite(vh).all(1)
        out["vel_rmse_est"] = float(np.sqrt(np.mean(((d["command"][idx][:, :2][ok] - vh[ok]) ** 2).sum(1))))
        out["vx_est_mean"] = float(np.mean(vh[ok, 0]))
    if distance_m and distance_m > 0:
        out["energy_j_per_m"] = out["power_w"] * out["walk_time_s"] / distance_m
        out["cot"] = out["energy_j_per_m"] / (mass_kg * 9.81)
        out["speed_mps"] = distance_m / out["walk_time_s"]
    return out


LABELS = {(0, 1): "fwd", (0, -1): "back", (1, 1): "left", (1, -1): "right", (2, 1): "yawL", (2, -1): "yawR"}


def segment_metrics(d, skip_s=1.0, min_run_s=1.5, tol=0.3):
    """Per constant-command segment of a replay trial (found from the logged command, no schedule needed).
    Returns a list of dicts: label, cmd, dur_s, v_hat_mean (policy estimate, NOT ground truth), gyro_z_mean and yaw_deg (integral of measured gyro z over the constant-command run only, so the 0.5 s ramp is missing; a real measurement), passed.
    pass = the measured quantity is within `tol` of the command: yaw segments use the gyro (real); translation segments use v_hat (an estimate), so check them against the tape / video."""
    mode = np.array([str(x) for x in d["mode"]]); walk = np.where(mode == "WALK")[0]
    if len(walk) < 50:
        return []
    t = d["t"][walk].astype(float); c = d["command"][walk].astype(float); gz = d["gyro"][walk][:, 2].astype(float)
    vh = d["v_hat"][walk][:, :2].astype(float) if "v_hat" in d.files else np.full((len(walk), 2), np.nan)
    out, i, n = [], 0, len(walk)
    while i < n:
        j = i
        while j + 1 < n and np.abs(c[j + 1] - c[i]).max() < 1e-3:
            j += 1
        dur = t[j] - t[i]
        if dur >= min_run_s and np.abs(c[i]).max() > 0.05:
            k = int(np.argmax(np.abs(c[i]))); label = LABELS[(k, 1 if c[i][k] > 0 else -1)]
            m = slice(i + int(skip_s / 0.02), j + 1)  # drop the ramp and the settle
            gm = float(np.nanmean(gz[m])) if j + 1 > m.start else float("nan")
            vm = np.nanmean(vh[m], axis=0) if j + 1 > m.start else np.full(2, np.nan)
            meas = gm if k == 2 else float(vm[k])
            out.append(dict(label=label, cmd=float(c[i][k]), dur_s=float(dur), v_hat_x=float(vm[0]), v_hat_y=float(vm[1]), gyro_z_mean=gm,
                            yaw_deg=float(np.degrees(np.trapz(gz[i:j + 1], t[i:j + 1]))), measured=float(meas),
                            passed=bool(np.isfinite(meas) and abs(meas - c[i][k]) <= tol * abs(c[i][k]) and np.sign(meas) == np.sign(c[i][k]))))
        i = j + 1
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
    seg_rows = []
    for e in man:
        if e["regime"] == "replay":
            d = np.load(os.path.join(a.dir, e["log"]), allow_pickle=True)
            for sg in segment_metrics(d):
                seg_rows.append(dict(index=e["index"], preference=e["preference"], trial=e["trial"], outcome=e["outcome"], **sg))
    if seg_rows:
        with open(os.path.join(a.dir, "per_segment.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, list(seg_rows[0].keys())); w.writeheader(); [w.writerow(r) for r in seg_rows]
        print("replay segments (pass = within 30 %% of the command; yaw from gyro = measured, others from v_hat = estimate):")
        for pref in sorted({r["preference"] for r in seg_rows}):
            line = []
            for lab in ("fwd", "back", "left", "right", "yawL", "yawR"):
                rr = [r for r in seg_rows if r["preference"] == pref and r["label"] == lab]
                line.append("%s %d/%d" % (lab, sum(r["passed"] for r in rr), len(rr)))
            print("  %-11s %s" % (pref, "  ".join(line)))
    keys = ["index", "preference", "regime", "trial", "outcome", "faults", "distance_m", "walk_time_s", "power_w", "power_abs_w", "energy_j_per_m", "cot", "speed_mps", "attitude_deg", "gyro_rp", "action_rate", "torque_rate", "vel_rmse_est", "vx_est_mean", "wz_rmse", "q_track_rmse", "tau_peak", "tau_sat_frac", "roll_pitch_rms_deg", "step_hz", "front_load_frac", "foot_air_frac_max"]
    with open(os.path.join(a.dir, "per_trial.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in rows]
    cells = sorted({(r["preference"], r["regime"]) for r in rows})
    cols = ["power_w", "energy_j_per_m", "cot", "speed_mps", "attitude_deg", "gyro_rp", "action_rate", "torque_rate", "vel_rmse_est", "vx_est_mean", "wz_rmse", "q_track_rmse", "tau_peak", "tau_sat_frac", "roll_pitch_rms_deg", "step_hz", "front_load_frac", "foot_air_frac_max"]
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
