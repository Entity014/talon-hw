# A1 hardware test checklist: what to record (prepared 2026-11-27)

Applies to the next robot session, for `oursA` (baseline) and `oursA_V2` (when training finishes). Goal: results that can be compared with `logs/interactive1` and `logs/interactive2` of 2026-11-26 and that decide the next step.

## 0. Before each session (once)

| Item | How | Why |
|---|---|---|
| Policy identity | `sha256sum policies/<name>/policy.npz`, and `default_q` from `policy_meta.json` | V2 must show thigh = 0.9 on all legs; oursA is 0.8 front / 1.0 rear |
| Code state | `git rev-parse HEAD` on talon-hw, plus the exact `run.py` / `eval_protocol.py` command line | results must be reproducible |
| IMU zero | `--imu-zero`, wait 4-5 s in STAND; write down roll/pitch bias printed | bias was +0.06 / +0.018 rad |
| Static load split | two scales, one under the front pair, one under the rear pair, robot in STAND; record kg for each | foot_force sums to 300-420 against ~118 N, so it is not calibrated; this settles the front-heavy claim and CoM offset |
| Battery voltage, floor, room temperature, motor temperatures | note at start and end | drift between trials |

## 1. Test matrix (same as interactive1/2 so numbers compare)

Flags: `--flip-imu-x --imu-zero --action-smooth 0.0 --dq-noise 0.0`, no bias flags (pure zero-shot).

- Preferences: balanced, tracking, stability, efficiency
- Regimes: replay (6 directions), move_fast (0.5 m/s), move_slow (0.25 m/s)
- 5 trials per cell if time allows; otherwise 3. Keep the seeded shuffle of `eval_protocol.py`.
- Run oursA and V2 in alternating order within the session, not all of one then all of the other.

## 2. Per trial (write on paper or in the prompt answer)

1. Outcome: OK / FAULT (name the fault, e.g. overtorque, stale_target, tilt) / ABORT, and the second at which it ended
2. Distance along the lane in metres (tape measure). Start and end marks on the floor
3. Video from the side (phone is enough): gives an independent speed estimate and shows foot clearance
4. Observed behaviour in one line: forward / back / left / right / yaw each OK or not; front feet drag; bounce; chatter or whole-body shaking; leg that lifts less than the others
5. Anything done by hand (stick input, restart, flag change)

## 3. Computed afterwards (`analyze_protocol.py`, same definitions as 2026-11-26)

power_w, energy_j_per_m, cot, speed_mps, attitude_deg, gyro_rp, action_rate, torque_rate, plus the count of OK / FAULT / ABORT per cell.
Caveats to print next to every table: `tau_est` is the firmware's PD-law value, not a measured torque, so power / energy / CoT are "commanded-torque" values (compare preferences with each other, do not compare with PROMO's numbers); distance is tape-measured; there is no mocap. `v_hat` is now logged (the policy's own velocity estimate, not ground truth); `wz_rmse` (gyro z vs command) is a real measurement.

Also computed now (no manual work): vel_rmse_est and vx_est_mean (from v_hat), wz_rmse, q_track_rmse, tau_peak, tau_sat_frac, roll_pitch_rms_deg, step_hz (hardware earlier 6.5-7.8 Hz, sim 4.7 Hz), front_load_frac, foot_air_frac_max (foot_force units uncalibrated). Compare vx_est_mean with the tape speed to see how far v_hat can be trusted.

## 4. Directional pass check (the thing the 11-26 logs could not say)

For each preference and each of the 6 directions in replay: commanded speed vs distance moved (tape or video), pass if the robot moves within about 30 % of the command without a fault. Record as a 4 x 6 table of pass / fail. This is the main number for "does V2 close the gap".

## 5. System identification data (for the sim, not the policy)

Robot on the box, legs unloaded, deploy gains Kp 55 / Kd 0.75:
- `latency_probe.py --box --flip-imu-x` (10 steps) and `motor_probe.py --box --flip-imu-x`
- add one step and one slow chirp per joint type (hip, thigh, calf) if the scripts allow it
- save the `probe_*.npz` files and sync them to this machine (`sync_from_robot.sh`)

Fit target: armature, Coulomb friction, viscous damping, delay per joint type, from q(t) and dq(t). Do not use `tau_est` for this.

## 6. Decision after the session

- All 4 preferences walk all 6 directions: report as one bundle (DR + symmetric pose + observation change); stop.
- Sim (HW-SI) passes but robot still bounces or chatters: retrain with one change only (armature, or the action EMA inside training with the filtered action fed back as `last_action`).
- Efficiency still stands still on the robot while V2 passes HW-SI: the DR box misses a term; use section 5 to find it.
- V2 fails in ideal sim as well: training under-converged (ep_len stuck near 250-400 in the first 1000 iterations); that is a training problem, not a sim-to-real result.

## 7. Files to keep

`logs/<session>/` with manifest.json, per-trial npz, summary.csv, per_trial.csv, the videos, the scale readings, and a short note of the commands used. Add a daily note to this folder after the session.

## 8. Replay trials: how to measure (added after the first draft)

The replay profile is scripted: fwd 0.3 m/s 4 s, back 0.3 m/s 4 s, left 0.2 m/s 4 s, right 0.2 m/s 4 s, yaw +0.4 rad/s 4 s, yaw -0.4 rad/s 4 s, with 2 s stops between. The path is closed on purpose (each move is undone by its opposite at the same speed and time), so the robot should end where it started. One total distance is meaningless; the end pose error is the measure. Measure:

1. Mark the start position and heading on the floor with tape. At the end of each replay trial measure the end pose against it: dx, dy (cm) and heading change (deg). Expected from the command: dx 0, dy 0, heading 0. Any non-zero value is the robot's error (drift, forward/back asymmetry), so smaller is better. Write the three numbers per trial. Replay logs from before 2026-11-27 used fwd 0.4 m/s 6 s and are not comparable.
2. Optional: video from above or at an angle with a taped floor grid, for the distance of each segment.
3. Automatic (`analyze_protocol.py` writes `per_segment.csv` and prints pass counts): per segment the mean `v_hat` (estimate), the mean gyro z and the yaw angle (integral of gyro z over the constant-command part; the 0.5 s ramp is excluded, so 0.4 rad/s x 3.5 s = 80 deg expected). Pass = within 30 % of the command. Yaw segments use the gyro and are measured; translation segments use `v_hat` and are only an estimate, so confirm them with item 1 or 2.
4. Put the table of pass counts per preference (fwd / back / left / right / yawL / yawR) into the session note; this is the directional pass table of section 4.
