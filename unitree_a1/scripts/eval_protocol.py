"""Scripted real-robot evaluation (PROMO-style: preferences x command regimes x trials) for the A1, run by one command so the operator only places the robot and answers two prompts.
Written 2026-11-25 against a fake driver (tests/test_eval_protocol.py); NOT yet run on the robot: try it first on the gantry or the loopback build (it needs the compiled _a1_driver for a real run).

    python3 scripts/eval_protocol.py --policy policies/oursA/policy.npz --flip-imu-x --out-dir logs/protocol1 --dry-run      # print the schedule and the time
    sudo python3 scripts/eval_protocol.py --policy policies/oursA/policy.npz --flip-imu-x --out-dir logs/protocol1 --trials 5   # run

What it does per trial (all timing from the software, no stick input, the remote only keeps the safety keys: L1+B crouch, L2+B e-stop): L1+A (stand up) -> wait for the ramp -> L1+X (start the policy) -> the command profile of the regime
-> L1+Y (back to stand) -> L1+B (crouch to passive). The run order is a seeded shuffle of all (preference, regime, trial) cells so drift (battery, floor, temperature) is not confounded with a condition. Every trial writes one run log
(npz, same format as run.py plus the condition in `meta`) and a line in manifest.json: condition, outcome (OK / FAULT / ABORT), faults, the walked distance you typed (m) and the run time.
Ground truth: this script has NO motion capture. After each trial it asks for the distance along the lane (tape measure) so speed and energy per metre have a measured denominator; Enter = unknown. If you have mocap, store its file next to the log and
give the path in the prompt answer starting with `mocap:` (it is only copied into the manifest).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from talon_a1.controller import PASSIVE, STAND, WALK, Controller, RunLog, START_WALK, provenance, STAND_UP, STOP_WALK, TO_PASSIVE, fault_names, obs_noise_amplitude  # noqa: E402
from talon_a1.remote import Remote  # noqa: E402

PREFERENCES = {  # the four preferences of the PROMO hardware study (w = track, stab, eff)
    "balanced": (0.333, 0.333, 0.334),
    "tracking": (0.8, 0.1, 0.1),
    "stability": (0.1, 0.8, 0.1),
    "efficiency": (0.1, 0.1, 0.8),
}

# replay is a closed path: every move is followed by its opposite at the same speed and duration (fwd/back 0.3 m/s 4 s, left/right 0.2 m/s 4 s, yaw +/-0.4 rad/s 4 s), so the commanded net displacement and heading change are zero and the end pose error measures the robot, not the profile. (Before 2026-11-27 forward was 0.4 m/s 6 s: net +1.2 m, not comparable.)
# command regimes: list of (duration s, (vx, vy, wz)); the command moves linearly between segments in RAMP seconds. The paper does not state its speeds or durations: these are ours.
REGIMES = {
    "replay": [(2.0, (0, 0, 0)), (4.0, (0.3, 0, 0)), (2.0, (0, 0, 0)), (4.0, (-0.3, 0, 0)), (2.0, (0, 0, 0)), (4.0, (0, 0.2, 0)), (2.0, (0, 0, 0)), (4.0, (0, -0.2, 0)), (2.0, (0, 0, 0)), (4.0, (0, 0, 0.4)), (2.0, (0, 0, 0)), (4.0, (0, 0, -0.4)), (2.0, (0, 0, 0))],
    "move_fast": [(2.0, (0, 0, 0)), (10.0, (0.5, 0, 0)), (2.0, (0, 0, 0))],
    "move_slow": [(2.0, (0, 0, 0)), (14.0, (0.25, 0, 0)), (2.0, (0, 0, 0))],
}
RAMP = 0.5
POLICY_DT = 0.02


def profile_duration(segments):
    return float(sum(d for d, _ in segments))


def profile_command(t, segments, ramp=RAMP):
    """Piecewise-constant command with a linear ramp of `ramp` s at the start of each segment (from the previous value); zero after the end."""
    prev = np.zeros(3)
    t0 = 0.0
    for dur, c in segments:
        c = np.asarray(c, dtype=np.float64)
        if t < t0 + dur:
            r = min(1.0, (t - t0) / ramp) if ramp > 0 else 1.0
            return ((1.0 - r) * prev + r * c).astype(np.float32)
        prev, t0 = c, t0 + dur
    return np.zeros(3, dtype=np.float32)


def make_schedule(preferences, regimes, trials, seed):
    cells = [(p, r, k) for p in preferences for r in regimes for k in range(trials)]
    random.Random(seed).shuffle(cells)
    return cells


class ScriptedRemote(Remote):
    """Remote whose key presses and command come from the runner; the real keys are still OR-ed in (L1+B, L2+B and SELECT keep working for the operator)."""

    def __init__(self, vx_max, vy_max, wz_max):
        Remote.__init__(self, vx_max, vy_max, wz_max)
        self.inject = 0
        self.cmd = np.zeros(3, dtype=np.float32)

    def update(self, state):
        self.prev_keys, self.keys = self.keys, int(state.keys) | self.inject

    def command(self, state):
        return np.clip(self.cmd, -self.max, self.max).astype(np.float32)


class Trial(object):
    """One scripted trial against a driver; step(state) is called once per 20 ms tick by the loop and returns True when the trial is over."""

    def __init__(self, ctl, segments, stand_timeout=15.0, phase_timeout=10.0):
        self.ctl, self.seg = ctl, segments
        self.phase = "up"
        self.t = 0.0
        self.phase_t = 0.0
        self.walk_t = None
        self.stand_timeout, self.phase_timeout = stand_timeout, phase_timeout
        self.outcome = None
        self.walk_start_t = None
        self.walk_end_t = None
        self.finishing = False
        self.pressed_down = False

    def _go(self, phase):
        self.phase, self.phase_t = phase, 0.0

    def keys_and_command(self, state):
        """The keys to inject on this tick and the command; called BEFORE ctl.step."""
        ctl = self.ctl
        inj, cmd = 0, np.zeros(3, np.float32)
        if state.faults and self.outcome is None:
            self.outcome = "FAULT"
        if self.finishing:  # an abort: bring the robot down the same way as a normal end (L1+B = crouch), never leave it standing or half way
            if ctl.mode in (STAND, WALK) and not self.pressed_down:
                inj = TO_PASSIVE
                self.pressed_down = True
            return inj, cmd
        if self.phase == "up":
            if ctl.mode == PASSIVE and self.phase_t < 0.05:
                inj = STAND_UP
            elif ctl.mode == STAND and ctl.ramp_t >= ctl.stand_time - 1e-6:
                inj = START_WALK
                self._go("start")
        elif self.phase == "start":
            if ctl.mode == WALK:
                self.walk_start_t = self.t
                self._go("walk")
        elif self.phase == "walk":
            wt = self.phase_t
            cmd = profile_command(wt, self.seg)
            if wt >= profile_duration(self.seg):
                inj = STOP_WALK
                self.walk_end_t = self.t
                self._go("stop")
        elif self.phase == "stop":
            if ctl.mode == STAND and ctl.ramp_t >= 1.0 - 1e-6:
                inj = TO_PASSIVE
                self._go("down")
        elif self.phase == "down":
            pass
        return inj, cmd

    def _abort(self):
        self.outcome = "ABORT"
        self.finishing = True
        self.phase_t = 0.0

    def advance(self, state):
        """After ctl.step: advance the clocks, return True when finished."""
        self.t += POLICY_DT
        self.phase_t += POLICY_DT
        ctl = self.ctl
        if self.outcome == "FAULT":
            return True  # the controller already went passive and disarmed
        if self.finishing:
            return ctl.mode == PASSIVE or self.phase_t > 5.0
        if self.phase == "down" and ctl.mode == PASSIVE:
            self.outcome = "OK"
            return True
        limit = self.stand_timeout if self.phase == "up" else (profile_duration(self.seg) + 1.0 if self.phase == "walk" else self.phase_timeout)
        if self.phase_t > limit + 0.5:
            self._abort()
        elif self.phase in ("stop", "start") and ctl.mode == PASSIVE:
            self._abort()
        elif self.phase == "walk" and ctl.mode != WALK:  # the operator pressed L1+B (or something else left WALK)
            self._abort()
        return False


def run_trial(drv, policy, segments, args, w, log_path, meta, sleep=True, fault_table=None):
    """Run one trial on `drv` (real or fake); returns the manifest entry. The log is saved to log_path."""
    from talon_a1 import contract

    remote = ScriptedRemote(args.vx_max, args.vy_max, args.wz_max)
    log = RunLog(obs_dim=getattr(policy, "obs_dim", contract.OBS_DIM))
    ctl = Controller(drv, policy, remote, dict(fault_table or {}), stand_time=args.stand_time, action_clip=args.action_clip, log=log,
                     policy_kp=args.policy_kp, allow_walk=True, soft_start=args.soft_start, action_scale=args.action_scale, w=w,
                     obs_noise=obs_noise_amplitude(getattr(args, "obs_noise_scale", 0.0), getattr(args, "dq_noise", None)) if hasattr(policy, "hist_len") else None, noise_seed=getattr(args, "noise_seed", 0), lead_ms=getattr(args, "lead_ms", 0.0) if hasattr(policy, "hist_len") else 0.0,
                     action_smooth=getattr(args, "action_smooth", 0.0), imu_zero=getattr(args, "imu_zero", False))
    tr = Trial(ctl, segments)
    t0 = time.monotonic(); next_t = t0
    while True:
        state = drv.state()
        start = time.monotonic()
        inj, cmd = tr.keys_and_command(state)
        remote.inject, remote.cmd = inj, cmd
        rec = ctl.step(state)
        rec["compute_s"] = time.monotonic() - start
        rec["t"] = start - t0
        log.append(state, rec)
        remote.inject = 0
        if tr.advance(state):
            break
        next_t += POLICY_DT
        if sleep:
            dt = next_t - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                ctl.overruns += 1
                next_t = time.monotonic()
    final = drv.state()
    faults = fault_names(int(final.faults), fault_table or {}) if final.faults else []
    entry = dict(meta, outcome=tr.outcome, faults=faults, log=os.path.basename(log_path), run_time_s=tr.t,
                 walk_time_s=None if tr.walk_start_t is None else (tr.walk_end_t if tr.walk_end_t is not None else tr.t) - tr.walk_start_t,
                 policy_overruns=ctl.overruns)
    log.save(log_path, dict(meta, **provenance(getattr(args, "policy", None), policy), outcome=tr.outcome, faults=faults, args={k: v for k, v in vars(args).items() if not k.startswith("_")}))
    return entry


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--preferences", nargs="+", default=list(PREFERENCES), choices=list(PREFERENCES))
    p.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-prompt", action="store_true", help="no operator prompts between trials (loopback / desk tests only)")
    p.add_argument("--action-clip", type=float, default=10.0)
    p.add_argument("--soft-start", type=float, default=2.0)
    p.add_argument("--stand-time", type=float, default=3.0)
    p.add_argument("--vx-max", type=float, default=0.5)
    p.add_argument("--vy-max", type=float, default=0.3)
    p.add_argument("--wz-max", type=float, default=0.5)
    p.add_argument("--tau-limit", type=float, default=30.0)
    p.add_argument("--tilt-limit", type=float, default=0.5)
    p.add_argument("--power-limit", type=float, default=300.0)
    p.add_argument("--damping-kd", type=float, default=1.0)
    p.add_argument("--flip-imu-x", action="store_true")
    p.add_argument("--imu-zero", action="store_true", help="auto-calibrate IMU roll/pitch bias in STAND before WALK")
    p.add_argument("--interactive", action="store_true", help="shortcut menu: a=replay, b=move_fast, c=move_slow, w=preference, q=quit")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--obs-noise-scale", type=float, default=0.0, help="the dq A/B test: training observation noise x this scale added to the observation (see run.py)")
    p.add_argument("--dq-noise", type=float, default=0.1, help="uniform +- noise on joint_vel (rad/s) added to the observation (see run.py); default 0.1 from hardware tuning")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--action-smooth", type=float, default=0.0, help="EMA action smoothing factor in [0, 1) to damp motor chatter (see run.py)")
    p.add_argument("--lead-ms", type=float, default=0.0, help="lead compensation of the observation in ms (see run.py); 0 = off")
    p.add_argument("--rest-s", type=float, default=0.0, help="wait this many seconds between trials (motors cool down)")
    args = p.parse_args()
    cells = make_schedule(args.preferences, args.regimes, args.trials, args.seed)
    total = sum(profile_duration(REGIMES[r]) + 3 + 1 + 1.5 + 3 for _, r, _ in cells)
    print("%d trials (%s x %s x %d), seed %d; about %.0f min of robot time without the operator's pauses" % (len(cells), ",".join(args.preferences), ",".join(args.regimes), args.trials, args.seed, total / 60))
    for i, (pref, reg, k) in enumerate(cells):
        print("  %2d  %-10s %-9s trial %d  w=%s" % (i + 1, pref, reg, k + 1, PREFERENCES[pref]))
    if args.dry_run:
        return

    from talon_a1 import _a1_driver, contract, sport
    from talon_a1.policy import load_policy

    first_w = PREFERENCES[cells[0][0]]
    policy = load_policy(args.policy, w=first_w)
    args.action_scale = getattr(policy, "action_scale", contract.ACTION_SCALE)
    args.policy_kp = getattr(policy, "kp", contract.KP)
    cfg = _a1_driver.SafetyConfig()
    cfg.tilt_limit_rad, cfg.tau_limit, cfg.power_limit_w, cfg.flip_imu_x, cfg.damping_kd = args.tilt_limit, args.tau_limit, args.power_limit, args.flip_imu_x, args.damping_kd
    if _a1_driver.WITH_SDK:
        sport.require_stopped()
        print("LOW-LEVEL control of the real A1. Remote in hand: L1+B crouch, L2+B e-stop. Check the lane and the tether.")
        if not args.yes:
            input("Press Enter to start the motor loop...")
    drv = _a1_driver.Driver(cfg, 0.002)
    drv.start()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    manifest = []
    try:
        if args.interactive:
            shortcuts = {"a": "replay", "b": "move_fast", "c": "move_slow"}
            pref_names = list(PREFERENCES)
            pref_idx = pref_names.index(args.preferences[0]) if args.preferences else 0
            trial_num = 0
            while True:
                pref = pref_names[pref_idx]
                ans = input("\n[pref=%s] a=replay, b=move_fast, c=move_slow, w=switch pref, q=quit: " % pref).strip().lower()
                if ans == "q":
                    break
                if ans == "w":
                    pref_idx = (pref_idx + 1) % len(pref_names)
                    continue
                if ans not in shortcuts:
                    print("  unknown key (use a, b, c, w, or q)")
                    continue
                reg = shortcuts[ans]
                trial_num += 1
                w = np.array(PREFERENCES[pref], dtype=np.float32)
                name = "%02d_%s_%s" % (trial_num, pref, reg)
                meta = {"index": trial_num, "preference": pref, "w": [float(x) for x in w], "regime": reg, "trial": trial_num, "seed": args.seed, "policy": os.path.basename(args.policy)}
                if hasattr(policy, "set_preference"):
                    policy.set_preference(w)
                entry = run_trial(drv, policy, REGIMES[reg], args, w, os.path.join(args.out_dir, name + ".npz"), meta, fault_table=dict(_a1_driver.FAULTS))
                print("   -> %s  faults %s  walk %.1f s" % (entry["outcome"], entry["faults"], entry["walk_time_s"] or 0.0))
                manifest.append(entry)
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=1)
                if entry["outcome"] == "FAULT":
                    print("   FAULT: clear it with SELECT on the remote before the next run")
                    input("   press Enter when the fault is cleared and the robot is safe... ")
                    drv.clear_faults()
            return

        for i, (pref, reg, k) in enumerate(cells):
            w = np.array(PREFERENCES[pref], dtype=np.float32)
            name = "%02d_%s_%s_t%d" % (i + 1, pref, reg, k + 1)
            if not args.no_prompt:
                ans = input("[%d/%d] %s: put the robot on the start mark, lane clear. Enter = run, s = skip, q = quit: " % (i + 1, len(cells), name)).strip().lower()
                if ans == "q":
                    break
                if ans == "s":
                    continue
            meta = {"index": i + 1, "preference": pref, "w": [float(x) for x in w], "regime": reg, "trial": k + 1, "seed": args.seed, "policy": os.path.basename(args.policy)}
            if hasattr(policy, "set_preference"):
                policy.set_preference(w)
            entry = run_trial(drv, policy, REGIMES[reg], args, w, os.path.join(args.out_dir, name + ".npz"), meta, fault_table=dict(_a1_driver.FAULTS))
            print("   -> %s  faults %s  walk %.1f s" % (entry["outcome"], entry["faults"], entry["walk_time_s"] or 0.0))
            if not args.no_prompt:
                d = input("   distance walked along the lane in metres (Enter = unknown; 'mocap:<file>' to link a mocap file): ").strip()
                if d.startswith("mocap:"):
                    entry["mocap_file"] = d[6:]
                elif d:
                    try:
                        entry["distance_m"] = float(d)
                    except ValueError:
                        print("   not a number, stored as unknown")
            manifest.append(entry)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=1)
            if entry["outcome"] == "FAULT":
                print("   FAULT: the controller stays passive; clear it with SELECT on the remote (or restart) before the next trial")
                if not args.no_prompt:
                    input("   press Enter when the fault is cleared and the robot is safe... ")
                    drv.clear_faults()
            if args.rest_s > 0:
                time.sleep(args.rest_s)
    finally:
        drv.disarm()
        drv.stop()
        print("manifest: %s (%d trials)" % (manifest_path, len(manifest)))


if __name__ == "__main__":
    main()
