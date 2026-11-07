"""Check motor commands one joint at a time at low force, one joint per Enter. No policy.

    python3 scripts/motor_probe.py         # lying on the ground, Kp 5: joint, direction and gain scale from torque
    python3 scripts/motor_probe.py --box   # feet off the ground on a box, Kp 15: thigh and calf must really move

Holds the current pose, then moves one joint 0.1 rad (at most 0.5 N*m lying, 1.5 N*m on the box) and back. With
--box the hips are never moved, only held where they are, so a leg cannot swing into the box. Stops in damping on
the first wrong joint, fault, tilt or unexpected motion. The gains and the step are fixed on purpose and have no
command line option.
"""

import argparse
import os
import select
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from talon_a1 import _a1_driver, contract, sport  # noqa: E402

KP = 5.0
BOX_KP = 15.0
KD = 1.0
STEP = 0.1  # rad
RAMP_S = 1.0
HOLD_S = 1.0
TILT_LIMIT = 0.3  # rad: tighter than a run, the robot should stay flat on the ground
OTHER_ABORT = 0.15  # rad: a joint that was not commanded moved this far from where it started
OTHER_SUSPECT = 0.03  # rad
MOVED = 0.02  # rad of real motion that counts as an answer
TORQUE_ANSWER = 0.1  # N*m change in tau_est that counts as an answer

# Away from what the lying pose is close to: hips inward (they lie splayed), calves unfolding (near the limit).
DIRECTION = {
    "FL_hip_joint": -1.0, "FR_hip_joint": 1.0, "RL_hip_joint": -1.0, "RR_hip_joint": 1.0,
    "FL_thigh_joint": 1.0, "FR_thigh_joint": 1.0, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
    "FL_calf_joint": 1.0, "FR_calf_joint": 1.0, "RL_calf_joint": 1.0, "RR_calf_joint": 1.0,
}


class Abort(Exception):
    pass


def judge(sign, dq, dtau, derr, others, need_motion=False):
    """Verdict for one joint. dq, dtau, derr: changes from before the step to the end of the hold.

    need_motion: a free leg must really move. Its torque follows gravity as the pose changes, so torque is neither
    an answer nor a gain check there.
    """
    if others > OTHER_SUSPECT:
        return "WRONG", "another joint moved %.3f rad" % others
    q_says = 0 if abs(dq) < MOVED else (1 if dq * sign > 0 else -1)
    if need_motion:
        if q_says < 0:
            return "WRONG", "moved against the command"
        if q_says == 0:
            return "INCONCLUSIVE", "did not move"
        return "PASS", ""
    tau_says = 0 if abs(dtau) < TORQUE_ANSWER else (1 if dtau * sign > 0 else -1)
    if q_says < 0 or tau_says < 0:
        return "WRONG", "moved or pushed against the command"
    if q_says == 0 and tau_says == 0:
        return "INCONCLUSIVE", "no motion and no torque change"
    if tau_says > 0 and abs(derr) > 1e-3:
        ratio = dtau / (KP * derr)
        if not 0.5 <= ratio <= 1.5:
            return "WRONG", "tau_est is %.2f x Kp*error, the gain scale is off" % ratio
    return "PASS", ""


def clamp(cfg, q_policy):
    """The target the safety core really applies (policy order)."""
    q = contract.policy_to_sdk(q_policy).astype(float)
    for i in range(12):
        kind = i % 3
        q[i] = min(max(q[i], cfg.q_min[kind] + cfg.q_margin), cfg.q_max[kind] - cfg.q_margin)
    return contract.sdk_to_policy(q)


class Probe(object):
    def __init__(self, drv, cfg, fault_table, rows, kp, need_motion):
        self.drv = drv
        self.cfg = cfg
        self.fault_table = fault_table
        self.rows = rows
        self.kp = kp
        self.need_motion = need_motion
        self.t0 = time.monotonic()

    def read(self):
        s = self.drv.state()
        return s, contract.sdk_to_policy(np.asarray(s.q)), contract.sdk_to_policy(np.asarray(s.tau_est))

    def check(self):
        s = self.drv.state()
        if s.faults:
            names = [n for n, bit in sorted(self.fault_table.items()) if s.faults & bit]
            raise Abort("fault: %s" % ", ".join(names))
        if s.mode != 1:
            raise Abort("the driver left active mode")

    def send(self, target):
        ok = self.drv.set_target(list(contract.policy_to_sdk(target).astype(float)), self.kp, KD)
        if not ok:
            raise Abort("target refused")
        self.check()
        s, q, tau = self.read()
        self.rows.append(np.concatenate([[time.monotonic() - self.t0], target, q, tau]))

    def run_for(self, duration, target_at, ref, joint):
        """50 Hz for duration seconds. target_at(r) with r from 0 to 1."""
        n = int(round(duration / contract.POLICY_DT))
        for k in range(n):
            start = time.monotonic()
            self.send(target_at((k + 1.0) / n))
            _, q, _ = self.read()
            others = np.abs(np.delete(q - ref, joint)) if joint is not None else np.abs(q - ref)
            if others.max() > OTHER_ABORT:
                raise Abort("unexpected motion of %.3f rad" % others.max())
            time.sleep(max(0.0, contract.POLICY_DT - (time.monotonic() - start)))

    def hold_until_enter(self, target, prompt):
        """Keeps sending the hold target while waiting: a blocked input() would trip the 40 ms watchdog."""
        sys.stdout.write(prompt)
        sys.stdout.flush()
        while True:
            start = time.monotonic()
            self.send(target)
            if select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()
                return
            time.sleep(max(0.0, contract.POLICY_DT - (time.monotonic() - start)))

    def one_joint(self, i, ref, hold, auto):
        name = contract.POLICY_JOINTS[i]
        sign = DIRECTION[name]
        if auto:
            self.run_for(0.1, lambda r: hold, ref, None)
        else:
            self.hold_until_enter(hold, "%s: move %+.2f rad. Enter to go, Ctrl-C to stop: " % (name, sign * STEP))
        _, q0, tau0 = self.read()
        err0 = hold[i] - q0[i]
        moved = hold.copy()
        moved[i] += sign * STEP
        moved = clamp(self.cfg, moved)

        self.run_for(RAMP_S, lambda r: hold + r * (moved - hold), ref, i)
        self.run_for(HOLD_S, lambda r: moved, ref, i)
        _, q1, tau1 = self.read()
        dq = q1[i] - q0[i]
        dtau = tau1[i] - tau0[i]
        derr = (moved[i] - q1[i]) - err0
        others = np.abs(np.delete(q1 - q0, i)).max()
        self.run_for(RAMP_S, lambda r: moved + r * (hold - moved), ref, i)

        verdict, why = judge(sign, dq, dtau, derr, others, self.need_motion)
        print("%-15s cmd %+.3f | q %+.3f | tau_est %+.2f | others %.3f | %s %s"
              % (name, moved[i] - hold[i], dq, dtau, others, verdict, why))
        return verdict


def joints_to_probe(box):
    """Policy-order indices. On the box the hips stay out: 0-3 are only ever held."""
    return [i for i in range(12) if not (box and "hip" in contract.POLICY_JOINTS[i])]


def run(drv, cfg, fault_table, rows, results, auto=False, box=False):
    """Fills results {joint: verdict} and rows as it goes, so an abort keeps what happened before it."""
    kp = BOX_KP if box else KP
    probe = Probe(drv, cfg, fault_table, rows, kp, box)
    time.sleep(0.1)  # let the motor loop read the robot
    _, ref, _ = probe.read()
    hold = clamp(cfg, ref)
    print("holding the current pose at Kp %g, Kd %g" % (kp, KD))
    if not (drv.set_target(list(contract.policy_to_sdk(hold).astype(float)), kp, KD) and drv.arm()):
        raise Abort("arm refused: no robot state, or a fault")
    try:
        for i in joints_to_probe(box):
            verdict = probe.one_joint(i, ref, hold, auto)
            results[contract.POLICY_JOINTS[i]] = verdict
            if verdict == "WRONG":
                raise Abort("%s is wrong, stopping" % contract.POLICY_JOINTS[i])
    finally:
        drv.disarm()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    p.add_argument("--auto", action="store_true", help="loopback build only: no Enter between joints")
    p.add_argument("--box", action="store_true", help="feet off the ground on a box: Kp 15, thigh and calf only")
    args = p.parse_args()

    if _a1_driver.WITH_SDK:
        if args.auto:
            sys.exit("--auto is for the loopback build only")
        sport.require_stopped()
        where = "on a stable box, feet off the ground" if args.box else "lying on the ground"
        print("The robot must be %s, hands away from the legs. L2+B, Ctrl-C or a fault ends in damping." % where)
    cfg = _a1_driver.SafetyConfig()
    cfg.tilt_limit_rad = TILT_LIMIT
    drv = _a1_driver.Driver(cfg, 0.002)
    drv.start()
    results, rows, failed = {}, [], None
    try:
        run(drv, cfg, dict(_a1_driver.FAULTS), rows, results, args.auto, args.box)
    except Abort as e:
        failed = str(e)
    except KeyboardInterrupt:
        failed = "stopped by Ctrl-C"
    finally:
        drv.disarm()
        drv.stop()

    if rows:
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)
        path = os.path.join(args.log_dir, time.strftime("probe_%Y%m%d_%H%M%S.npz"))
        np.savez_compressed(path, rows=np.asarray(rows), columns=np.array("t, target 12, q 12, tau_est 12; policy order"))
        print("log: %s" % path)
    counts = {v: list(results.values()).count(v) for v in set(results.values())}
    print("result: %s" % (failed or "finished"), counts)
    sys.exit(1 if failed or set(counts) - {"PASS"} else 0)


if __name__ == "__main__":
    main()
