"""Measure command -> motor-state latency of the A1 driver. No policy.

    python3 scripts/latency_probe.py --box    # feet off the ground on a box: one thigh, Kp 15, 10 small steps

Holds the current pose, then steps one thigh target by 0.1 rad at a random moment and polls the driver state at
about 1 kHz. Reports how long after set_target the joint position first moves (> 0.004 rad) and how long until
tau_est first changes (> 0.3 N*m). The LowState tick advances once per ms, so the tick difference is the latency in
ms as the robot saw it; the host clock gives the same number from our side. This is the delay between our
set_target and the robot's answer, not the policy's own compute time (run logs have compute_s for that).
Use --flip-imu-x exactly as for run.py. Only the FL thigh moves; the hips and the other joints are held. Stops in damping on a fault or on motion elsewhere.
"""

import argparse
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from talon_a1 import _a1_driver, contract, sport  # noqa: E402
from motor_probe import Abort, clamp, OTHER_ABORT  # noqa: E402

KP = 15.0
KD = 1.0
STEP = 0.1  # rad
JOINT = contract.POLICY_JOINTS.index("FL_thigh_joint")
MOVE_Q = 0.004  # rad
MOVE_TAU = 0.3  # N*m
REPEATS = 10
SETTLE_S = 1.0
WAIT_S = 0.3  # give up on one step after this long


def sdk_q(state, i):
    return float(contract.sdk_to_policy(np.asarray(state.q))[i])


def sdk_tau(state, i):
    return float(contract.sdk_to_policy(np.asarray(state.tau_est))[i])


def send(drv, target):
    if not drv.set_target(list(contract.policy_to_sdk(target).astype(float)), KP, KD):
        raise Abort("target refused")


def hold_for(drv, target, seconds, ref):
    """Keep sending the target at 50 Hz (the driver's watchdog needs it) and watch the other joints."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        send(drv, target)
        s = drv.state()
        if s.faults:
            raise Abort("fault %d" % s.faults)
        q = contract.sdk_to_policy(np.asarray(s.q))
        if np.abs(np.delete(q - ref, JOINT)).max() > OTHER_ABORT:
            raise Abort("unexpected motion of another joint")
        time.sleep(contract.POLICY_DT)


def one_step(drv, hold, moved, ref):
    s0 = drv.state()
    q0, tau0, tick0 = sdk_q(s0, JOINT), sdk_tau(s0, JOINT), s0.tick
    t_send = time.monotonic()
    send(drv, moved)
    lat_q = lat_tau = None
    while time.monotonic() - t_send < WAIT_S:
        s = drv.state()
        dt = (time.monotonic() - t_send) * 1000.0
        if lat_tau is None and abs(sdk_tau(s, JOINT) - tau0) > MOVE_TAU:
            lat_tau = (dt, s.tick - tick0)
        if lat_q is None and abs(sdk_q(s, JOINT) - q0) > MOVE_Q:
            lat_q = (dt, s.tick - tick0)
        if lat_q and lat_tau:
            break
        send(drv, moved)  # same target again: keeps the watchdog fed, changes nothing
        time.sleep(0.0005)
    return lat_q, lat_tau


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--box", action="store_true", help="required on the robot: feet off the ground on a stable box")
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--flip-imu-x", action="store_true", help="same as run.py: needed while the IMU reads upside down")
    p.add_argument("--tilt-limit", type=float, default=0.5, help="rad; a robot lifted by hand or hanging may not be level")
    args = p.parse_args()

    cfg = _a1_driver.SafetyConfig()
    cfg.tilt_limit_rad = args.tilt_limit
    if args.flip_imu_x:
        cfg.flip_imu_x = True
    if _a1_driver.WITH_SDK:
        if not args.box:
            sys.exit("run with --box: the robot must stand on a box, feet off the ground")
        sport.require_stopped()
        print("Feet off the ground on a stable box, hands away from the legs. L2+B or Ctrl-C ends in damping.")
        input("Press Enter to start...")
    drv = _a1_driver.Driver(cfg, 0.002)
    drv.start()
    time.sleep(0.2)
    results = []
    failed = None
    try:
        ref = contract.sdk_to_policy(np.asarray(drv.state().q))
        hold = clamp(cfg, ref)
        moved = hold.copy()
        moved[JOINT] += STEP
        moved = clamp(cfg, moved)
        send(drv, hold)
        if not drv.arm():
            raise Abort("arm refused: no robot state, or a fault")
        for k in range(args.repeats):
            hold_for(drv, hold, SETTLE_S + random.random() * 0.2, ref)
            q_lat, tau_lat = one_step(drv, hold, moved, ref)
            results.append((q_lat, tau_lat))
            print("step %2d: q moved after %s, tau_est changed after %s" % (
                k + 1, "%.1f ms (%d ticks)" % q_lat if q_lat else "never", "%.1f ms (%d ticks)" % tau_lat if tau_lat else "never"))
            hold_for(drv, moved, 0.5, ref)
            hold_for(drv, hold, 0.5, ref)
    except Abort as e:
        failed = str(e)
    except KeyboardInterrupt:
        failed = "stopped by Ctrl-C"
    finally:
        drv.disarm()
        drv.stop()

    for name, idx in (("position q", 0), ("torque tau_est", 1)):
        ms = [r[idx][0] for r in results if r[idx]]
        if ms:
            print("%s: median %.1f ms, min %.1f, max %.1f (%d of %d steps)" % (
                name, float(np.median(ms)), min(ms), max(ms), len(ms), len(results)))
    print("result:", failed or "finished")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
