"""Run the deployment RMA policy on the A1 (or on the loopback desk plant when the driver was built without the SDK).

    python3 scripts/run.py --policy policies/rma_a1.npz

Hang the robot on a gantry for the first runs. Remote combos are listed in talon_a1/controller.py.
Ctrl-C disarms, leaves the motors damped and writes the run log.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from talon_a1 import _a1_driver, contract, sport  # noqa: E402
from talon_a1.controller import Controller, RunLog  # noqa: E402
from talon_a1.policy import RmaPolicy  # noqa: E402
from talon_a1.remote import Remote, combo  # noqa: E402


def desk_demo(drv):
    """Loopback only: press the remote combos a person would, so the full stack runs unattended."""
    for at, keys, ly in ((0.5, combo("L1", "A"), 0.0), (0.7, 0, 0.0), (4.0, combo("L1", "X"), 0.0),
                         (4.2, 0, 0.6), (7.0, combo("L1", "Y"), 0.0), (7.2, 0, 0.0), (9.0, combo("L1", "B"), 0.0)):
        time.sleep(max(0.0, at - (time.monotonic() - desk_demo.t0)))
        drv.sim_set_remote(keys, 0.0, ly, 0.0, 0.0)


def wait_for_robot(drv, timeout=1.0):
    first = drv.state().tick
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if drv.state().tick != first:
            return True
        time.sleep(0.01)
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--action-clip", type=float, default=1.0,
                   help="bring-up, restrained only: bound on the action in (0, 1], 1 is the trained behaviour")
    p.add_argument("--kp", type=float, default=contract.KP,
                   help="bring-up, restrained only: policy Kp in (0, %g], below 50 is outside training" % contract.KP)
    p.add_argument("--action-scale", type=float, default=contract.ACTION_SCALE,
                   help="rad per unit action in (0, %g]; below it is outside training" % contract.ACTION_SCALE)
    p.add_argument("--soft-start", type=float, default=1.0,
                   help="seconds to ease the action and gains in after L1+X, 0 = jump at once")
    p.add_argument("--stand-only", action="store_true", help="L1+X does nothing: the policy can never start")
    p.add_argument("--vx-max", type=float, default=0.5)
    p.add_argument("--vy-max", type=float, default=0.3)
    p.add_argument("--wz-max", type=float, default=0.5)
    p.add_argument("--stand-time", type=float, default=3.0)
    p.add_argument("--power-limit", type=float, default=300.0, help="W of sum |tau_est * dq|, latched fault after 20 ms")
    p.add_argument("--tilt-limit", type=float, default=0.5, help="rad, latched fault while armed")
    p.add_argument("--tau-limit", type=float, default=30.0, help="N*m on tau_est, latched fault while armed")
    p.add_argument("--damping-kd", type=float, default=1.0, help="Kd of the damping mode every fault ends in")
    p.add_argument("--duration", type=float, default=None, help="seconds; default runs until Ctrl-C")
    p.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    p.add_argument("--desk-demo", action="store_true", help="loopback build only: scripted remote presses")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()
    if not 0.0 < args.action_clip <= 1.0:
        p.error("--action-clip must be in (0, 1]")
    if not 0.0 < args.action_scale <= contract.ACTION_SCALE:
        p.error("--action-scale must be in (0, %g]" % contract.ACTION_SCALE)
    if args.soft_start < 0.0:
        p.error("--soft-start must not be negative")
    if not 0.0 < args.kp <= contract.KP:
        p.error("--kp must be in (0, %g]" % contract.KP)

    policy = RmaPolicy(args.policy)
    cfg = _a1_driver.SafetyConfig()
    cfg.tilt_limit_rad = args.tilt_limit
    cfg.tau_limit = args.tau_limit
    cfg.power_limit_w = args.power_limit
    if not 0.0 < args.damping_kd <= cfg.kd_max:
        p.error("--damping-kd must be in (0, %g]" % cfg.kd_max)
    cfg.damping_kd = args.damping_kd

    if _a1_driver.WITH_SDK:
        sport.require_stopped()
        print("LOW-LEVEL control of the real A1. The robot must hang on the gantry, remote in hand (L2+B = e-stop).")
        if not args.yes:
            input("Press Enter to start the motor loop...")
    else:
        print("Loopback desk plant: no robot is connected.")
    if args.desk_demo and _a1_driver.WITH_SDK:
        sys.exit("--desk-demo is for the loopback build only")

    drv = _a1_driver.Driver(cfg, 0.002)
    drv.start()
    print("motor loop running, SCHED_FIFO: %s" % drv.realtime)
    if not wait_for_robot(drv):
        print("warning: LowState.tick is not changing; arming will be refused until the robot answers")

    log = RunLog()
    ctl = Controller(drv, policy, Remote(args.vx_max, args.vy_max, args.wz_max), dict(_a1_driver.FAULTS),
                     stand_time=args.stand_time, action_clip=args.action_clip, log=log, policy_kp=args.kp,
                     allow_walk=not args.stand_only, soft_start=args.soft_start,
                     action_scale=args.action_scale)
    duration = args.duration
    if args.desk_demo:
        desk_demo.t0 = time.monotonic()
        threading.Thread(target=desk_demo, args=(drv,), daemon=True).start()
        duration = duration or 10.0
    try:
        ctl.run(duration)
    except KeyboardInterrupt:
        pass
    finally:
        drv.disarm()
        drv.stop()
        # Compressing the log takes seconds on the Atom; a second Ctrl-C here used to lose the whole run.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("saving the log, please wait...")
        final = drv.state()
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)
        path = os.path.join(args.log_dir, time.strftime("run_%Y%m%d_%H%M%S.npz"))
        meta = {"args": vars(args), "policy": policy.meta, "with_sdk": bool(_a1_driver.WITH_SDK),
                "realtime": drv.realtime, "policy_overruns": ctl.overruns, "motor_overruns": final.overruns,
                "motor_max_period_s": final.max_period}
        log.save(path, meta)
        print(json.dumps({k: meta[k] for k in ("policy_overruns", "motor_overruns", "motor_max_period_s")}))
        print("log: %s" % path)


if __name__ == "__main__":
    main()
