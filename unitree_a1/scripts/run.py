"""Run the PROMO policy on the A1 (or on the loopback desk plant when the driver was built without the SDK).

    python3 scripts/run.py --policy policies/promo_a1.npz --preference balanced

Hang the robot on a gantry for the first runs. Remote combos are listed in talon_a1/controller.py.
Ctrl-C disarms, leaves the motors damped and writes the run log.
"""

import argparse
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from talon_a1 import _a1_driver, contract  # noqa: E402
from talon_a1.controller import Controller, RunLog  # noqa: E402
from talon_a1.policy import PromoPolicy  # noqa: E402
from talon_a1.remote import Remote, combo  # noqa: E402


def parse_preference(text):
    if text in contract.PREFERENCE_ANCHORS:
        return contract.PREFERENCE_ANCHORS[text]
    w = [float(v) for v in text.split(",")]
    if len(w) != contract.PREF_DIM:
        raise argparse.ArgumentTypeError("preference: an anchor name or three comma-separated weights")
    return w


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
    p.add_argument("--preference", type=parse_preference, default="balanced",
                   help="anchor (%s) or w_track,w_stab,w_eff" % ", ".join(contract.PREFERENCE_ANCHORS))
    p.add_argument("--vx-max", type=float, default=0.5)
    p.add_argument("--vy-max", type=float, default=0.3)
    p.add_argument("--wz-max", type=float, default=0.5)
    p.add_argument("--stand-time", type=float, default=3.0)
    p.add_argument("--power-level", type=int, default=1, help="SDK PowerProtect level 1-10, 0 = off")
    p.add_argument("--tilt-limit", type=float, default=0.8, help="rad, latched fault while armed")
    p.add_argument("--duration", type=float, default=None, help="seconds; default runs until Ctrl-C")
    p.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    p.add_argument("--desk-demo", action="store_true", help="loopback build only: scripted remote presses")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()  # argparse also runs `type` on the string default

    policy = PromoPolicy(args.policy, args.preference)
    cfg = _a1_driver.SafetyConfig()
    cfg.tilt_limit_rad = args.tilt_limit

    if _a1_driver.WITH_SDK:
        print("LOW-LEVEL control of the real A1. The robot must hang on the gantry, remote in hand (L2+B = e-stop).")
        if not args.yes:
            input("Press Enter to start the motor loop...")
    else:
        print("Loopback desk plant: no robot is connected.")
    if args.desk_demo and _a1_driver.WITH_SDK:
        sys.exit("--desk-demo is for the loopback build only")

    drv = _a1_driver.Driver(cfg, 0.002, args.power_level)
    drv.start()
    print("motor loop running, SCHED_FIFO: %s" % drv.realtime)
    if not wait_for_robot(drv):
        print("warning: LowState.tick is not changing; arming will be refused until the robot answers")

    log = RunLog()
    ctl = Controller(drv, policy, Remote(args.vx_max, args.vy_max, args.wz_max), dict(_a1_driver.FAULTS),
                     stand_time=args.stand_time, log=log)
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
