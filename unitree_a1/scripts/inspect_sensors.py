"""Print what the policy would see, with the motors damped (never arms). First thing to run in the lab.

    python3 scripts/inspect_sensors.py

Move the robot by hand and check against the README bring-up list: joint names and signs, projected gravity,
gyro axes, remote sticks and buttons.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from talon_a1 import _a1_driver, contract, sport  # noqa: E402
from talon_a1.obs import projected_gravity  # noqa: E402
from talon_a1.remote import KEYS, Remote  # noqa: E402


def main():
    if _a1_driver.WITH_SDK:
        sport.require_stopped()
    drv = _a1_driver.Driver(_a1_driver.SafetyConfig(), 0.002, 1)
    drv.start()
    remote = Remote(1.0, 1.0, 1.0)
    np.set_printoptions(precision=3, suppress=True, linewidth=140)
    try:
        while True:
            s = drv.state()
            q = contract.sdk_to_policy(np.asarray(s.q)) - contract.DEFAULT_JOINT_POS
            keys = [k for k, bit in KEYS.items() if s.keys & bit]
            sys.stdout.write("\x1b[2J\x1b[H")
            print("tick %d  loops %d  overruns %d  faults %d  sdk %s" % (s.tick, s.loops, s.overruns, s.faults,
                                                                        _a1_driver.WITH_SDK))
            for name, v in zip(contract.POLICY_JOINTS, q):
                print("  %-16s q - default % .3f" % (name, v))
            print("projected gravity", projected_gravity(s.quat), "  (flat: 0 0 -1, nose down: +x, left side down: +y)")
            print("gyro             ", np.asarray(s.gyro), "  (yaw left: +z, roll right-down: +x, nose down: +y)")
            print("rpy              ", np.asarray(s.rpy))
            print("remote command   ", remote.command(s), "  (left stick up: +vx, left: +vy; right stick left: +wz)")
            print("keys             ", " ".join(keys))
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        drv.stop()


if __name__ == "__main__":
    main()
