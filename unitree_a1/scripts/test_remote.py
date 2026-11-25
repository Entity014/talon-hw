#!/usr/bin/env python3
"""Interactive test of Wireless Remote / Joypad (Unitree A1).

Tests both raw joystick inputs and simulated policy commands WITHOUT moving any motor.
If running with SDK (on robot): reads real remote from LowState.
If running without SDK (desk/laptop): allows testing remote parser and key combinations.

Usage:
    python3 scripts/test_remote.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from talon_a1 import _a1_driver, contract, sport
from talon_a1.controller import CLEAR, STAND_UP, START_WALK, STOP_WALK, TO_PASSIVE
from talon_a1.remote import KEYS, Remote, combo

ESTOP = combo("L2", "B")


def main():
    if _a1_driver.WITH_SDK:
        sport.require_stopped()
        print("Connecting to robot remote stream (motors stay DAMPED)...")
    else:
        print("Running in loopback mode (no SDK). Keyboard/loopback demo.")

    drv = _a1_driver.Driver(_a1_driver.SafetyConfig(), 0.002)
    drv.start()

    remote = Remote(vx_max=0.5, vy_max=0.3, wz_max=0.5)

    print("\n" + "=" * 65)
    print(" Unitree A1 Remote / Joypad Live Inspector")
    print("=" * 65)
    print("Combos:")
    print("  L1 + A   -> STAND UP (Ramp to standing pose)")
    print("  L1 + X   -> START WALK (Policy starts walking)")
    print("  L1 + Y   -> STOP WALK (Ramp back to stand)")
    print("  L1 + B   -> PASSIVE (Motors damp)")
    print("  SELECT   -> CLEAR FAULTS")
    print("  L2 + B   -> EMERGENCY STOP (Hard latch in driver)")
    print("Sticks:")
    print("  Left Stick UP/DOWN    -> vx (+ forward / - backward)")
    print("  Left Stick LEFT/RIGHT -> vy (+ left / - right)")
    print("  Right Stick LEFT/RIGHT-> wz (+ turn left / - turn right)")
    print("=" * 65)
    print("Press Ctrl-C to exit.\n")
    time.sleep(1.0)

    try:
        while True:
            state = drv.state()
            remote.update(state)
            cmd = remote.command(state)

            pressed_keys = [k for k, bit in KEYS.items() if state.keys & bit]

            combos_detected = []
            if remote.pressed(STAND_UP):
                combos_detected.append(">>> STAND_UP (L1+A) <<<")
            if remote.pressed(START_WALK):
                combos_detected.append(">>> START_WALK (L1+X) <<<")
            if remote.pressed(STOP_WALK):
                combos_detected.append(">>> STOP_WALK (L1+Y) <<<")
            if remote.pressed(TO_PASSIVE):
                combos_detected.append(">>> TO_PASSIVE (L1+B) <<<")
            if remote.pressed(CLEAR):
                combos_detected.append(">>> CLEAR_FAULTS (SELECT) <<<")
            if (state.keys & ESTOP) == ESTOP:
                combos_detected.append("!!! E-STOP (L2+B) !!!")

            combo_str = " | ".join(combos_detected) if combos_detected else "-"

            # Overwrite terminal line
            sys.stdout.write("\x1b[2J\x1b[H")
            print("=" * 65)
            print(f" TICK: {state.tick:<8} | FAULTS: {state.faults:<4} | SDK: {_a1_driver.WITH_SDK}")
            print("=" * 65)
            print(f" Raw Sticks:  lx={state.lx:+.3f}  ly={state.ly:+.3f}  rx={state.rx:+.3f}  ry={state.ry:+.3f}")
            print(f" Active Keys: {' '.join(pressed_keys) if pressed_keys else 'None'}")
            print(f" Last Combo:  {combo_str}")
            print("-" * 65)
            print(f" Policy Velocity Command (m/s, rad/s):")
            print(f"   vx (forward) : {cmd[0]:+6.3f} m/s  [max: 0.50]")
            print(f"   vy (lateral) : {cmd[1]:+6.3f} m/s  [max: 0.30]")
            print(f"   wz (yaw rate): {cmd[2]:+6.3f} rad/s [max: 0.50]")
            print("=" * 65)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nExiting remote monitor.")
    finally:
        drv.stop()


if __name__ == "__main__":
    main()
