# unitree_a1

Hardware deployment of the deployment RMA locomotion policy (talon-rl branch `deployment`, task
`TalonRl-PromoRma-A1`, frozen checkpoint `phase2_500.pt`) on a Unitree A1, running on the robot's onboard
computer.

```
 50 Hz  Python (talon_a1/)                         500 Hz  C++ (_a1_driver, pybind11)
 ┌──────────────────────────────────────┐          ┌────────────────────────────────────────┐
 │ remote -> command (vx, vy, wz)       │          │ UDP LowState in  (unitree_legged_sdk)  │
 │ obs.py      x_t (33), history (50)   │ targets  │ SafetyCore: watchdog, clamp, e-stop,   │
 │ policy.py   phi -> v_est -> actor    ├─────────►│   tilt, overspeed, PowerProtect latch  │
 │ controller  PASSIVE / STAND / WALK   │◄─────────┤ UDP LowCmd out   (Kp/Kd PD on motors)  │
 └──────────────────────────────────────┘  state   └────────────────────────────────────────┘
```

All timing-critical safety runs in the C++ loop, so a stalled or crashed Python process ends in
damping. The policy is a NumPy port of `talon_rl.rma.networks`' deployable path (no torch on the robot);
`export_policy.py` checks it against PyTorch.

## Interface contract

`talon_a1/contract.py` holds every value, with its source in talon-rl / Isaac Lab:

| | |
|---|---|
| obs `x_t` (33) | base ang vel 3, projected gravity 3, command 3, raw `q` 12, `dq` 12. No scaling or normalization, no `base_lin_vel` (the velocity estimator supplies it) |
| history | 50 steps of `(x, a)`, oldest first, up to `t-1`; zeros on reset, as in training |
| network | `phi(history) -> z_hat (8)`; `velocity(z_hat, last 5 x) -> v_hat (3)`; `actor(x, a_prev, z_hat, v_hat) -> tanh a (12)` |
| action | `q_target = q_default + 0.4 a`, PD Kp 55, Kd 0.8, 50 Hz |
| default pose | hip +0.05 left / -0.05 right, thigh 0.8, calf -1.4 |
| joint order | policy: type-grouped FL/FR/RL/RR hip, thigh, calf. SDK: FR, FL, RR, RL legs. Remapped by name |

## Build

Desk (no robot; loopback plant, for development and tests):

```bash
pip install pybind11 numpy pytest
cmake -S . -B build -DTALON_A1_WITH_SDK=OFF -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
cmake --build build -j && (cd build && ctest) && python3 -m pytest tests
python3 scripts/run.py --policy policies/rma_a1.npz --desk-demo
```

Robot (onboard computer):

```bash
git clone -b 3.3.4 https://github.com/unitreerobotics/unitree_legged_sdk ~/unitree_legged_sdk   # match the A1 firmware
# motion computer 192.168.123.161: Ubuntu 16.04, Python 3.5, CMake 3.5, no pip. On another machine:
#   pip download --no-deps --only-binary=:all: --python-version 3.5 --implementation cp --abi cp35m \
#       --platform manylinux1_x86_64 numpy==1.18.5
#   pip download --no-deps --only-binary=:all: --python-version 3.5 pybind11==2.9.2   # last pybind11 for 3.5
# then copy both wheels over and unpack them, since a wheel is a zip:
for w in ~/a1_wheels/*.whl; do python3 -m zipfile -e "$w" ~/.local/lib/python3.5/site-packages/; done
cmake -H. -Bbuild -DUNITREE_SDK_DIR=$HOME/unitree_legged_sdk -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
cmake --build build -- -j2
```

The SDK library is picked by architecture (`amd64` / `arm64` / `arm32`). Only `comm.h`, `udp.h` and
`safety.h` are used, so Boost is not needed, but the SDK library itself links against LCM (`liblcm`).

## Policy export (training machine)

```bash
PYTHONPATH=../../talon-rl/src python3 scripts/export_policy.py \
    --checkpoint policies/phase2_500.pt --out policies/rma_a1.npz
```

Writes `rma_a1.npz` and `rma_a1.parity.json` (fails if the NumPy action differs from PyTorch by more
than 1e-5). Only the actor, phi and the velocity estimator are exported. Copy both to the robot.

## Run

```bash
sudo python3 scripts/run.py --policy policies/rma_a1.npz
```

`sudo` (or `CAP_SYS_NICE`) lets the motor loop use `SCHED_FIFO`; it still runs without, with more jitter.
Every run writes `logs/run_<time>.npz` (state, obs, action, v_hat, targets, timing, faults).

| Remote | |
|---|---|
| L1 + A | PASSIVE -> STAND: arm, ramp to the default pose (`--stand-time`) |
| L1 + X | STAND -> WALK (after the ramp) |
| L1 + Y | WALK -> STAND |
| L1 + B | -> PASSIVE (damping) |
| **L2 + B** | **e-stop**, latched in C++ in every mode |
| SELECT | clear latched faults (in PASSIVE) |

Latched faults (`_a1_driver.FAULTS`): no new target for 40 ms (20–40 ms holds the last one), LowState
stopped for 50 ms, |roll| or |pitch| > 0.5 rad (`--tilt-limit`), |dq| > 30 rad/s, |tau_est| > 30 N·m
(`--tau-limit`), bad target / gains, SDK PowerProtect, e-stop. Faults end in damping (`Kp = 0`, Kd from
`--damping-kd`), which slows the robot down but does not hold it up. Targets are clamped 0.05 rad inside the
`a1_const.h` joint limits.

## Bring-up checklist (not yet done on hardware)

Nothing here has run on a robot yet. In order, with the robot hung on a gantry:

1. Onboard computer: `lsb_release -a; uname -m; python3 --version; lscpu`. Build, `ctest`.
2. `sudo scripts/sport_mode.sh stop`. Unitree's `A1_sport_1` commands the MCU from its own ports, so a free UDP
   8080 does not mean nothing else is commanding the motors. Reboot restores sport mode. Also nothing else bound
   to UDP 8080 (`ss -ulpn | grep 8080`): if the SDK cannot bind the port it calls `exit()`.
3. `scripts/inspect_sensors.py` (motors stay damped), by hand:
   - move each leg: the joint that changes is the one printed (joint order);
   - tilt nose down / left side down: projected gravity goes +x / +y; yaw left gives gyro +z
     (IMU frame vs Isaac's base frame);
   - sticks: left up = +vx, left = +vy, right stick left = +wz; buttons show up by name.
4. `scripts/motor_probe.py`, robot lying on the ground: commands each joint 0.1 rad at Kp 5, checks joint,
   direction and gain scale.
5. `run.py --stand-only --power-level 1`, L1+A only: check the stand ramp and holding the default pose. Standing uses
   sport mode's Kp 50, Kd 3 and, from lying, its crouch first; L1+X switches to the policy's Kp 55, Kd 0.75. Compare
   `tau_est` in the log with `Kp (q_target - q) - Kd dq`, then L1+B.
6. Check timing in the log: `compute_s` p99, `loop_overruns`, `motor_max_period_s`.
7. Then L1+X with the sticks centered, feet off the ground and body strapped down, stepping up the bring-up limits:
   `--action-clip 0.1 --kp 20`, `0.25 / 35`, `0.5 / 55`, then the defaults. These limits only check that the
   commands reach the motors correctly. They are outside training, so never walk with them.
8. Only then on the ground with an overhead restraint: command 0 first, then `vx = 0.1`.

## Differences from training to record

- The deterministic actor mean is used (training sampled), and joint targets are clamped to the hardware limits.
- No added observation noise; motor strength is 1 and Kp/Kd are fixed at the trained centre (training randomized
  them); commands limited by `--vx-max/--vy-max/--wz-max`.
- PD runs in the motor firmware; training used Isaac Lab's DCMotor model. Kp and Kd are commanded in every
  LowCmd, and the SDK does not report them back, so the firmware's gain units are checked from `tau_est`.
- Gyro and orientation come from the A1 IMU, not the simulated base frame (checklist step 3).
