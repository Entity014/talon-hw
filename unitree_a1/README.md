# unitree_a1

Hardware deployment of the PROMO-on-A1 locomotion policy (talon-rl branch `research/v1`, task
`TalonRl-Promo-A1`) on a Unitree A1, running on the robot's onboard computer.

```
 50 Hz  Python (talon_a1/)                         500 Hz  C++ (_a1_driver, pybind11)
 ┌──────────────────────────────────────┐          ┌────────────────────────────────────────┐
 │ remote -> command (vx, vy, wz)       │          │ UDP LowState in  (unitree_legged_sdk)  │
 │ obs.py      o_t (45), history (10)   │ targets  │ SafetyCore: watchdog, clamp, e-stop,   │
 │ policy.py   E_H -> E_V -> actor      ├─────────►│   tilt, overspeed, PowerProtect latch  │
 │ controller  PASSIVE / STAND / WALK   │◄─────────┤ UDP LowCmd out   (Kp/Kd PD on motors)  │
 └──────────────────────────────────────┘  state   └────────────────────────────────────────┘
```

All timing-critical safety runs in the C++ loop, so a stalled or crashed Python process ends in
damping. The policy is a NumPy port of `PromoActorCritic`'s deployable path (no torch on the robot);
`export_policy.py` checks it against PyTorch.

## Interface contract

`talon_a1/contract.py` holds every value, with its source in talon-rl / Isaac Lab:

| | |
|---|---|
| obs `o_t` (45) | base ang vel 3, projected gravity 3, command 3, `q - q_default` 12, `dq` 12, last action 12. No scaling, no `base_lin_vel` (E_V estimates it) |
| history | 10 steps, oldest first, newest = `o_t`; filled with `o_0` on reset |
| preference `w` | 3 (tracking, stability, efficiency), fixed per run |
| action | `q_target = q_default + 0.25 a`, PD Kp 25, Kd 0.5, 50 Hz |
| joint order | policy: type-grouped FL/FR/RL/RR hip, thigh, calf. SDK: FR, FL, RR, RL legs. Remapped by name |

## Build

Desk (no robot; loopback plant, for development and tests):

```bash
pip install pybind11 numpy pytest
cmake -S . -B build -DTALON_A1_WITH_SDK=OFF -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
cmake --build build -j && (cd build && ctest) && python3 -m pytest tests
python3 scripts/run.py --policy policies/promo_a1.npz --desk-demo
```

Robot (onboard computer):

```bash
git clone -b 3.3.4 https://github.com/unitreerobotics/unitree_legged_sdk ~/unitree_legged_sdk   # match the A1 firmware
pip3 install numpy pybind11==2.9.2   # motion computer 192.168.123.161 is Ubuntu 16.04 (Python 3.5); 2.9 is the last pybind11 for 3.5
cmake -S . -B build -DUNITREE_SDK_DIR=$HOME/unitree_legged_sdk -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
cmake --build build -j
```

The SDK library is picked by architecture (`amd64` / `arm64` / `arm32`). Only `comm.h`, `udp.h` and
`safety.h` are used, so neither LCM nor Boost is needed.

## Policy export (training machine)

```bash
uv run --project ../../talon-rl python scripts/export_policy.py \
    --checkpoint ../../talon-rl/runs/<run>/final.pt --out policies/promo_a1.npz
```

Writes `promo_a1.npz` and `promo_a1.parity.json` (fails if the NumPy action differs from PyTorch by more
than 1e-5). Copy both to the robot.

## Run

```bash
sudo python3 scripts/run.py --policy policies/promo_a1.npz --preference balanced
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
stopped for 50 ms, |roll| or |pitch| > 0.8 rad, |dq| > 30 rad/s, bad target / gains, SDK PowerProtect,
e-stop. Targets are clamped 0.05 rad inside the `a1_const.h` joint limits.

## Bring-up checklist (not yet done on hardware)

Nothing here has run on a robot yet. In order, with the robot hung on a gantry:

1. Onboard computer: `lsb_release -a; uname -m; python3 --version; lscpu`. Build, `ctest`.
2. Nothing else bound to UDP 8080 (`ss -ulpn | grep 8080`), and the A1 switched to low-level mode the way
   your firmware's manual says. If the SDK cannot bind the port it calls `exit()`.
3. `scripts/inspect_sensors.py` (motors stay damped), by hand:
   - move each leg: the joint that changes is the one printed (joint order);
   - tilt nose down / left side down: projected gravity goes +x / +y; yaw left gives gyro +z
     (IMU frame vs Isaac's base frame);
   - sticks: left up = +vx, left = +vy, right stick left = +wz; buttons show up by name.
4. `run.py` with `--power-level 1`, L1+A only: check the stand ramp, holding the default pose with Kp 25
   (from lying down it may be too soft), then L1+B.
5. Check timing in the log: `compute_s` p99, `loop_overruns`, `motor_max_period_s`.
6. Then L1+X with the sticks centered, on the gantry, before the ground.

## Differences from training to record

- Action clipped to ±4 (training never clipped) and joint targets clamped to the hardware limits.
- No observation noise; preference fixed for the whole run; commands limited by `--vx-max/--vy-max/--wz-max`.
- PD runs in the motor firmware; training used Isaac Lab's DCMotor model (torque-speed saturation 33.5 N·m,
  21 rad/s).
- Gyro and orientation come from the A1 IMU, not the simulated base frame (checklist step 3).
