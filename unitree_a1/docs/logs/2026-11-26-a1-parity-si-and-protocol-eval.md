# A1 `oursA` (`checkpoint_04900`): Model Parity, Hardware System ID, Protocol Eval, and MuJoCo SI Tuning — 2026-11-26

สรุปการทำงานและผลการทดลองทั้งหมดของวันที่ 2026-11-26 บน Unitree A1 จริง และ MuJoCo Simulator

---

## 1. Model Conversion Parity (`.pt` → `.onnx` → `.npz`)

ตรวจสอบด้วย [scripts/check_policy_parity.py](file:///home/xero/Master's%20Degree/Thesis/talon-hw/unitree_a1/scripts/check_policy_parity.py) ทั้ง 3 ระดับ (Static 100 samples: zeros/ones/normal/uniform[-10,10], Layer-by-Layer, และ Real-Log Replay):

| Model | Static Max Abs Diff (`pt` vs `npz` / `onnx`) | Layer-by-Layer Max Abs | Replay / Closed-Loop | ผลสรุป |
|---|---|---|---|---|
| `policies/base_v2` | `1.91e-06` / `4.77e-06` | `<= 3.58e-07` | — | **PASS** |
| `policies/oursA` (`final.pt`) | `2.86e-06` / `1.05e-05` | `<= 1.91e-06` | `5.96e-07` (`pt` vs `npz`, 2,003 ticks) | **PASS** |
| `policies/oursA/checkpoint_04900` | `2.86e-06` (`pt` vs `npz`) | `<= 1.91e-06` | MuJoCo 500-step joint diff `2.45e-06` | **PASS** |

- Export [policies/oursA/checkpoint_04900.npz](file:///home/xero/Master's%20Degree/Thesis/talon-hw/unitree_a1/policies/oursA/checkpoint_04900.npz) จาก `checkpoint_04900.pt` (max abs parity error `4.17e-07`)
- ยืนยันว่า **`.pt → .onnx → .npz` ไม่ได้ทำให้โมเดลพัง** และ `OnnxPolicy` กับ `PromoNumpyPolicy` ให้ค่า action ตรงกันระดับ `~1e-6`

---

## 2. Hardware System Identification (SI) บน Unitree A1 จริง

1. **Latency (`scripts/latency_probe.py --box --flip-imu-x`, `Kp=15, Kd=1`, 10 steps):**
   - Command → Torque (`tau_est`) delay: **median `9.4 ms`** (`5.7–10.2 ms`, ~10 ticks)
   - Command → Position (`q`) lag (`> 0.004 rad`): **median `23.5 ms`** (`22.6–25.8 ms`, 24–26 ticks)
2. **Motor & Joint Mapping (`scripts/motor_probe.py --box --flip-imu-x`, `Kp=15, Kd=1`):**
   - เพิ่ม `--flip-imu-x` ใน [scripts/motor_probe.py](file:///home/xero/Master's%20Degree/Thesis/talon-hw/unitree_a1/scripts/motor_probe.py) (แก้ fault `tilt`, `tests/test_motor_probe.py` ผ่าน `18/18`)
   - **PASS 8/8 ข้อต่อ** (`FL/FR/RL/RR` thigh & calf), `others = 0.000` ทุกข้อต่อ (`logs/probe_20261127_031010.npz`)
   - ยืนยันว่ามอเตอร์, เอ็นโค้เดอร์, ทิศทาง, และ PD gain scale ปกติทั้งขาซ้ายและขาขวา
3. **Wire PD Truncation (`talon_a1/contract.py`):**
   - Sim เทรนด้วย fixed `Kp = 55.0, Kd = 0.80` (ไม่มี DR ของ `Kp/Kd`)
   - บนโปรโตคอล A1: `Kp = 55.0 → 55.00` (ตรง 100%), `Kd = 0.80 → 12/16 = 0.75` (`-6.25%`)
4. **IMU Static Offset (`--imu-zero`):**
   - คาลิเบรตในท่า `STAND` (รอ 4–5 วินาทีหลัง `L1+A` ก่อนกด `L1+X`): `roll_bias ≈ +0.058 – +0.062 rad` (~3.4°), `pitch_bias ≈ +0.017 – +0.019 rad` (~1.0°)

---

## 3. ผลเทส `checkpoint_04900.npz` แบบ Manual (`scripts/run.py`)

### 3.1 `action-smooth` และ `dq-noise` (`w=[0.8, 0.1, 0.1]`, `--flip-imu-x`)
| Run Log | Config | WALK | `act_rate_rms` (นิ่ง / เดิน) | `gyro_rms` roll / pitch (rad/s) | Power เฉลี่ย |
|---|---|---:|---:|---:|---:|
| `run_20261127_024004.npz` | `smooth=0.0, dq=0.0` | 47.2s | `0.111 / 0.248` | `0.411 / 0.359` | `26.0 W` |
| `run_20261127_024309.npz` | `smooth=0.0, dq=0.1` | 26.2s | `0.124 / 0.252` | `0.463 / 0.364` | `29.6 W` |
| `run_20261127_024344.npz` | `smooth=0.1, dq=0.1` | 15.7s | `0.122 / 0.197` | `0.306 / 0.321` | `19.0 W` |
| `run_20261127_024226.npz` | `smooth=0.3, dq=0.1` | 31.6s | `0.117 / 0.156` | **`0.266 / 0.300`** | **`17.0 W`** |
| `run_20261127_024722.npz` | `smooth=0.3, dq=0.0` | 24.4s | **`0.094`** `/ 0.187` | `0.385 / 0.373` | `25.7 W` |

### 3.2 การลอง Compensation Flags (ไม่ใช่ pure zero-shot)
- `--pitch-bias 0.12 --front-lift-bias 0.12` (`run_20261127_030039.npz`): ถ่ายน้ำหนักจากขาหน้ามาขาหลังได้จริง (`FWD foot_force` หน้า `184/132 → 157/120`, หลัง `25/81 → 66/90`) แต่ค่า `0.12` แรงเกินไปจน `act_rate`, `gyro_rms`, และ `Power` สูงขึ้น
- `--lead-ms 10` (`run_20261127_031441.npz`): ขาหลังขวา `RR` ลอย (`foot_force <= 5` ถึง `27.7%`), แนวทแยง `FR–RL` รับน้ำหนัก `269 N` vs `FL–RR` `158 N`
- `--kp 48 --w 0.5 0.4 0.1` ไม่ใส่ `--imu-zero` (`run_20261127_031602.npz`): ขาหลังลอยคู่ (`RR` `34.2%`, `RL` `27.1%`) และ `projected_gravity` `gy = -0.0767`
- **ข้อสรุป:** ตัดสินใจกลับไปใช้แนวทาง **pure zero-shot** (ใส่เฉพาะ `--flip-imu-x` / `--imu-zero` ที่เป็นฮาร์ดแวร์อินเทอร์เฟซ) และแก้ Sim-to-Real gap ด้วยการ retrain ใหม่

---

## 4. Scripted Interactive Evaluation (`logs/interactive1` & `logs/interactive2`)

เพิ่ม `--interactive` (`a=replay, b=move_fast, c=move_slow, w=switch pref, q=quit`) และ `--imu-zero` ใน [scripts/eval_protocol.py](file:///home/xero/Master's%20Degree/Thesis/talon-hw/unitree_a1/scripts/eval_protocol.py) (`tests/test_eval_protocol.py` ผ่าน `7/7`)

### `logs/interactive1` (`--action-smooth 0.0 --dq-noise 0.0 --imu-zero`)
| Preference | Regime | Outcome | `power_w` | `attitude_deg` | `gyro_rp` (rad/s) | `action_rate` (1/s) | `torque_rate` (Nm/s) |
|---|---|---|---:|---:|---:|---:|---:|
| `tracking` | `move_fast` | OK | `42.6 W` | `4.35°` | `0.514` | `11.1` | `145` |
| `stability` | `move_fast` | OK | `33.7 W` | `3.21°` | `0.258` | `9.56` | `115` |
| `efficiency` | `move_fast` | FAULT (`stale_target`) | — | — | — | — | — |
| `tracking` | `replay` | OK | `10.1 W` | `9.25°` | `0.175` | `4.75` | `58.9` |
| `stability` | `replay` | OK | `7.33 W` | `3.01°` | `0.118` | `4.01` | `46.9` |
| `efficiency` | `replay` | OK | `1.38 W` | `4.36°` | `0.024` | `0.407` | `8.74` |
| `tracking / stab / eff` | `move_slow` | OK | `1.10 / 1.13 / 1.08 W` | `5.02° / 2.44° / 2.41°` | `0.016–0.026` | `0.26–0.55` | `7.9–9.7` |

### `logs/interactive2` (`--action-smooth 0.0 --dq-noise 0.0 --imu-zero`)
| Preference | Regime | Outcome | `power_w` | `attitude_deg` | `gyro_rp` (rad/s) | `action_rate` (1/s) | `torque_rate` (Nm/s) |
|---|---|---|---:|---:|---:|---:|---:|
| `balanced` | `move_fast` | OK | `47.4 W` | `3.44°` | `0.500` | `12.8` | `164` |
| `efficiency` | `move_fast` | OK | `1.51 W` | `4.50°` | `0.0329` | `0.905` | `12.9` |

**ข้อสังเกต:** ในสภาพจริง หุ่นเดินได้ชัดเจนเฉพาะ `tracking` (`move_fast`, และเดินช้าๆ ใน `move_slow` ได้บ้างเมื่อ `smooth=0.0`) และ `balanced` (`move_fast`) ส่วน `efficiency` ยืนค้างไม่เดิน (`1.51 W`)

---

## 5. MuJoCo HW-SI Tuning & Sim-to-Real Gap (`scripts/mujoco_si_eval.py`)

สร้าง [scripts/mujoco_si_eval.py](file:///home/xero/Master's%20Degree/Thesis/talon-hw/unitree_a1/scripts/mujoco_si_eval.py) และจูนพารามิเตอร์ให้พฤติกรรมตรงกับ `logs/interactive1` + `logs/interactive2`:
- **Tuned HW-SI Parameters:** `delay_ms = 10.0`, `Kp = 55.0`, `Kd = 0.75`, `frictionloss = 1.0 Nm`, `damping = 0.2 Nm·s/rad`

| Condition | Regime | Preference | `vx` (m/s) | `power` (W) | `gyro_rp` (rad/s) | `act_rate` (1/s) |
|---|---|---|---:|---:|---:|---:|
| Ideal Sim (`0ms, Kd=0.8, fl=0, damp=0`) | `move_fast` | `tracking` | `+0.583` | `45.9` | `0.341` | `13.41` |
| Ideal Sim | `move_fast` | `stability` | `+0.496` | `36.1` | `0.377` | `10.42` |
| Ideal Sim | `move_fast` | `efficiency` | `+0.352` | `27.8` | `0.341` | `8.69` |
| Ideal Sim | `move_slow` | `tracking / stab / eff` | `0.300 / 0.243 / 0.089` | `25.1 / 15.6 / 14.8` | — | — |
| **HW-SI Sim (`10ms, Kd=0.75, fl=1.0, damp=0.2`)** | `move_fast` | `tracking` | **`+0.518`** | `64.9` | `0.376` | `16.02` |
| **HW-SI Sim** | `move_fast` | `balanced` | **`+0.492`** | `52.5` | `0.346` | **`12.85`** |
| **HW-SI Sim** | `move_fast` | `stability` | `+0.440` | `41.7` | `0.389` | **`10.56`** |
| **HW-SI Sim** | `move_fast` | `efficiency` | **`+0.000`** | `0.0` | `0.000` | `0.00` |
| **HW-SI Sim** | `move_slow` | `tracking` | **`+0.094`** | `14.5` | `0.128` | `8.15` |
| **HW-SI Sim** | `move_slow` | `stab / eff / bal` | `0.001–0.003` | `0.0–0.2` | `0.000–0.015` | `0.01–1.06` |

---

## 6. แนวทางสำหรับ Retrain รอบถัดไป (เทียบ `table_domain_randomization.tex`)

สิ่งที่ขาดใน [table_domain_randomization.tex](file:///home/xero/Master's%20Degree/Thesis/talon-thesis/05_Paper/JCSSE2027/figures/table_domain_randomization.tex) และควรเพิ่มตอน retrain ทั้ง `base_v2` และ `oursA`:
1. **Action Delay:** สุ่ม `[0, 20] ms` (ครอบคลุม `9.4 ms` torque / `23.5 ms` pos lag)
2. **Joint Friction & Damping:** สุ่ม Coulomb friction `[0.2, 1.2] Nm` และ viscous damping `[0.05, 0.3] Nm·s/rad`
3. **PD Gain Randomization:** `Kp ∈ [45, 60]`, `Kd ∈ [0.5, 1.0]` (ครอบ `Kd = 0.75`)
4. **Static IMU Offset per Episode:** สุ่ม roll/pitch bias `±0.07 rad` (ครอบ `roll ≈ 0.06`, `pitch ≈ 0.018 rad`)
5. **Symmetric `default_q`:** ใช้ `thigh` หน้า-หลังเท่ากัน (เช่น `0.8` หรือ `0.9 rad` ทุกขา) + foot clearance / air-time reward
