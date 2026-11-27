# A1 `oursA_v1_drv2` (`checkpoint_06800`): Protocol Evaluation (36 Trials) & Box Probes — 2026-11-27

สรุปผลการทดสอบภาคสนามบน Unitree A1 จริง สำหรับ Policy `oursA_v1_drv2` (`checkpoint_06800.npz`) เทียบกับผล `oursA` เมื่อวันที่ 2026-11-26

---

## 0. ข้อมูลอ้างอิง Session
- **Policy:** `policies/oursA_v1_drv2/checkpoint_06800.npz`
  - `sha256sum`: `7ccda79fb8e012a0d6b3d1ed06a3089c28a2c68758a4d0dae9eeaebdc0718090`
  - Lineage: `oursA v1 final.pt` (it 5000) resumed 2000 iterations (`5001–7001`) under hardware DR (action delay 0–20 ms, PD gains, joint friction/damping, IMU bias, CoM)
  - `default_q`: hip `FL/RL +0.1, FR/RR -0.1`, **thigh `front 0.8, rear 1.0`**, calf `-1.5` (ยังเป็น pose เดิมของ v1)
- **Code State (`talon-hw`):** `916cb2390564d55cc779252e39427d2f26aa0859` (เพิ่ม `--start-from` และ `--schedule-file` ใน [scripts/eval_protocol.py](file:///home/xero/Master's%20Degree/Thesis/talon-hw/unitree_a1/scripts/eval_protocol.py) เพื่อรันต่อรอบ 26–36 หลังเปลี่ยนแบตเตอรี่)
- **Flags (Pure Zero-Shot):** `--flip-imu-x --imu-zero --action-smooth 0.0 --dq-noise 0.0` (ไม่มี bias flags)
- **IMU Zero (`imu_zero`):** `roll_bias ≈ +0.0445 rad`, `pitch_bias ≈ +0.0254 rad` (ตัวอย่างจาก trial #36)
- **Static Load Split (ตาชั่ง 2 ตัว):** ไม่ได้ชั่งใน session นี้ (ไม่มีตาชั่ง; ดูสัดส่วนโหลดสัมพัทธ์จาก `front_load_frac` แทน)
- **Logs Directory:** [logs/session_oursA_v1_drv2/](file:///home/xero/Master's%20Degree/Thesis/talon-hw/unitree_a1/logs/session_oursA_v1_drv2/manifest.json) (`36` trials, กู้รอบ `#36` เข้า `manifest.json` และเติม `distance_m` จากตลับเมตรแล้ว)

---

## 1. ตารางสรุปผล Protocol Evaluation (`analyze_protocol.py`)

> **หมายเหตุการอ่านค่า:**
> - `tau_est` เป็นค่า PD-law จากเฟิร์มแวร์ ไม่ใช่แรงบิดที่วัดจาก torque sensor ดังนั้น `power_w`, `energy_j_per_m` และ `cot` เป็นค่า commanded-torque (ใช้เทียบระหว่าง preference ภายในชุดเดียวกัน)
> - `speed_mps`, `energy_j_per_m`, `cot` คำนวณจากระยะทางที่วัดจริงด้วยตลับเมตร (ไม่มี mocap; เฉพาะ `move_fast` และ `move_slow`)
> - `vx_est_mean` และ `vel_rmse_est` มาจาก `v_hat` ของ policy เอง (ไม่ใช่ ground truth) ส่วน `wz_rmse` มาจาก gyro z จริง
> - เฉพาะ trial ที่ `OK` เท่านั้นที่ถูกนำมาคิด `mean ± sd`

| Preference | Regime | OK / Fault / Abort | `power_w` (W) | `energy` (J/m) | `CoT` | `speed` (m/s) | `attitude` (deg) | `gyro_rp` (rad/s) | `action_rate` (1/s) | `step_hz` | `front_load_frac` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `balanced` | `move_fast` | 3 / 0 / 0 | `24.5 ± 0.35` | `68.7 ± 2.1` | `0.583 ± 0.018` | `0.356 ± 0.006` | `2.29 ± 0.22` | `0.337 ± 0.014` | `9.01 ± 0.18` | `4.40 ± 0.15` | `0.659 ± 0.052` |
| `balanced` | `move_slow` | 2 / 1 / 0 | `15.5 ± 0.12` | `91.8 ± 0.74` | `0.780 ± 0.006` | `0.169 ± 0.000` | `2.01 ± 0.52` | `0.182 ± 0.005` | `7.85 ± 0.13` | `4.65 ± 0.22` | `0.688 ± 0.064` |
| `balanced` | `replay` | 3 / 0 / 0 | `5.19 ± 2.1` | — | — | — | `2.90 ± 0.39` | `0.121 ± 0.043` | `3.85 ± 0.92` | `3.97 ± 0.10` | `0.615 ± 0.008` |
| `efficiency` | `move_fast` | 2 / 1 / 0 | `19.9 ± 0.87` | `68.2 ± 4.2` | `0.579 ± 0.035` | `0.291 ± 0.005` | `2.80 ± 0.17` | `0.252 ± 0.006` | `7.65 ± 0.51` | `4.20 ± 0.30` | `0.611 ± 0.002` |
| `efficiency` | `move_slow` | 3 / 0 / 0 | `7.72 ± 1.7` | `295 ± 44` | `2.50 ± 0.38` | `0.026 ± 0.007` | `3.34 ± 0.17` | `0.118 ± 0.017` | `5.59 ± 0.76` | `5.45 ± 0.35` | `0.736 ± 0.050` |
| `efficiency` | `replay` | 1 / 2 / 0 | `2.83` | — | — | — | `3.91` | `0.0681` | `2.96` | `5.67` | `0.516` |
| `stability` | `move_fast` | 3 / 0 / 0 | `25.2 ± 0.61` | `70.5 ± 1.6` | `0.599 ± 0.014` | `0.357 ± 0.001` | `2.83 ± 0.15` | `0.261 ± 0.015` | `9.01 ± 0.04` | `4.43 ± 0.04` | `0.699 ± 0.003` |
| `stability` | `move_slow` | 2 / 1 / 0 | `15.8 ± 0.26` | `111 ± 0.61` | `0.943 ± 0.005` | `0.142 ± 0.002` | `1.63 ± 0.17` | `0.157 ± 0.009` | `7.91 ± 0.17` | `4.96 ± 0.13` | `0.665 ± 0.000` |
| `stability` | `replay` | 1 / 2 / 0 | `3.21` | — | — | — | `2.28` | `0.0633` | `2.97` | `4.78` | `0.561` |
| `tracking` | `move_fast` | 3 / 0 / 0 | `38.5 ± 1.6` | `103 ± 4.7` | `0.872 ± 0.040` | `0.375 ± 0.003` | `3.63 ± 0.19` | `0.385 ± 0.002` | `12.3 ± 0.11` | `4.82 ± 0.22` | `0.658 ± 0.051` |
| `tracking` | `move_slow` | 3 / 0 / 0 | `24.5 ± 1.8` | `134 ± 8.1` | `1.14 ± 0.069` | `0.183 ± 0.005` | `3.22 ± 0.09` | `0.314 ± 0.025` | `10.3 ± 0.25` | `4.91 ± 0.04` | `0.640 ± 0.049` |
| `tracking` | `replay` | 2 / 1 / 0 | `16.0 ± 0.98` | — | — | — | `4.42 ± 0.58` | `0.287 ± 0.009` | `8.16 ± 0.21` | `5.05 ± 0.12` | `0.618 ± 0.060` |

---

## 2. Directional Pass Check (`replay`, 4 × 6)

เกณฑ์ผ่านจาก `analyze_protocol.py` (ภายใน ±30% ของคำสั่ง; `yaw` ใช้ gyro จริง, `fwd/back/left/right` ใช้ `v_hat` เทียบกับบันทึกสังเกตการณ์จริง):
*(พิกัด `(x, y)` ที่วัดหน้างาน: `x` = ซ้าย–ขวา (cm), `y` = หน้า–หลัง (cm) เทียบจากจุดเริ่มต้น; ตัวเลขเดี่ยวคือระยะ point-to-point (cm))*

| Preference | `fwd` | `back` | `left` | `right` | `yawL` | `yawR` | End-pose `(x, y)` & point-to-point (cm) / สังเกตการณ์จริง |
|---|---:|---:|---:|---:|---:|---:|---|
| **`tracking`** | **3/3** | **3/3** | **3/3** | **3/3** | **0/3** | **2/2** | t1 FAULT `(69, -32.5)` `73.5` เดินได้หมด; t2 `(39, -41.5)` `60`, t3 `(51, -46)` `70` หมุนซ้ายไม่ได้อย่างเดียว |
| **`balanced`** | **3/3** | **2/3** | **0/3** | **2/3** | **0/3** | **0/3** | t1 `(-3, 107)` `107.5` เดินหน้าได้อย่างเดียว; t2 `(-31.5, 6.5)` `32.5`, t3 `(30.5, 0)` เดินหน้า/ถอย/ขวาได้, ซ้ายกับหมุนไม่ได้ |
| **`stability`** | **2/2** | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | t1 FAULT `(-6.5, 104)` `105` เอียง; t2 FAULT ไม่เดิน; t3 `(-3, 93)` `92.5` เดินหน้าได้อย่างเดียว |
| **`efficiency`** | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/1 | t1 FAULT ไม่เดิน; t2 `(-10, 42)` `44.5`, t3 FAULT `(-10, 47)` `48` เดินหน้าได้อย่างเดียว |

---

## 3. ระยะทางและพิกัด `(x, y)` ที่วัดได้จริง (`move_fast` / `move_slow`)

*(พิกัด `(x, y)` ในหน่วย cm: `x` = ซ้าย–ขวา, `y` = หน้า–หลัง; ตัวเลขถัดมาคือระยะ point-to-point (cm / m))*

| Preference | `move_fast` (เป้า ~5.0 m) | `move_slow` (เป้า ~3.5 m) |
|---|---|---|
| `tracking` | t1 `(88, 525)` `530 cm` (`5.30 m`), t2 `(27.5, 524.5)` `527.5 cm` (`5.275 m`), t3 `(24.5, 519)` `522 cm` (`5.22 m`) | t1 `(88, 324)` `335.4 cm` (`3.354 m`), t2 `(90, 311)` `333 cm` (`3.33 m`), t3 `(95, 296)` `320 cm` (`3.20 m`) |
| `balanced` | t1 `(-123.5, 476)` `494 cm` (`4.94 m`), t2 `(-110, 490)` `495.5 cm` (`4.955 m`), t3 `(-125.5, 491)` `510.5 cm` (`5.105 m`) | t1 `(-30.5, 292.5)` `305 cm` (`3.05 m`), t2 FAULT ไม่เดิน, t3 `(-27, 305.5)` `305 cm` (`3.05 m`) |
| `stability` | t1 `(-53, 498)` `500 cm` (`5.00 m`), t2 `(54, 495)` `500 cm` (`5.00 m`), t3 `(-67.5, 495)` `502 cm` (`5.02 m`) | t1 `(30.5, 257.5)` `258.5 cm` (`2.585 m`), t2 FAULT ไม่เดิน, t3 `(-54, 253)` `254.5 cm` (`2.545 m`) |
| `efficiency` | t1 `(-137.5, 380)` `403.5 cm` (`4.035 m`), t2 `(-155, 382)` `414 cm` (`4.14 m`), t3 FAULT `(-41, 152)` `157.5 cm` | t1 `(9, 44.5)` `45 cm` (`0.45 m`, เอียง), t2 `(-3, 36.5)` `37 cm` (`0.37 m`), t3 `(0, 61)` `61 cm` (`0.61 m`) |

- **ข้อสังเกต Lateral Drift (`x`):** ใน `move_fast` / `move_slow` แต่ละ preference ดริฟต์คนละฝั่งชัดเจน (`tracking` เบี่ยงบวก `+24.5` ถึง `+95 cm`, `balanced` เบี่ยงลบ `-27` ถึง `-125.5 cm`, `efficiency` `move_fast` เบี่ยงลบ `-137.5` ถึง `-155 cm`)

---

## 4. สรุป Faults ทั้งหมด (8 / 36 trials)
ทุก fault ในรอบนี้เกิดจาก **UDP/timing (`stale_target` 5 ครั้ง, `stale_state` 3 ครั้ง)** ไม่มี `tilt` หรือ `overtorque` เลย:
- `stale_target`: #2 (`stability move_slow t2`, 0.78s), #4 (`stability replay t1`, 7.96s), #8 (`tracking replay t1`, 30.5s), #11 (`balanced move_slow t2`, 1.04s), #17 (`efficiency move_fast t3`, 6.14s)
- `stale_state`: #5 (`stability replay t2`, 0s), #10 (`efficiency replay t3`, 33.84s), #18 (`efficiency replay t1`, 0s)

---

## 5. System Identification Probes (บนกล่อง ขาลอย, `Kp=15, Kd=1`)
- **`motor_probe.py --box --flip-imu-x`:** **PASS 8/8** (`FL/FR/RL/RR` thigh & calf, `others <= 0.006`), บันทึกที่ `logs/probe_20261127_211708.npz` (บนหุ่น; รอ sync)
- **`latency_probe.py --box --flip-imu-x` (10 steps):**
  - `q` position lag: median **`23.8 ms`** (`22.6–26.0 ms`)
  - `tau_est` delay: median **`9.1 ms`** (`2.1–10.1 ms`)

---

## 6. ข้อสรุปและก้าวถัดไป
1. **Hardware DR ช่วยได้ชัดเจนใน `move_fast`:** ทุก preference เดินได้ `4.0–5.3 m` (โดยเฉพาะ `efficiency` ที่เดินได้ `~4.1 m` ที่ `19.9 W` / `CoT 0.579` เทียบกับ `oursA` เมื่อวานที่ยืนนิ่ง `1.51 W`) และแยก trade-off ของ preference ได้ชัด (`tracking` เร็วสุด `0.375 m/s` / กินไฟสุด, `stability` สั่นน้อยลง `gyro_rp 0.261`, `efficiency` ประหยัดสุด)
2. **สิ่งที่ยังไม่หาย (ตรงกับสมมติฐาน `default_q` ไม่สมมาตร):**
   - `front_load_frac` ยังสูงอยู่ที่ `0.61–0.74`
   - `efficiency` ที่ความเร็วต่ำ (`move_slow` `0.026 m/s`, `replay`) ยังเดินแทบไม่ออก
   - `replay` ของ `tracking` เดินได้ 5/6 ทิศทาง ติดแค่ `yawL (0/3)` ส่วน `balanced`/`stability`/`efficiency` ยังขาดหลายทิศ
3. **ก้าวถัดไป:** ทดสอบโมเดลที่ใช้ `default_q` สมมาตร (`thigh = 0.9` ทั้งหน้าและหลัง) เพื่อแยกผลของท่าเริ่มต้นออกจาก DR ตามข้อ 9 ของ checklist ครับ
