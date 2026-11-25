# A1 oursA: เดินครั้งแรก, สาเหตุที่ไม่ก้าว และ latency — 2026-11-26

Policy: `policies/oursA/policy.npz` (OURS-A, `final.pt` iteration 5000, sha256 `13cedd9a…`). รันด้วย `scripts/run.py`, `--soft-start 2.0`, Kp 55 / Kd 0.8, action scale 0.25. ทุกข้อด้านล่างมาจาก log บนหุ่นและ MuJoCo ของวันนี้ ยกเว้นที่ระบุว่ายังไม่ยืนยัน

## 1. หุ่นเอียงตามคันโยกแต่ไม่ก้าว (`run_20261126_003951.npz`)

- **สาเหตุ:** `run.py` clip action ที่ ±1 (`--action-clip` default 1.0) แต่ PROMO actor มี output เป็น linear ไม่มี tanh และต้องใช้ |a| > 1 ตอนก้าว action ที่ clip แล้วถูกป้อนกลับเป็น `last_action` ด้วย policy จึงไม่เห็นสิ่งที่สั่งจริง ใน log action ของน่องชน 1.0 พอดี
- **หลักฐาน:** `scripts/promo/verify_onnx.py` (talon-rl) กับ `final.pt` จริงเดินได้ 5.03 m ใน MuJoCo (ไม่ clip) ใน harness เดียวกันถ้า clip ที่ 1 ได้ 0.015 m/s, 1.5 ได้ 0.37, 2 ขึ้นไปได้ 0.50 m/s
- **ที่ตัดออกแล้ว:** replay obs ใน log ผ่าน policy ได้ action ตรงกับที่หุ่นสั่ง (error ~1e-6), npz ตรง onnx และ `reference_io.npz`, torque จริงตรง 55·err − 0.8·dq (slope ~0.95), คณิต IMU flip ถูก, joint order และ obs layout ตรงกับ `export_onnx.py`
- **แก้:** `run.py` default `--action-clip` เป็น 10 สำหรับ policy PROMO และ 1 สำหรับ RMA (commit `c4aa6e8`) หลังแก้หุ่นเดินได้ (`run_20261126_005955.npz`)

## 2. หลังแก้ clip: เด้ง และบางทิศไม่ได้

บน balanced w: เดินหน้าได้แต่เด้ง (ก้าว 6.5–7.6 Hz, gyro std 0.4–0.5 rad/s; sim ที่ delay 0 ได้ 4.7 Hz), ถอยหลังเรียบ, ซ้ายบางครั้ง, ขวาไม่ได้, หมุนบางครั้ง

- **ไม่ใช่สาเหตุ (ทดสอบใน sim):** IMU roll bias ±0.067 rad, noise ของ dq/gyro, Kd 0.75
- **Latency:** `scripts/latency_probe.py --box --flip-imu-x` วัดได้ `tau_est` ตอบหลังสั่ง median 9.0 ms (5.1–10.2) ส่วน `q` ขยับหลัง 24.5 ms ซึ่งรวมเวลาเร่งของขาที่ Kp 15 ไม่ใช่ delay ล้วน
- **Sim + delay (MuJoCo, ไม่ clip):** delay 5–15 ms ทำให้ขวา/หมุนซ้ายเสียและเด้งเพิ่ม แต่ผลรายทิศไม่นิ่ง (หมุนซ้ายพังที่ 5 ms ผ่านที่ 10 ms) ตรงกับอาการบนหุ่น ยังไม่ถือว่ายืนยันว่า delay เป็นต้นเหตุเดียว เพราะหุ่นจริงยังก้าวถี่กว่า sim ที่ delay 15 ms (5.2 Hz)
- **เปเปอร์ PROMO** (arXiv 2610.01260) ใช้ Go2 และไม่ได้สุ่มหรือรายงาน latency (Table B.2 สุ่มแค่ friction, มวลฐาน, แรง/ทอร์กรบกวน, push)

## 3. ปรับ w ตอน deploy โดยไม่เทรนใหม่

ใน sim (delay 10 และ 15 ms, 3 seed, ผ่านเมื่อความเร็วจริงใกล้คำสั่ง 6 ทิศ):

| w | delay 10 ms | delay 15 ms |
|---|---|---|
| balanced | ผ่าน 4/6 (ขวา, หมุนซ้ายไม่ผ่าน) | ผ่าน 5/6 |
| 0.8/0.1/0.1 | ผ่านครบทุก seed | ผ่านครบทุก seed |
| 0.6/0.2/0.2 | ผ่าน 17/18 | ผ่านครบ |

บนหุ่น (`run_20261126_011854.npz`, `--w 0.8 0.1 0.1`): เดินต่อเนื่อง 146 s ครบทุกทิศ (ขวาและหมุนใช้ได้แล้ว) ไม่ล้ม จบด้วย fault overtorque (limit 30 Nm; log 50 Hz เห็นสูงสุด 23.7 Nm แต่ driver ตรวจที่ 500 Hz) ข้อเสีย: เด้งมากกว่า sim ที่ w เดียวกันราว 1.3 เท่า (เดินหน้า 7.8 Hz, gyro std 0.62/0.65 เทียบ 6.9 Hz, 0.49) และ hip เบี่ยงออกข้างราว 0.11 rad ซึ่งเป็นท่าเดินของ policy ที่ w นี้ (sim ได้ 0.07) ลดความเร็วช่วยเรื่องเด้งไม่ได้ใน sim

## 4. IMU

IMU ผิดทิศ (roll ใกล้ ±π) ต้องใช้ `--flip-imu-x` ในทุกรอบที่ล็อกไว้วันนี้ ผู้ทดสอบแจ้งว่าแก้ที่ IMU แล้ว แต่รอบหลังยังใส่ flag นี้ ควรตรวจ roll/pitch ตอนยืนด้วย `inspect_sensors.py` ก่อนตัดสินใจเอาออก

## ยังไม่ทำ / ยังไม่รู้

- ยังไม่รู้ว่าทำไมหุ่นเด้งมากกว่า sim ที่ใส่ delay แล้ว ยังไม่ได้ตรวจ motor model ตอนเทรน (Isaac DC motor) เทียบ MuJoCo (clip คงที่ 33.5 Nm) และยังไม่ได้ตรวจว่า OURS-A เทรนด้วย observation noise หรือไม่
- ข้อเสนอ: เทรน OURS-A ใหม่พร้อมสุ่ม action delay 0–20 ms แล้วทดสอบใน MuJoCo ที่ delay 0/10/20 ms ก่อนลงหุ่น คงชุด baseline เดิมไว้เทียบใน sim
- ใน sim ที่ delay 0 BASE-v2 เดินได้แค่ไปข้างหน้า (ถอยหลัง ข้าง และหมุนไม่ได้) จึงไม่ใช่ทางเลือกแทน OURS-A
- `policies/` ถูก ignore จาก git แล้ว ไฟล์ `final.pt` ไม่มี backup ใน git

## คำสั่งรอบล่าสุด

```
sudo python3 scripts/run.py --policy policies/oursA/policy.npz --soft-start 2.0 --vx-max 0.5 --vy-max 0.3 --wz-max 0.5 --w 0.7 0.15 0.15 --tau-limit 33 --flip-imu-x
```

(`--w 0.7 0.15 0.15` และ `--tau-limit 33` ยังไม่ได้ทดสอบบนหุ่น)
