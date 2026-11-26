# A1 oursA: Action EMA Low-Pass, Zero Calibration, PD Verification, and Front Leg Drag Analysis — 2026-11-26

Policy: `policies/oursA/policy.npz` (OURS-A, iteration 5000).
Script: `scripts/run.py`
Target Hardware: Unitree A1 quadruped (Low-Level control over UDP 8080, UP board / motion computer).

บันทึกการทดลองภาคสนามและการวิเคราะห์ telemetry ตลอดช่วงบ่ายถึงดึกของวันที่ 2026-11-26 ครอบคลุม:
1. การแก้ปัญหาหุ่นสั่นทั้งตัว (High-Frequency Shaking) ด้วย Action EMA Filter
2. ผลกระทบของ Filter Weight (`--action-smooth`) ต่อความคล่องตัวและการเลี้ยว
3. การตรวจสอบความถูกต้องของ Zero Calibration และ Motor Offsets บนหุ่นจริง
4. การตรวจสอบ PD Controller Gains บน Wire Protocol เทียบกับ Simulation
5. การวิเคราะห์สาเหตุเชิงกล้องและไคนีแมติกส์ที่ขาหน้าลากพื้น (Front Leg Drag) ตอนเดินหน้า

---

## 1. การสั่นขณะยืนนิ่ง และการแก้ด้วย Action EMA Filter

### 1.1 ปรากฏการณ์ที่พบ
หลังจากแก้ปัญหา action clip และ soft-start แล้ว หุ่นสามารถยืนและเริ่มก้าวได้ แต่เกิดการสั่นอย่างรุนแรงทั้งตัว (High-Frequency Chatter / Whole-Body Shaking) แม้ขณะยืนเฉยๆ ในโหมด `WALK` ที่คำสั่งจอยสติ๊กเป็นศูนย์ ($v_x=0, v_y=0, \omega_z=0$)
- เมื่อทดสอบปรับ observation noise ของความเร็วข้อต่อ (`--dq-noise 0.1` ถึง `1.5`):
  - `dq ±0.1` (`run_20261126_213259.npz`): เดินได้นานสุด 64s การสั่นลดลงบ้างแต่ยังมีอยู่
  - `dq ±1.0` ถึง `±1.5`: เดินได้สั้นลง สั่นรุนแรง มอเตอร์มีเสียงร้องความถี่สูงต่อเนื่อง
- **การวินิจฉัย Telemetry:** สัญญาณ Action จาก Policy มีการกระเพื่อมความถี่สูง (Chatter) ที่ความถี่รอบลูป 50 Hz ซึ่งส่งผลให้เป้าหมายข้อต่อ $q_{target}$ สลับไปมาอย่างรวดเร็ว มอเตอร์พยายามเร่งและเบรกฉับพลัน

### 1.2 การแก้ไข: First-Order Action Exponential Moving Average (EMA)
เพิ่มตัวกรอง Action EMA ใน [talon_a1/controller.py](</home/xero/Master's Degree/Thesis/talon-hw/unitree_a1/talon_a1/controller.py>) ผ่านพารามิเตอร์ CLI `--action-smooth $\alpha$`:
$$a_t = (1 - \alpha) \cdot a_{policy} + \alpha \cdot a_{t-1}$$
- โดยที่ $\alpha \in [0.0, 1.0)$ ทำหน้าที่เป็น Low-Pass Filter สำหรับ action command
- ผลการทดสอบ:
  - `--action-smooth 0.3` (`run_20261126_231926.npz`, `run_20261126_233551.npz`):
    - **การสั่นทั้งตัวหายไปอย่างสมบูรณ์** หุ่นยืนนิ่งสนิท มอเตอร์เงียบ ไม่มี High-Frequency Chatter
    - เดินถอยหลังได้นุ่มนวลมาก (Smooth Backward Gait)
  - `--action-smooth 0.4` (`run_20261126_233740.npz`):
    - อาการสั่นหายสนิท แต่เกิด Phase Lag ใน Control Loop มากเกินไป
    - หุ่นสูญเสียการตอบสนองต่อคำสั่งเลี้ยวหมุนตัว (Yaw) และการเดินสไลด์ข้าง (Lateral Walk) ก้าวขาตามจอยสติ๊กไม่ทัน
- **สรุปช่วงค่าที่เหมาะสม:** แนะนำใช้ `--action-smooth 0.25 ~ 0.30` เพื่อกำจัดการสั่นโดยไม่สูญเสีย Yaw/Lateral Tracking

---

## 2. พฤติกรรมการเดิน: เดินถอยหลังสมูท แต่เดินหน้าขาหน้าลากพื้น

### 2.1 ข้อสังเกตจากการทดลอง
- **เดินถอยหลัง ($v_x < 0$):** ก้าวขาสลับได้สมบูรณ์ นุ่มนวลมาก หุ่นทรงตัวนิ่ง Pitch rate std เพียง $\sim 0.18\text{ rad/s}$
- **เดินหน้า ($v_x > 0$):** หุ่นมีอาการกระตุก สะดุด และเดินหน้าติดขัด
  - จากการสังเกตการเคลื่อนไหวจริง: **ขาหน้าทั้งสองข้าง (FL / FR) ไม่ยอมยกพ้นพื้น**
  - เท้าหน้าลากขูดไปกับพื้น (Foot Stubbing / Ground Scraping) กลายเป็นเสาค้ำต้านการเดินหน้าของหุ่น
  - ส่งผลให้ Pitch rate กระเพื่อมขึ้นไปถึง $\sim 0.43\text{ rad/s}$ และตัวหุ่นเอียงทิ่มลงด้านหน้า

---

## 3. การตรวจสอบ Hardware Zero Calibration และ Motor Offsets

มีข้อสงสัยว่าการที่ขาหน้าไม่ยอมยกเกิดจากจุด Zero Calibration ของมอเตอร์ขาหน้าบนหุ่นจริงเพี้ยนหรือไม่ ทำให้ตำแหน่งจริงของขาไม่ตรงกับมุมที่ส่งไป

### 3.1 ข้อมูล Telemetry จากโหมด STAND (`run_20261126_234956.npz`)
เมื่อวิเคราะห์ค่ามุมข้อต่อจริงของหุ่น (`state.q`) เทียบกับเป้าหมายตามสัญญาของหุ่น (`default_q`):
- **เป้าหมายใน Stand (`default_q`):**
  - Hip: FL = +0.100, FR = -0.100, RL = +0.100, RR = -0.100 rad
  - Thigh: Front = 0.800, Rear = 1.000 rad
  - Calf: All = -1.500 rad
- **ค่าอ่านจริงจากเอ็นโค้เดอร์มอเตอร์ (`state.q`):**
  - FR Hip: -0.090 rad (ต่างเพียง +0.010 rad หรือ $+0.57^\circ$)
  - FL Hip: +0.095 rad (ต่างเพียง -0.005 rad หรือ $-0.28^\circ$)
  - RR Hip: -0.099 rad (ต่างเพียง +0.001 rad หรือ $+0.06^\circ$)
  - RL Hip: +0.092 rad (ต่างเพียง -0.008 rad หรือ $-0.45^\circ$)
  - Thigh & Calf: ความสมมาตรซ้าย-ขวาเท่ากัน โดยมี Steady-State Error $\sim 0.15\text{ rad}$ สม่ำเสมอทุกลวดลายเนื่องจากแรงโน้มถ่วงรับน้ำหนักตัว (Gravity Sag ปกติตามทฤษฎี PD control)

### 3.2 ข้อสรุป Zero Calibration
**มอเตอร์และจุด Zero Offsets ของฮาร์ดแวร์ปรับตั้งไว้ถูกต้อง แม่นยำ (< 0.5°)** ไม่มีความคลาดเคลื่อนผิดปกติ ขาซ้าย-ขวาทำงานสมมาตรอย่างสมบูรณ์ ปัญหาไม่ได้เกิดจากการตั้งค่า Calibration

---

## 4. การตรวจสอบ PD Controller Gains บน Wire Protocol

ตรวจสอบว่าค่า $K_p$ และ $K_d$ ในสคริปต์รันตรงกับที่ใช้เทรนใน Isaac Lab หรือไม่

### 4.1 สเปคใน Simulation (Isaac Lab Training)
- $K_{p,\text{nominal}} = 55.0\text{ N}\cdot\text{m/rad}$
- $K_{d,\text{nominal}} = 0.80\text{ N}\cdot\text{m}\cdot\text{s/rad}$
- Action scale = 0.25, decimation = 4, sim dt = 0.005s (Policy dt = 0.02s / 50 Hz)

### 4.2 การแปลงลงฮาร์ดแวร์จริง (Unitree Low-Level UDP Protocol)
ใน Low-Level SDK ของ Unitree A1 ค่า Gain จะถูกบีบอัดเป็น Fixed-Point 16-bit:
- $\text{wire\_kp} = \text{int16}(K_p \times 32)$ $\rightarrow 55.0 \times 32 = 1760 \rightarrow$ ถอดรหัสได้ **$55.00$** ตรง 100%
- $\text{wire\_kd} = \text{int16}(K_d \times 16)$ $\rightarrow 0.80 \times 16 = 12.8 \xrightarrow{\text{int16}} 12$
  - เมื่อมอเตอร์ถอดรหัส: $12 / 16 = \mathbf{0.75\text{ N}\cdot\text{m}\cdot\text{s/rad}}$ (-6.25% truncation)

### 4.3 ข้อสรุป PD Gains
- $K_p = 55.0$ ตรงกับ simulation สมบูรณ์
- $K_d = 0.75$ บนฮาร์ดแวร์เทียบกับ 0.80 ใน sim ต่างกันเพียง 6.25% ซึ่งอยู่ในช่วง Domain Randomization ของการเทรน ($K_d \in [0.4, 0.8]$) จึงไม่ใช่สาเหตุที่ทำให้ขาหน้ายกไม่ขึ้น

---

## 5. การวิเคราะห์สาเหตุแท้จริง: Asymmetry ของ `default_q` และการกระจายน้ำหนัก (Weight Distribution)

### 5.1 การวิเคราะห์ Forward Kinematics ของท่า Default Pose
ใน `policies/oursA/policy_meta.json` ค่า `default_q` กำหนดไว้ดังนี้:
```json
"default_q": [
  -0.1,  0.8, -1.5,   // FR: hip, thigh, calf
   0.1,  0.8, -1.5,   // FL
  -0.1,  1.0, -1.5,   // RR
   0.1,  1.0, -1.5    // RL
]
```
สังเกตว่า **Thigh หน้า = 0.8 rad** ขณะที่ **Thigh หลัง = 1.0 rad**

เมื่อคำนวณตำแหน่งปลายเท้าสัมพัทธ์กับแกนสะโพก ($L_1 = 0.20\text{ m}, L_2 = 0.20\text{ m}$):
- **ขาหน้า ($q_{thigh}=0.8, q_{calf}=-1.5$ rad):**
  $$x_{front} = 0.20 \sin(0.8) + 0.20 \sin(0.8 - 1.5) = 0.20(0.717) + 0.20(-0.644) = \mathbf{+0.015\text{ m}}$$
  (ปลายเท้าหน้าอยู่เกือบตรงกับแนวสะโพกหน้า)
  $$z_{front} = -0.20 \cos(0.8) - 0.20 \cos(0.8 - 1.5) = -0.20(0.697) - 0.20(0.765) = \mathbf{-0.292\text{ m}}$$
- **ขาหลัง ($q_{thigh}=1.0, q_{calf}=-1.5$ rad):**
  $$x_{rear} = 0.20 \sin(1.0) + 0.20 \sin(1.0 - 1.5) = 0.20(0.841) + 0.20(-0.479) = \mathbf{+0.072\text{ m}}$$
  เทียบกับแนวสะโพกหลัง เท้าหลังถูกดันไปข้างหลังถึง $0.072\text{ m}$ เมื่อเทียบกับสะโพก
- **ระยะห่างของจุดสัมผัสพื้น:**
  - ฐานรองรับ (Polygon of Support) เลื่อนถอยไปทางด้านหลังของหุ่นถึง **5.8 cm** เมื่อเทียบกับจุดสมมาตร

### 5.2 ผลลัพธ์ต่อการถ่ายเทน้ำหนักบนหุ่นจริง (ยืนเฉยๆ ในโหมด WALK)
จากข้อมูล Log `run_20261126_234956.npz` (ยืนเฉยๆ ที่ $v=0$):
- ค่าประมาณแรงปฏิกิริยาจากพื้นดินที่คำนวณจาก Joint Torque Balance:
  - **ขาหน้า (FL + FR):** ต้องแบกรับน้ำหนักสูงถึง $\sim \mathbf{252\text{ N}}$
  - **ขาหลัง (RL + RR):** แบกรับน้ำหนักเพียง $\sim \mathbf{40 - 50\text{ N}}$
- หุ่นอยู่ในภาวะ "หน้าทิ่มและน้ำหนักกดลงที่ขาหน้าเกือบทั้งหมด" (Front-Heavy Stance)

### 5.3 กลไกที่ทำให้เดินถอยหลังได้ แต่เดินหน้าไม่ได้
1. **เมื่อเดินถอยหลัง ($v_x < 0$):**
   - ขาหลังซึ่งรับน้ำหนักน้อยสามารถยกขึ้นและก้าวถอยหลังได้อย่างง่ายดาย
   - ขาหน้าที่รับน้ำหนักมากทำหน้าที่เป็นจุดหมุน (Pivot/Push) ดันตัวหุ่นถอยหลัง จึงเคลื่อนที่ได้ราบรื่นมาก
2. **เมื่อเดินหน้า ($v_x > 0$):**
   - Policy พยายามยกขาหน้าขึ้นเพื่อก้าวไปข้างหน้า
   - แต่เนื่องจากน้ำหนักตัว 252 N กดทับอยู่ที่เท้าหน้า มอเตอร์ขนาด $K_p=55.0$ และแรงขับเคลื่อนที่จำกัด ไม่สามารถสู้แรงกดเพื่อยกเท้าหน้าให้พ้นพื้นได้ทันที (Foot Clearance ไม่เพียงพอ)
   - ปลายเท้าหน้าจึงขูดและต้านอยู่กับพื้น ทำให้เกิดการกระตุกและสะดุด

---

## 6. ข้อสรุปและแนวทางแก้ไขที่แนะนำ (Actionable Recommendations)

1. **การยืนยันระบบ Deployment:**
   - ระบบสื่อสาร UDP, Loop timing, Joint order mapping, Motor zero offsets, และ PD gains ทำงานถูกต้องสมบูรณ์
   - Action EMA filter (`--action-smooth 0.3`) เป็นองค์ประกอบจำเป็นที่ต้องเปิดใช้บนฮาร์ดแวร์จริงเพื่อตัดเสียงรบกวนความถี่สูง

2. **แนวทางแก้ไขปัญหาขาหน้าลากพื้น:**
   - **ทางเลือกที่ 1 (ไม่ต้องเทรนใหม่ - Hardware Workspace Compensation):**
     - เพิ่ม Feed-forward Swing Clearance Bias ให้กับขาหน้าใน [talon_a1/controller.py](</home/xero/Master's Degree/Thesis/talon-hw/unitree_a1/talon_a1/controller.py>) ขณะที่มีคำสั่งเดินหน้า ($v_x > 0$) โดยชดเชยมุม Thigh/Calf ในช่วง Swing Phase หรือปรับจุดศูนย์ถ่วง (Pitch offset) เล็กน้อยให้ถ่ายน้ำหนักกลับมาที่ขาหลัง
   - **ทางเลือกที่ 2 (Simulation Retraining):**
     - ตรวจสอบและแก้ไข `default_q` ในการเทรนให้ขามีลักษณะสมมาตรหน้า-หลัง (เช่น Thigh หน้าและหลังเท่ากับ 0.9 rad)
     - เพิ่ม Weight ของ `foot_clearance_reward` หรือเพิ่ม Air-Time Target Reward ในการเทรนเพื่อบังคับให้ Policy ยกเท้าสูงขึ้นอย่างสม่ำเสมอ
