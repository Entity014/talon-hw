# สรุปงาน Unitree A1 — Motion Computer Audit

สถานะ ณ 2 พฤศจิกายน 2026

ตอนนี้เราตรวจสอบ Motion Computer ของ Unitree A1 ไปค่อนข้างมากแล้วครับ ทั้ง Hardware, Controller, SDK, Network Communication และ Sensor Feedback

ผลสำคัญที่สุดคือ เราสามารถรับข้อมูลจาก Main MCU และอ่าน IMU ได้แล้ว แต่ยังอ่าน Joint Position และ Joint Velocity ทั้ง 12 ข้อต่อไม่ได้อย่างถูกต้อง

## 1. Hardware ที่ตรวจสอบแล้ว

Unitree A1 — Motion Computer — **PASS**

| รายการ      | ค่า                           |
| ----------- | ----------------------------- |
| IP Address  | `192.168.123.161`             |
| SSH User    | `unitree`                     |
| CPU         | Intel Atom x5-Z8350 (4 cores) |
| RAM         | 3.3 GB                        |
| Storage     | 28 GB, ว่างประมาณ 16 GB       |
| OS          | Ubuntu 16.04.6 LTS            |
| Kernel      | `4.4.86-rt99` (Real-Time)     |
| Unitree SDK | v3.3                          |

เครื่องนี้รองรับการพัฒนาโปรแกรม C++ และมี SDK สำหรับสื่อสารกับ Main MCU ติดตั้งอยู่แล้ว

## 2. Controller และ Startup System

เราพบว่า Controller ที่กำลังทำงานคือ

```
/home/unitree/Unitree/
└── keep_program_alive/
    ├── start.sh
    ├── keep_program_alive.sh
    └── bin/
        ├── A1_sport_1
        └── A1_sport_2
```

จาก Source Script ที่ตรวจจริง:

- `start.sh` เรียก `ipconfig.sh` เพื่อตั้งค่า Network ก่อนเริ่ม Supervisor
- `keep_program_alive.sh` ตรวจ Process ทุก 0.2 วินาที
- หาก Sport Controller หยุดทำงาน Supervisor อาจสลับไปเปิดอีกตัวหนึ่ง
- `A1_sport_1` เป็นโปรแกรมที่สื่อสารกับ Main MCU อยู่ในปัจจุบัน

สถานะ: PASS สำหรับการระบุโครงสร้างและ Watchdog แต่ยังไม่ได้ทดสอบ Controller Handoff

> **ข้อสำคัญ:** เราไม่ควร Kill `A1_sport_1` โดยไม่จัดการ Watchdog เพราะอาจทำให้ `A1_sport_2` เริ่มทำงานแทน

## 3. Communication ระหว่าง Motion Computer กับ Main MCU

เราใช้ Passive Packet Capture ตรวจสอบการสื่อสารจริง ได้ผลดังนี้

| ตัวชี้วัด     | ค่า                     |
| ------------- | ----------------------- |
| UDP Frequency | ~1 kHz                  |
| UDP Payload   | 820 B                   |
| Captured      | 4,000 Packets / ~4 s    |

เส้นทางที่ตรวจพบ:

```
Main MCU                         Motion Computer
192.168.123.10                   192.168.123.161
UDP 8007 ─── State packets ───→ UDP 8008
                                      │
                                      ▼
                                  A1_sport_1
```

นี่เป็นหลักฐานว่า MCU ส่งข้อมูลเข้า Motion Computer อย่างต่อเนื่อง แต่ยังไม่ใช่การยืนยันว่า Sensor ทุกตัวอัปเดตด้วยความถี่ 1 kHz

## 4. Sensor Feedback ที่อ่านได้แล้ว

### IMU — PASS เบื้องต้น

เรา Decode ค่า IMU จาก Packet จริงได้ 3 ตัวอย่าง โดยได้ค่าประมาณ:

| Sensor          | ค่าที่อ่านได้ |
| --------------- | ------------- |
| Quaternion Norm | 1.0000        |
| Roll            | 0.0059 rad    |
| Pitch           | −0.0113 rad   |
| Yaw             | 0.2123 rad    |
| Gyroscope       | ใกล้ 0 rad/s  |

ผลสอดคล้องกับหุ่นที่อยู่นิ่ง และยืนยันความเป็นไปได้สูงของ IMU Layout แต่ยังไม่ได้ตรวจความถูกต้องระหว่างการเคลื่อนไหว

### Joint State — ยังไม่ผ่าน

เราได้ตรวจ SDK Struct จริง:

| Structure       | Size      |
| --------------- | --------- |
| `IMU`           | 53 bytes  |
| `MotorState`    | 38 bytes  |
| `LowState`      | 891 bytes |
| `HighState`     | 265 bytes |
| Raw MCU Payload | 820 bytes |

เมื่อทดลองอ่าน Joint State ด้วย Offset ที่สมมติจาก `LowState` พบค่าผิดปกติ เช่น `dq = -542763057152 rad/s`

จึงตัดสินว่าการ Decode Joint State แบบนั้น FAIL และยังไม่ควรนำไปใช้งาน

นี่ไม่ได้พิสูจน์ว่า Encoder เสีย แต่แสดงว่ารูปแบบ Packet ยังไม่ตรงกับที่เราสมมติ

## 5. ช่องทางอื่นที่ตรวจแล้ว

| ช่องทาง                                  | ผล                           |
| ---------------------------------------- | ---------------------------- |
| ROS `/joint_states`                      | ไม่พบ Publisher              |
| ROS Runtime บน Motion Computer           | ไม่พบ                        |
| LCM References ใน `A1_sport_1`           | พบใน Binary                  |
| LCM Short Messages (`LC02`)              | ไม่พบในช่วง Capture 5 วินาที |
| SDK `LowState`                           | มีข้อมูล Motor ครบตาม Struct |
| Official Read-only Joint State Interface | ยังไม่ยืนยัน                 |

การตรวจ LCM ยังไม่ครอบคลุมทุก Message Format จึงไม่สรุปว่าไม่มี LCM ในระบบทั้งหมด

## 6. สิ่งที่เราตกลงกันเกี่ยวกับ RL Controller

คุณระบุว่า RL Policy ของเราใช้ Joint Position Control

ดังนั้น Architecture ที่วางไว้สำหรับอนาคตคือ:

```
IMU + Joint State
         │
         ▼
   RL Observation
         │
         ▼
      RL Policy
         │
         ▼
  12 Joint Targets
         │
         ▼
 Low-Level Position PD
         │
         ▼
       Main MCU
```

SDK v3.3 มี `LowCmd.motorCmd[i]` ที่ประกอบด้วย `q`, `dq`, `Kp`, `Kd` และ `tau` สำหรับการควบคุมระดับข้อต่อ

สำหรับ Full Low-Level Control ในอนาคต เราสามารถพัฒนา Controller และ Supervisor ของเราเองโดยไม่ต้องพึ่ง `keep_program_alive.sh` เดิม แต่ต้องเตรียมระบบ Safety และ Handoff ให้ผ่านก่อน

เรายังไม่ได้ Deploy RL, เปลี่ยน Startup, หยุด Sport Controller หรือส่ง Motor Command จากโปรแกรมของเรา

## 7. สถานะ QA รวม

| Gate                                 | Verdict        |
| ------------------------------------ | -------------- |
| M0 — Hardware Identification         | PASS           |
| M1 — Controller / SDK Identification | PASS           |
| M1 — Safe Controller Handoff         | PENDING        |
| M2 — UDP Communication               | PASS           |
| M2 — IMU Decoding                    | PASS เบื้องต้น |
| M2 — Joint Position / Velocity       | NOT PASSED     |
| M3 — RL Policy Compatibility         | NOT STARTED    |
| M4 — CPU Inference Benchmark         | NOT STARTED    |
| M5 — Hardware Deployment             | NOT STARTED    |

## 8. แล้วควรทำอะไรต่อ?

ผมแนะนำให้ ทำ M2 ให้จบก่อน ไม่ต้องแตะ RL ในตอนนี้ ตามเป้าหมายที่คุณต้องการ

งานที่เหลือคือ:

1. ตรวจช่องทางอ่าน `LowState` ของ Unitree SDK v3.3 และรูปแบบการสื่อสารที่ถูกต้อง
2. เตรียม Controlled Handoff โดยจัดการ Watchdog และรองรับน้ำหนักหุ่นอย่างปลอดภัย หากจำเป็นต้องหยุด Sport Controller
3. เขียนโปรแกรม C++ สำหรับอ่าน Joint State โดยไม่ส่ง Motor Command หาก Protocol รองรับ
4. ยืนยัน `q`, `dq` และ Joint Index Mapping ของทั้ง 12 ข้อต่อ

เมื่อทำสำเร็จ เราควรได้ผลลักษณะนี้:

```
Unitree A1 — Joint State Monitor

Joint      Position(rad)   Velocity(rad/s)
FR_0           ...               ...
FR_1           ...               ...
FR_2           ...               ...
FL_0           ...               ...
FL_1           ...               ...
FL_2           ...               ...
RR_0           ...               ...
RR_1           ...               ...
RR_2           ...               ...
RL_0           ...               ...
RL_1           ...               ...
RL_2           ...               ...
```

**สรุป:** Motion Computer, Controller Lifecycle, SDK Interface, Network และ IMU ได้รับการตรวจสอบแล้วในระดับที่กำหนด ส่วนงานเดียวที่เราควรโฟกัสต่อคือ การอ่าน Joint State ทั้ง 12 ตัวให้ถูกต้องและปลอดภัย ก่อนขยับไปงานอื่นครับ
