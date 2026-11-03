# Unitree A1 — Perception Feasibility Summary

31 ตุลาคม 2026 · Hardware & Software Audit · Depth Camera Benchmark

**Verdict: Feasibility PASS** — Raspberry Pi 4 + Intel RealSense D435i

| Native Depth | Missing frame numbers | CPU Usage | Obstacle ROI Processing |
|---|---|---|---|
| 29.99 FPS | 0 | ~117% (เทียบเท่า 1.17 คอร์) | 0.10 ms เฉลี่ยต่อเฟรม |

ข้อสรุป: Hardware เดิมเพียงพอสำหรับการอ่าน Metric Depth และตรวจระยะสิ่งกีดขวางแบบง่ายในสภาวะที่ทดสอบ ไม่จำเป็นต้องเปลี่ยนเป็น Jetson สำหรับงานระดับนี้

## 1. Hardware และ Network ที่ตรวจสอบ

เชื่อมต่อ Laptop กับ Unitree A1 ผ่าน LAN และเข้าถึงคอมพิวเตอร์ภายในหุ่นได้ทั้งสองเครื่อง

| Component | IP | รายละเอียด |
|---|---|---|
| Motion Computer | `192.168.123.161` | Ubuntu 16.04, x86_64, RT Kernel 4.4.86-rt99 |
| Perception Computer | `192.168.123.12` | Raspberry Pi 4 Model B Rev 1.4, RAM 4 GB |
| Depth Camera | USB 3.0 | Intel RealSense D435i |

ผลตรวจเครือข่าย:

- Ping จาก Laptop ไปทั้งสองเครื่องผ่าน ไม่มี Packet Loss ในชุดทดสอบ
- SSH เข้า Motion Computer ผ่านบัญชี `unitree` ได้
- SSH เข้า Raspberry Pi ผ่านบัญชี `pi` ได้
- กล้อง D435i เชื่อมต่อด้วย USB SuperSpeed `5000M`
- Raspberry Pi ใช้ Raspbian 10 Buster, ARMv7

Network Connectivity: PASS

## 2. Software เดิมของ Unitree

ไม่ได้ติดตั้งระบบใหม่ ตรวจเฉพาะสิ่งที่มีอยู่ก่อนแล้ว

```
/home/pi/
├── RobotVisionSystem/
├── RobotSLAMSystem/
├── catkin_ws/
└── librealsense-2.31.0/
```

| Software | ผล |
|---|---|
| ROS Melodic | Installed |
| librealsense 2.31.0 | Installed |
| OpenCV 3.2 | มี Library ที่โปรแกรมใช้งาน |
| Unitree Legged SDK | มี Library |
| RobotVisionSystem | ทำงานจริง |
| RobotSLAMSystem | พบไฟล์ ยังไม่ยืนยันการทำงาน |

`RobotVisionSystem` มีโค้ดเกี่ยวกับ 4 โหมด: `Image Show`, `Depth Perception`, `Person Detection`, `Object Tracking`

พบ UDP Socket จาก Raspberry Pi ไปยัง Motion Computer (`8090 → 8082`) แต่การตรวจ Traffic 6 วินาทีไม่พบ Packet ที่ตรงเงื่อนไข จึงยังสรุปไม่ได้ว่าระบบเดิมกำลังส่ง Motion Command อยู่หรือไม่

## 3. RGB Camera Streaming

HTTP Server ของ Raspberry Pi: `http://192.168.123.12:8080/` ทดสอบรับภาพ MJPEG จริง

| Metric | ผล |
|---|---|
| Requested FPS | 30 |
| Frames Received | 296 |
| Test Duration | 10 วินาที |
| Measured FPS | 29.59 |
| RobotVisionSystem CPU | ~100% ของหนึ่งคอร์ |
| RobotVisionSystem RAM | ~105 MiB |

RGB Streaming: PASS

ข้อจำกัด: HTTP Endpoint ที่ตรวจส่งภาพ RGB ได้ แต่ยังไม่พบ API ที่ยืนยันว่าเปิดเผย Metric Depth ออกมาโดยตรง

## 4. Native Metric Depth

`RobotVisionSystem` ครอบครองกล้องอยู่ ทำให้เปิด Depth Pipeline อีกตัวไม่สำเร็จ หลังปิดโปรแกรมเดิมอย่างสุภาพเพื่อปล่อย Camera Ownership จึงทดสอบผ่าน C++ และ RealSense SDK ได้

| Metric | 15 FPS Test | 30 FPS Test |
|---|---|---|
| Resolution | 640×480 | 640×480 |
| Frames | 150 | 300 |
| Actual FPS | 15.00 | 29.99 |
| Valid Depth | 80.69% | 80.71% |
| Center Distance | 2.86 m | 2.82 m |
| Exit Code | 0 | 0 |

Depth Scale ที่กล้องรายงาน: `0.001 m/unit`

Native Metric Depth: PASS

CPU ขณะอ่าน Depth ที่ 30 FPS ประมาณ 113–117% (หน่วย CPU ของ Linux) RAM ของ Process ประมาณ 21 MiB โดยยังรักษา 29.99 FPS ได้

## 5. Obstacle Detection — กล่อง 3 ตำแหน่ง

วางกล่องสามใบที่ระยะต่างกัน ใช้ Depth ROI และค่า P10 ประเมินระยะด้านซ้าย กลาง ขวา

| ตำแหน่ง | ระยะที่จัดไว้ | วัดได้โดยประมาณ | Absolute Error |
|---|---|---|---|
| LEFT | 1.50 m | 1.49 m | 0.01 m |
| CENTER | 1.00 m | 0.99 m | 0.01 m |
| RIGHT | 0.50 m | 0.54 m | 0.04 m |

ประสิทธิภาพของโปรแกรม Obstacle Probe:

- FPS: 29.99
- Missing frame numbers: 0
- Average ROI processing: 0.10 ms
- Center ROI valid depth: ประมาณ 64–73%
- Exit Code: `0`

Static Obstacle Proximity: PASS

ข้อจำกัด: ระยะที่จัดไว้เป็นค่าอ้างอิงเบื้องต้น ไม่ใช่การสอบเทียบด้วยเครื่องมือวัดมาตรฐาน และโปรแกรมยังไม่ได้แยกพื้นออกจากสิ่งกีดขวางอย่างสมบูรณ์

## 6. สถานะตาม QA Gates

| Gate | วัตถุประสงค์ | Verdict |
|---|---|---|
| N0 | LAN / SSH Connectivity | PASS |
| H0 | Hardware & Software Audit | PASS |
| V0 | RGB Streaming | PASS |
| D0 | Camera Ownership Release | PASS |
| D1 | Metric Depth @ 15 FPS | PASS |
| D2 | Metric Depth @ 30 FPS | PASS |
| D3 | CPU / RAM Feasibility | PASS เบื้องต้น |
| D4 | Static Obstacle Proximity | PASS |
| D5 | Robust Obstacle Detection | NOT TESTED |
| D6 | Floor Segmentation / Local Map | NOT TESTED |
| D7 | Navigation Integration | NOT TESTED |

## 7. Architecture ที่เสนอ (ยังไม่ได้ Deploy)

คง Hardware เดิมไว้ก่อน:

```
Joystick                         คำสั่งความเร็ว / ทิศทางจากผู้ใช้
   │
   ▼
Local Navigation                 วางแผนจาก Joystick + Obstacle Information   (ยังไม่พัฒนา / ยังไม่ทดสอบ)
   │                ▲
   ▼                │
Motion Computer     D435i → Raspberry Pi 4
RL Locomotion       Metric Depth + Static Proximity: ผ่านแล้ว
(Deployment ยังไม่ทดสอบ)   ต้องพัฒนา Obstacle Representation ก่อนเชื่อม Planner
```

Pi 4 รับผิดชอบ Perception เป็นหลัก Motion Computer รับผิดชอบการควบคุมการเดิน และ Navigation เป็นชั้นที่แปลงคำสั่ง Joystick ตามข้อจำกัดของสิ่งกีดขวาง

## 8. งานที่พักไว้

พักการทดสอบ Perception เพิ่มเติม (Floor Segmentation, Dynamic Obstacle, SLAM, End-to-End Latency, Navigation Integration) เพราะเป็นงานของ Autonomous Navigation ไม่ใช่เงื่อนไขพิสูจน์ความสามารถพื้นฐานของ Pi 4

**ข้อควรระวังด้านสถานะเครื่อง (ค้างอยู่):** การทดสอบล่าสุดหยุด `RobotVisionSystem` เดิมเพื่อให้ Native Depth ใช้กล้อง แต่ยังไม่มีผลยืนยันว่าคืนโปรแกรมเดิมแล้ว ถ้าไม่ได้ทดสอบต่อ ให้คืนระบบตาม Startup Script ที่ตรวจพบ ภายใต้สภาพหุ่นที่ปลอดภัย และตรวจ HTTP Stream ว่ากลับมาทำงาน โดยไม่เปิด Depth Probe ซ้อนกัน

## Final Decision

**Perception Feasibility — CLOSED / PASS within tested scope**

Raspberry Pi 4 Model B (4 GB) รับ Metric Depth จาก Intel RealSense D435i ที่ 640×480, 30 FPS และประมวลผลระยะสิ่งกีดขวางสามโซนได้ โดยไม่พบการข้ามหมายเลขเฟรมในการทดสอบระยะสั้น

- Hardware Upgrade: ยังไม่จำเป็นสำหรับงานที่พิสูจน์แล้ว
- Perception Feasibility: ผ่าน
- Navigation Readiness: ยังไม่ผ่านการพิสูจน์
- TALON-RL Deployment: แยกเป็นงานขั้นต่อไป

ขั้นต่อไป: ย้ายโฟกัสไปที่ Motion Computer (`192.168.123.161`) เพื่อ Audit ระบบควบคุมและเตรียม RL Deployment
