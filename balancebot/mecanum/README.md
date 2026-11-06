# balancebot/mecanum

Self-balancing robot on 4 mecanum wheels, all coaxial (two pairs, one pair per
side). Pitch is the only unstable axis. Trained with `talon-rl` as its own
platform, like `biped/` and `spider/`.

Scope: same boundary as the top-level `README.md`. This is a different topology
from the A1, so nothing here is §3.7 zero-shot transfer.

## Locked so far

- **Layout:** 4 mecanum wheels on one axle line, 2 per side. Chirality is
  mirrored so that equal drive on all wheels cancels lateral roller force
  (behaves like one wide wheel for pitch balance).
- **Actuation:** 4 independent BLDC motors, one per wheel. Gives forward, yaw
  and strafe while balancing. Roll has no actuator and relies on track width.
- **Design targets:** wheel 60 mm (r = 0.03 m), mass 1-2 kg, max speed 1 m/s
  (~320 rpm at the wheel; lowered from 2 m/s). Speed target is a ceiling, not
  a requirement to hit.
- **Wheel torque need** (first-order, not bench-verified, CoM height 10-15 cm):
  ~0.03-0.06 N·m per wheel for balance and 2 m/s² accel, up to ~0.2 N·m per
  wheel to recover from a 15° lean at 2 kg / 15 cm (quasi-static bound).
- **Motor:** Mitoot 5010 outrunner, **360KV** variant (50×10 mm, 80 g, 4 mm
  shaft, rear mounting holes 16/19 mm, listed for 2-6S). Pancake shape keeps
  the wheel stack narrow.
  - **Outline (from seller drawing, read from an image):** rotor Ø50 x 8.5 mm,
    base-to-rotor-front 20 mm, overall 23 mm. Rotor face: 4x M3 on a Ø12 bolt
    circle (pulley mounting). Stator base: 4x M3 on Ø16 / Ø19 (chassis
    mounting). Front shaft Ø4 (user-confirmed; one listing said 6 mm, ignore it).
    A short rear shaft nub (~2 mm,
    possibly with a clip) appears in the drawing: check it can take a Ø6 x
    2.5 magnet.
- **Transmission:** GT2 timing belt, 2:1 (motor pulley to wheel hub), wheel on
  its own bearings. Motor is offset from the wheel axis, so width is set by the
  wheels, not the motors. Ratio kept <= 2-2.5:1 to limit reflected inertia
  (grows with N²) and keep the wheel backdrivable.
- **Battery:** 12 V (3S).
- **Why 360KV, not 750KV** (estimates from listed KV, Kt = 9.55/KV): at 2:1
  the 0.2 N·m wheel peak needs ~4 A with 360KV but ~8 A with 750KV, which is
  the DRV8316's 8 A peak. At 1 m/s the motor turns ~640 rpm, back-EMF is only
  ~1.8 V (360KV), so a 12 V bus is plenty.
  - **Pole pairs: 7** (user-reported, i.e. 14 magnets, 12N14P). Mitoot
    listings do not state it and a different 5010 360KV listing says 24N28P,
    so confirm on the real motor by counting magnets or with SimpleFOC's
    pole-pair search.
  - **Unconfirmed, measure on the real motor:** KV, phase R, phase L, whether
    the rear shaft nub is long enough for the encoder magnet.
  - Other 5010-class numbers found (not this motor): R 0.085 Ω, L 85 µH
    (Berkeley Humanoid Lite motor notes, unit label looks wrong); R 0.152 Ω,
    no-load 0.34 A (a 360KV listing). Use only as order of magnitude.
  - **Main risk:** low inductance (~tens to ~100 µH) means large PWM current
    ripple. Rough ripple dI = V/(4·L·f): ~1.2 A p-p at 12 V, 85 µH, 30 kHz,
    vs ~1 A nominal current. Literature says low-L motors may need 50-100 kHz.
    Needs a check that STM32G4 + DRV8316 FOC runs at that rate. If it does
    not, a higher-L, lower-KV shaft motor (KV ~100-150 works at 12 V, 2:1) is
    the fallback.
  - Belt side load on a prop-class motor's bearings: keep the pulley close to
    the bell and the tension light.
- **Driver board:** one custom board per motor, identical design, derived from
  SimpleFOCMini v2.3 (DRV8316, 3-PWM, 3 low-side current sensors, 5-35 V).
  - **MCU:** STM32G431 (native FDCAN, ADC synced to PWM for current sense)
  - **Encoder:** AS5048A (14-bit, SPI) on a **separate small sensor board**
    mounted behind the motor, magnet on the rear shaft end. The driver board
    has a JST-GH SPI connector (3.3 V, GND, SCK, MISO, MOSI, CS) and no sensor
    of its own. This frees the driver board from the motor's 50 mm outline.
    Keep SPI cable <= ~15-20 cm, SPI clock 1-4 MHz, series resistors and local
    decoupling at the sensor. Keep phase wires <= ~10-15 cm, twisted (both
    limits are rough, not measured). An off-the-shelf AS5048A breakout can
    be used for bench tests now.
  - **Bus:** CAN to the onboard computer
- **Compute:** Raspberry Pi 4/5 runs the policy and talks to all four boards.
- **Reward:** `talon_rl` 5-term vector variant plus a `balance` term, as in
  `biped/`. `impact` and `clearance` have no matching event here yet.

## Open design notes

- **Chirality placement:** put chirality A on the outer wheels and B on the
  inner wheels (or the mirror). Then strafe produces no parasitic yaw moment.
- **Strafe limit:** lateral accel tips the robot sideways (no roll actuator).
  Needs an accel limit in the command or reward.
- **Sim model of the rollers:** do not model individual rollers. Plan is a
  smooth cylinder with a custom ground-force model (drive force along the
  roller axis direction, near-free along the other). Not decided.

## TODO before any build work starts

- **Bench test, one motor** (optional, the board is going ahead without it):
  Mini v2.3 + Nucleo-G474RE (or similar G4) + AS5048A + Mitoot 5010 360KV.
  Prove torque mode with current sense, high PWM frequency with low L, and
  cogging at low speed. Doing this before the PCB order is cheaper than after.
- Confirm SimpleFOC torque mode with low-side current sense runs on STM32G4.
- Check the SimpleFOCMini license before deriving schematic/layout from it.
  The repo page does not state it.
- Check DRV8316 continuous vs peak current in the datasheet (the repo page
  only gives "max 8 A").
- Board v1: design, order 4-5 pieces. Needs the `kicad-jlcpcb` MCP working.
- CAN protocol: one broadcast frame per control cycle for sync, per-board
  command and state frames. Measure Pi → CAN → board latency and
  randomize it in sim.
- IMU: mounted on the body (Pi or a separate board), not on a wheel board.
- Power: 12 V (3S) battery, within DRV8316 range; Pi needs its own 5 V buck
  from the battery. Check 4 motors x ~4 A peak against battery C-rating.
- Wheel size and chassis (3D printed is the likely default). Not decided.
- Design the `balance` reward term and a mecanum `RewardVectorCfg` variant
  plus env in `talon_rl`.
