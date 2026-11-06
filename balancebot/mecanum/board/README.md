# balancebot/mecanum/board

Draft spec for the per-motor BLDC driver board (one board per wheel motor, 4
identical boards). Not a schematic yet.

## Approach: derive from SimpleFOCMini v2.3

Keep the Mini's power stage as is and add the MCU side around it. Source:
github.com/simplefoc/SimpleFOCMini (EasyEDA project, Gerber, BOM, schematic
PDF). Repo says MIT; it is not stated whether that covers the hardware files,
so read the LICENSE file and keep attribution.

**Keep from the Mini (do not redraw):**

- DRV8316 power stage, VM caps, charge pump, current-sense routing, layout.
- The Mini's DRV8316 is **DRV8316CTRGFR** (the "C" variant). Its BOM has
  resistors R_MODE, R_DV1, R_DV2, R_DRVOFF, R_FLT, which suggests pin
  (hardware) configuration with no SPI to the driver. This is inferred from the
  BOM, not confirmed: read the DRV8316C datasheet before relying on it.
- Current-sense gain 0.15 V/A (set by resistor). Enough for us: 4 A peak is
  ±0.6 V around mid-scale, ~5 mA per ADC LSB. No change needed.

**Change / add:**

- **Buck inductor.** The Mini has none: it uses buck resistor mode (R_BK = 22 Ω,
  C_AVDD1 = 22 µF), limited to ~40 mA total (datasheet SLVSF16B, Table 8-5).
  Our MCU side needs more, so replace R_BK with a **47 µH inductor + 22 µF**
  (inductor mode, up to 200 mA minus AVDD load; saturation current margin,
  buck can source up to 600 mA peak). Confirm the C variant has the same buck
  options.
- **STM32G431** (new), CAN transceiver, SPI sensor connector, SWD, status LED.
- Mini is 24 x 25 mm; ours will be larger. No 50 mm motor-outline limit,
  because the sensor is on a separate board.

**Fallback:** if the Mini layout is hard to reuse, redraw the power stage from
the DRV8316 datasheet typical application, using the Mini's BOM values.

## Mini schematic notes (from the schematic image the user supplied)

The image's title block says REV 2.2 (2025-05-19), not 2.3. Compare with the
latest schematic PDF in the repo before copying values.

- Pin configuration, no SPI: 3x PWM (R_MODE 47 kΩ), slew 200 V/µs (SLEW to
  AVDD), current-sense gain 0.15 V/A (GAIN to GND), buck output 3.3 V
  (VSEL_BK to GND), OCP/SR to GND (note says "around 16 A").
- **OCP at ~16 A is above the driver's 8 A peak rating.** In the datasheet the
  lowest OCP trip is ~16 A typ (10-20 A) for the same setting family, so the
  driver cannot protect the 4-8 A range. Do current limiting in firmware from
  SOA/SOB/SOC.
- INLA/B/C tied high, nSLEEP is the enable. nFAULT has a 4.7 kΩ pull-up and an
  LED.
- VM sense: R_DV1 47 k / R_DV2 4.7 k divider (VM/11, 35 V maps to 3.2 V) on
  the "0.1_VCC" net, for an ADC.
- Buck is unused: R_BK 22 Ω + C 22 µF (resistor mode, max 40 mA, "not used").
  AVDD (3.3 V, 1 µF) is brought out to the host header.
- Host header H1 (2x5, 2.54 mm): GND, 3V3 (AVDD), nSLEEP, IN1, IN2, IN3, SOA,
  SOB, SOC, VM/11. This is exactly the interface the STM32G431 needs: 3 PWM
  (TIM1), 3 ADC, VM ADC, nSLEEP and nFAULT GPIO. Keep the same header on our
  board so a stock Mini can be bench-tested with the same firmware.

## Block diagram

```text
12 V battery ──> [TVS + fuse + bulk cap] ──> DRV8316C (VM)
                                              │ buck (47 µH) 3.3 V ──> STM32G431, AS5048A, CAN xcvr
STM32G431 ──TIM1 3x PWM──> INHA/B/C ──> OUTA/B/C ──> motor (3 phase pads)
STM32G431 <──ADC── SOA/SOB/SOC (current sense)
STM32G431 ──GPIO──> DRVOFF / nFAULT in
STM32G431 <──SPI── AS5048A sensor board (JST-GH, off board)
STM32G431 <──FDCAN── transceiver ──> CAN bus (JST-GH)
```

## Driver facts (DRV8316 datasheet SLVSF16B, covers R and T only)

- 4.5-35 V operating (40 V abs max), 8 A peak, 95 mΩ HS+LS, PWM up to 200 kHz.
- **No continuous current rating.** Only "power dissipation and thermal limits
  must be observed". RθJA 25.7 °C/W (JEDEC board). Rough conduction loss at
  4 A RMS per phase is ~2 W: fine for our short peaks, confirm on the real
  board.
- 3x PWM mode (INHx only) with SOx current-sense outputs available.
- Bulk cap rated at least twice VM (12 V bus: >= 25 V; the Mini uses a 10 µF
  1210 part, check its voltage rating and add more bulk).

## MCU: STM32G431

- TIM1 center-aligned for the 3 PWM outputs. At 100 kHz that is ~850 counts per
  period (about 9-10 bit duty resolution). Current loop can run slower than PWM.
- ADC triggered from TIM1 for SOA/SOB/SOC, plus a VM divider.
- One SPI bus to the AS5048A (no SPI to the driver if the C variant is
  pin-configured).
- FDCAN with a 3.3 V CAN transceiver, 120 Ω termination selectable by a solder
  jumper.
- GPIO: DRVOFF, nFAULT in, one status LED.
- Verify every pin mapping in CubeMX before layout.

## Connectors

- Power in: XT30 or similar, 12 V, plus TVS and fuse.
- Phase out: 3 solder pads (or bullet connectors) rated >= 6 A.
- Sensor: JST-GH 6 pin: 3V3, GND, SCK, MISO, MOSI, CS (AS5048A on its own small
  board at the motor).
- CAN: JST-GH 4 pin (CANH, CANL, GND, spare).
- SWD header and a UART debug header.

## Bring-up features (board v1 is going out without a bench test)

- Test points: 3 phase outputs, SOA/SOB/SOC, VREF, 3V3, VM, nFAULT, SPI lines.
- Solder jumpers: CAN termination.
- Order 4-5 boards first.
- Optional: buy 1-2 stock SimpleFOCMini boards to bench-test the motor now
  (with a Nucleo G4 and an AS5048A breakout) while the derived board is made.

## Constraints

- Phase wires from motor to board <= ~10-15 cm, twisted (rough, not measured).
- Sensor SPI cable <= ~15-20 cm, 1-4 MHz.

## Open items

- DRV8316C datasheet: interface, buck options, current-sense specs.
- Whether the motor's low inductance forces PWM >= 40-100 kHz and whether the
  STM32G4 loop can keep up (measure on the real motor).
- Hardware license scope (MIT stated; hardware files not explicitly covered).
- Layout tool: the Mini source is EasyEDA, which also hands off to JLCPCB.
  The `kicad-jlcpcb` MCP did not connect in this session (ENOENT).
