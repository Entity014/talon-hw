# spider

Spider-topology legged robot.

## Locked so far

- **Envelope:** ≤ 20×20×20 cm
- **Actuation:** BLDC throughout (motor driver/controller board still open —
  compact envelope means favoring small integrated BLDC+driver modules over
  separate large BLDC + external ESC)

## TODO before any build work starts

- MCU / compute (onboard inference target)
- Specific BLDC module + driver/control board, gear ratio, leg count
- Sensors (IMU at minimum — anything else?)
- Joint count / DoF, matched against `talon_rl`'s `ActionSpaceCfg` once
  this platform's action space is defined
