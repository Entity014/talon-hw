# spider

Spider-topology legged robot.

## Locked so far

- **Envelope:** ≤ 20×20×20 cm
- **DoF:** 4 legs × 2 joints (hip pitch + knee pitch) = 8 actuated DoF total
  — no hip ab/adduction, unlike A1's real 3-DoF/leg. Consequence: no lateral
  stance adjustment, turning-in-place and lateral-disturbance recovery will
  be more limited than A1; expect a qualitatively different gait (more
  insect-crawl-like), not an A1-style trot. This is fine as its own
  generalization test, just don't expect A1 gaits to transfer.
- **Actuation:** BLDC throughout (motor driver/controller board still open —
  compact envelope means favoring small integrated BLDC+driver modules over
  separate large BLDC + external ESC)
- **Reward objective:** the existing 5-term `talon_rl.config.RewardVectorCfg`
  (progress, clearance, energy, impact, smoothness) as-is — no new term
  needed, unlike `biped/` — spider is quasi-static like A1 (doesn't need
  active balance to stand), so the same reward structure applies directly.

## TODO before any build work starts

- MCU / compute (onboard inference target)
- Specific BLDC module + driver/control board, gear ratio
- Sensors (IMU at minimum — anything else?)
- Once hardware is set, add a spider-specific `ActionSpaceCfg`
  (`dim=8`, not A1's 12) and `IsaacLabSpiderEnv`/equivalent in `talon_rl`
