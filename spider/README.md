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
- **Weight budget:** ≤ 2 kg total (chassis + battery + electronics + motors)
- **Leg-joint motor:** both hip and knee use a GBM3506-class motor
  (e.g. `iPower GBM3506H-130T`), **direct-drive, no belt/gearbox** — see
  top-level `README.md`'s "Motor + transmission plan" table. Direct-drive
  is plausible here (unlike biped's hip) because spider is quasi-static
  with multiple legs sharing load, keeping per-joint torque need lower.
  Not yet bench-verified against real continuous-current specs.
- **Perception:** 2 cameras for stereo depth, feeding a real Exteroception
  signal into the policy (not the scripted `obstacle_dist` placeholder
  `DummyTalonEnv`/`clearance_reward` currently use) — see the fork noted in
  the top-level `README.md`'s TODO on off-the-shelf stereo-depth module vs.
  DIY dual-camera + manual calibration/disparity pipeline.

## TODO before any build work starts

- MCU / compute (onboard inference target) — note stereo depth computation
  itself has a real compute cost; factor this in alongside motor control.
  **Candidate: Raspberry Pi 4 or 5** (same as `biped/`, see its README for
  reasoning) — spider has no balance-control latency constraint stacked on
  top, so should fit more comfortably than biped's combined workload.
- **Driver/control board: Dagor Brushless Controller (candidate, locked)**
  — same as `biped/`'s hip/knee/wheel, see its README for full spec. Single
  on-board encoder is sufficient here (direct-drive, no belt — no second
  output-side encoder needed the way biped's hip has).
- Bench-test GBM3506H-130T's real torque/current before fully trusting
  the KV-derived estimate in the top-level README
- Power system (battery — voltage must match the BLDC driver + MCU + camera
  compute, not decided yet)
- Chassis/mechanical structure (3D printed is the likely default at this
  scale — not decided)
- Sensors: IMU at minimum, plus the stereo camera pair above
- Once hardware is set, add a spider-specific `ActionSpaceCfg`
  (`dim=8`, not A1's 12) and `IsaacLabSpiderEnv`/equivalent in `talon_rl` —
  and an Exteroception Encoder input path once the stereo pipeline exists,
  matching `talon-thesis`'s §3.2.2 architecture (this repo would be the
  first place that module gets tested on real hardware, not just A1)
