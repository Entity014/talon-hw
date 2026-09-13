# biped

2-leg wheeled robot.

## Locked so far

- **Envelope:** ≤ 20×20×20 cm
- **DoF:** 2 legs × (2 leg joints + 1 driven wheel) = 6 actuated DoF total
- **Actuation:** BLDC throughout (motor driver/controller board still open —
  compact envelope means favoring small integrated BLDC+driver modules,
  e.g. gimbal-style or mini-quadruped-actuator form factors, over separate
  large BLDC + external ESC)
- **Reward objective:** must fulfill the full 5-term vector from
  `talon_rl.config.RewardVectorCfg` (progress, clearance, energy, impact,
  smoothness) — same as A1/spider, not a reduced subset — **plus a new 6th
  term for balance/upright stability**, since this platform statically
  cannot stand (2-leg + wheels = dynamic balance / inverted-pendulum
  control, unlike A1's or spider's quasi-static stance).

  **This isn't just a control-difficulty risk to route around — `balance`
  vs. `clearance`/`progress` is a genuine Multi-Objective Module
  demonstration case, the same pattern as §3.3.1/§3.10.3(D)'s gap/pit
  qualitative test (high $w_{progress}$/low $w_{impact}$ → cross fast,
  accept risk; high $w_{impact}$ → go slow/around).** For biped, the same
  $w$-sweep test applies directly to the new term: high $w_{balance}$ /
  low $w_{progress}$ → approach an obstacle cautiously or refuse to cross
  it; high $w_{progress}$ / low $w_{balance}$ → cross fast, accept a
  higher tip-over risk. If anything this is a *clearer* demonstration than
  A1's impact-mitigation case, since the failure mode (falling over) is
  immediately visible rather than an internal force measurement. `balance`
  should be added to `talon_rl.config.RewardVectorCfg.term_names` as a
  first-class, $w$-conditioned term for this platform (A1/spider keep the
  existing 5 unchanged — this is a biped-specific config variant, not a
  change to the shared one), not bolted on outside the preference vector.

  `impact` and `clearance` still need biped-specific reinterpretation
  before they mean anything here: `impact_reward` currently penalizes peak
  **foot** contact force (discrete footstep landings) — a continuously
  rolling wheel doesn't land, so this term needs a different event
  definition for biped (e.g. curb/bump strikes) rather than reusing the
  A1 definition as-is. `clearance` (obstacle negotiation) carries over
  conceptually but its implementation (scripted `obstacle_dist` signal,
  same treatment as `DummyTalonEnv`) needs its own biped env, not a
  borrowed one.

## TODO before any build work starts

- MCU / compute (onboard inference target)
- Specific BLDC module + driver/control board, gear ratio
- Sensors (IMU at minimum — required for the new balance term; anything else?)
- Design the `balance` reward term's exact formula (likely pitch/roll
  deviation from upright, similar in spirit to how `smoothness` penalizes
  action rate) — not started yet
- Once the above are set, add a biped-specific `RewardVectorCfg` variant
  and `IsaacLabBipedEnv`/equivalent in `talon_rl`, matching the 6-DoF
  `ActionSpaceCfg` this platform needs
