# talon-hw

Physical hardware for informal validation of the `talon-rl` training
pipeline — does a policy trained by `talon-rl` actually work once it leaves
simulation, on hardware that's actually on hand.

## Scope boundary — read before citing this anywhere in the thesis

**This is NOT evidence for `talon-thesis`'s §3.7 / Objective 2 (Cross-Embodiment
Generalization).** §3.7 specifically claims *zero-shot* transfer within the
same joint topology as the Unitree A1. The two platforms here —

- `biped/` — 2-leg wheeled robot
- `spider/` — spider-topology legged robot

— are different topologies from the A1 (quadruped), so nothing run on them
is zero-shot transfer in §3.7's sense. Each platform needs its own training
run through the same pipeline, not a transferred A1 policy. That's a valid
thing to test (does the Multi-Objective Module / MOPPO setup generalize as
a *method* across embodiments, via retraining) — but it is a different claim
than §3.7 makes, and should not be presented as Objective 2 evidence without
explicitly checking with the thesis advisor first.

## Status

Scaffold only — no hardware specs locked yet (MCU, motor, sensor choices
TBD for both platforms). See `biped/README.md` and `spider/README.md`.

## Related repos

- [`talon-rl`](https://github.com/Entity014/talon-rl) — the training pipeline being validated here
- [`talon-thesis`](https://github.com/Entity014/talon-thesis) — the thesis this pipeline belongs to
