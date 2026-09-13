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

## Open design fork: stereo depth module

Both platforms plan a 2-camera stereo depth setup feeding a real
Exteroception signal into the policy. Two ways to get there:

1. **Off-the-shelf stereo-depth module** (e.g. Intel RealSense — same
   family already used for A1 in `talon-thesis` §3.2.2 — or a
   Luxonis/OAK-style module that computes depth on-device). Reuses the
   same Exteroception Encoder input format across all three topologies,
   and doesn't burden the platform's own MCU with disparity computation.
2. **DIY dual RGB camera + manual stereo pipeline** (sync, calibration,
   rectification, disparity, depth conversion). Cheaper and more flexible
   on baseline/placement for a ≤20 cm chassis, but is real, nontrivial
   engineering work on top of everything else on the TODO lists, and adds
   compute load the MCU choice has to account for.

Leans toward (1) on cost-of-engineering-time grounds (reach for the
existing module before building the pipeline yourself) — not decided yet.

## Shared leg-joint motor across both platforms

Spider and biped are locked to the same envelope (≤20×20×20cm) and weight
budget (≤2kg), and the worst-case leg-joint torque estimate (single-leg
support + 2× dynamic margin) came out to the same target for both —
**~6 N·m at the hip, ~3 N·m at the knee** (see `biped/README.md`'s torque
note; assumes ~7-8cm leg segments, revisit once real leg dimensions are
fixed). Decision: **use the same motor model for both platforms' leg
joints** — one part to source/stock instead of two, cheaper on a limited
budget. This does NOT cover biped's wheel motor, which is a separate
selection (different function — propulsion + balance correction, not
leg-joint torque — sizing not done yet).

Direct-drive was ruled out (see `biped/README.md`) for backdrivability —
looking at Quasi-Direct Drive (QDD): large-diameter/pancake BLDC + low
gear ratio (<10:1), same pattern MIT Mini Cheetah and mjbots' moteus-c1
target. No specific motor SKU picked yet — search "QDD actuator" /
mini-cheetah-style actuator kits for concrete parts+pricing rather than
trusting a part number pasted here, this market moves fast.

## Status

Scaffold only — no hardware specs locked yet beyond envelope, DoF, weight
budget, and actuation type. See `biped/README.md` and `spider/README.md`
for the full locked-spec / TODO lists.

## Related repos

- [`talon-rl`](https://github.com/Entity014/talon-rl) — the training pipeline being validated here
- [`talon-thesis`](https://github.com/Entity014/talon-thesis) — the thesis this pipeline belongs to
