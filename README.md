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

## Motor + transmission plan (2 motor sizes, whole project)

Superseded the original "single shared motor, QDD gearbox" idea below once
we tore down a real reference product for grounding (Mondo Robotics' Beni —
see `biped/README.md`'s reference-product note; teardown source: YouTube
"We Cracked Beni Open!"). The teardown showed labeled per-joint connectors
(R-KNEE/R-HIP/L-HIP/L-KNEE) on a custom centralized driver board (not a
purchasable dev board — proprietary, not something to copy directly), a
visibly **smaller motor at the hip driving through a timing belt**, and a
**larger motor, same size as the wheel motor, at the knee** (direct, no
visible belt) — plus a four-bar linkage at the knee (possibly with a
spring/shock, unconfirmed) suggesting passive compliance there.

Locked allocation, reasoned from that evidence + $K_t$ estimates from each
motor's KV (torque constant $K_t \approx 9.55/K_v$, ballpark from
datasheet KV only — **not verified against real continuous-current specs
for either motor**, bench-test before fully trusting):

| Platform | Joint | Motor | Transmission | Why |
|---|---|---|---|---|
| Biped | Hip | GBM3506-class (~55KV, smaller/cheaper) | + timing belt | Needs the most mechanical advantage (balance + occasional jump loading); belt gives torque multiplication + remote motor mounting (lower leg inertia) without losing backdrivability the way a high-ratio gearbox would |
| Biped | Knee | GBM5010-class (90KV, e.g. Rctimer GBM5010-150T) | direct | Matches Beni's same-size knee/wheel motor; less mechanical-advantage need than hip |
| Biped | Wheel | GBM5010-class | direct | Propulsion + balance correction; same size as knee per teardown |
| Spider | Hip | GBM3506-class | direct (no belt) | Quasi-static, multi-leg load-sharing keeps torque need lower than biped's single-point dynamic loading — direct-drive plausible here where it wasn't for biped's hip |
| Spider | Knee | GBM3506-class | direct | Same reasoning as spider hip; needs less torque than hip per the original moment-arm estimate, so no reason to upsize |

**2 motor sizes total for the whole project** (3506-class, 5010-class) —
cheaper to stock/source than a different part per joint. Belt-drive is
biped-hip-only; nothing else in the plan uses one right now.

**Still open / not verified:**
- Exact motor SKUs and current pricing — `iPower GBM3506H-130T` and
  `Rctimer GBM5010-150T 90KV` (~฿597) are the concrete candidates discussed
  so far, not final purchases
- Real continuous-current / stall-torque datasheet numbers for either motor
  — everything above is a KV-derived $K_t$ estimate
- Belt ratio, pulley sizes, and mounting geometry for biped's hip — not
  designed yet, only "yes, use a belt" is decided
- Whether biped's knee four-bar + possible spring is worth replicating —
  unconfirmed from the teardown photos alone

## Status

Scaffold only — no hardware specs locked yet beyond envelope, DoF, weight
budget, and actuation type. See `biped/README.md` and `spider/README.md`
for the full locked-spec / TODO lists.

## Related repos

- [`talon-rl`](https://github.com/Entity014/talon-rl) — the training pipeline being validated here
- [`talon-thesis`](https://github.com/Entity014/talon-thesis) — the thesis this pipeline belongs to
