# spider

Spider-topology legged robot — hardware specs not locked yet.

TODO before any build work starts:
- MCU / compute (onboard inference target)
- Motor + driver choice, gear ratio, leg count
- Sensors (IMU at minimum — anything else?)
- Joint count / DoF, matched against `talon_rl`'s `ActionSpaceCfg` once
  this platform's action space is defined
