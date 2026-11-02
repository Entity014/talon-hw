"""50 Hz controller: PASSIVE -> STAND -> WALK state machine on top of the driver's safety core.

Remote combos (edge triggered):
    L1 + A   PASSIVE -> STAND   arm and ramp to the default pose
    L1 + X   STAND   -> WALK    start the policy (only once the ramp is done)
    L1 + Y   WALK    -> STAND   ramp back to the default pose
    L1 + B   any     -> PASSIVE disarm (damping)
    SELECT   PASSIVE            clear latched faults
    L2 + B   any                e-stop, handled in the C++ motor loop and latched there
"""

import time

import numpy as np

from . import contract
from .obs import History, ObservationBuilder
from .remote import Remote, combo

PASSIVE, STAND, WALK = "PASSIVE", "STAND", "WALK"

STAND_UP = combo("L1", "A")
START_WALK = combo("L1", "X")
STOP_WALK = combo("L1", "Y")
TO_PASSIVE = combo("L1", "B")
CLEAR = combo("SELECT")


def fault_names(mask, table):
    return [name for name, bit in sorted(table.items()) if mask & bit]


class Controller(object):
    def __init__(self, driver, policy, remote=None, fault_table=None, stand_time=3.0, action_clip=4.0, log=None):
        self.drv = driver
        self.policy = policy
        self.remote = remote or Remote()
        self.fault_table = fault_table or {}
        self.stand_time = stand_time
        self.action_clip = action_clip
        self.log = log
        self.obs = ObservationBuilder()
        self.hist = History()
        self.mode = PASSIVE
        self.ramp_from = None
        self.ramp_t = 0.0
        self.ramp_time = stand_time
        self.overruns = 0

    # -- state machine --------------------------------------------------------------------------------------
    def _set(self, mode, why=""):
        if mode != self.mode:
            print("[talon_a1] %s -> %s %s" % (self.mode, mode, why))
        self.mode = mode

    def _start_ramp(self, state, duration):
        self.ramp_from = contract.sdk_to_policy(np.asarray(state.q, dtype=np.float32))
        self.ramp_t = 0.0
        self.ramp_time = duration

    def _ramp_target(self):
        r = min(1.0, self.ramp_t / self.ramp_time)
        return (1.0 - r) * self.ramp_from + r * contract.DEFAULT_JOINT_POS, r >= 1.0

    def _send(self, q_policy):
        return self.drv.set_target(list(contract.policy_to_sdk(q_policy).astype(float)), contract.KP, contract.KD)

    def step(self, state):
        """One 20 ms tick. Returns the record logged for it."""
        rec = {"mode": self.mode, "faults": int(state.faults)}
        self.remote.update(state)

        if state.faults:
            if self.mode != PASSIVE:
                self._set(PASSIVE, "fault: %s" % ", ".join(fault_names(state.faults, self.fault_table)))
                self.drv.disarm()
            if self.remote.pressed(CLEAR):
                self.drv.clear_faults()
                print("[talon_a1] faults cleared")
            return rec

        if self.remote.pressed(TO_PASSIVE) and self.mode != PASSIVE:
            self.drv.disarm()
            self._set(PASSIVE, "(L1+B)")
            return rec

        if self.mode == PASSIVE:
            if self.remote.pressed(STAND_UP):
                self._start_ramp(state, self.stand_time)
                q, _ = self._ramp_target()
                if self._send(q) and self.drv.arm():
                    self._set(STAND, "(L1+A)")
                else:
                    print("[talon_a1] arm refused (no robot state, or a fault)")
            return rec

        if self.mode == STAND:
            self.ramp_t += contract.POLICY_DT
            q, done = self._ramp_target()
            self._send(q)
            rec["q_target"] = q
            if done and self.remote.pressed(START_WALK):
                self.obs.reset()
                self.hist.reset()
                self._set(WALK, "(L1+X)")
            return rec

        # WALK
        if self.remote.pressed(STOP_WALK):
            self._start_ramp(state, 1.0)
            self._set(STAND, "(L1+Y)")
            return rec
        command = self.remote.command(state)
        o = self.obs.build(state, command)
        hist = self.hist.push(o)
        a, v_hat = self.policy.action_mean(o, hist)
        # Training never clipped the action; this bound only stops a wild output. Feed back what was applied.
        a = np.clip(a, -self.action_clip, self.action_clip).astype(np.float32)
        self.obs.last_action[:] = a
        q = contract.DEFAULT_JOINT_POS + contract.ACTION_SCALE * a
        self._send(q)
        rec.update(command=command, obs=o, action=a, v_hat=v_hat, q_target=q)
        return rec

    # -- loop -----------------------------------------------------------------------------------------------
    def run(self, duration=None):
        dt = contract.POLICY_DT
        t0 = time.monotonic()
        next_t = t0
        while duration is None or next_t - t0 < duration:
            state = self.drv.state()
            start = time.monotonic()
            rec = self.step(state)
            rec["compute_s"] = time.monotonic() - start
            rec["t"] = start - t0
            if self.log is not None:
                self.log.append(state, rec)
            next_t += dt
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                self.overruns += 1
                next_t = time.monotonic()


class RunLog(object):
    """Per-tick arrays, saved as one .npz. NaN where a field does not apply in that mode."""

    FIELDS = (("q", 12), ("dq", 12), ("tau_est", 12), ("quat", 4), ("gyro", 3), ("rpy", 3), ("foot_force", 4))
    REC = (("command", 3), ("obs", contract.OBS_DIM), ("action", 12), ("v_hat", 3), ("q_target", 12))

    def __init__(self):
        self.rows = {name: [] for name, _ in self.FIELDS + self.REC}
        self.rows.update(t=[], mode=[], faults=[], compute_s=[], tick=[], loop_overruns=[])

    def append(self, state, rec):
        for name, n in self.FIELDS:
            self.rows[name].append(np.asarray(getattr(state, name), dtype=np.float32))
        for name, n in self.REC:
            v = rec.get(name)
            self.rows[name].append(np.full(n, np.nan, np.float32) if v is None else np.asarray(v, np.float32))
        self.rows["t"].append(rec["t"])
        self.rows["mode"].append(rec["mode"])
        self.rows["faults"].append(rec["faults"])
        self.rows["compute_s"].append(rec["compute_s"])
        self.rows["tick"].append(state.tick)
        self.rows["loop_overruns"].append(state.overruns)

    def save(self, path, meta=None):
        arrays = {k: np.asarray(v) for k, v in self.rows.items()}
        if meta is not None:
            import json

            arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
