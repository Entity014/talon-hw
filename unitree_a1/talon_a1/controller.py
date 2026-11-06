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
from .obs import History, build_observation
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
    def __init__(self, driver, policy, remote=None, fault_table=None, stand_time=3.0, action_clip=1.0, log=None,
                 policy_kp=contract.KP, allow_walk=True):
        self.drv = driver
        self.policy = policy
        self.remote = remote or Remote()
        self.fault_table = fault_table or {}
        self.stand_time = stand_time
        self.action_clip = action_clip
        self.policy_kp = policy_kp
        self.allow_walk = allow_walk
        self.log = log
        self.last_action = np.zeros(contract.ACT_DIM, dtype=np.float32)
        self.hist = History()
        self.mode = PASSIVE
        self.ramp_from = None
        self.ramp_t = 0.0
        self.ramp_legs = []
        self.overruns = 0

    # -- state machine --------------------------------------------------------------------------------------
    def _set(self, mode, why=""):
        if mode != self.mode:
            print("[talon_a1] %s -> %s %s" % (self.mode, mode, why))
        self.mode = mode

    def _start_ramp(self, state, duration):
        self.ramp_from = contract.sdk_to_policy(np.asarray(state.q, dtype=np.float32))
        self.ramp_t = 0.0
        if self.ramp_from[8:12].mean() < contract.LYING_CALF:
            self.ramp_legs = [(contract.CROUCH_JOINT_POS, duration / 2.0), (contract.DEFAULT_JOINT_POS, duration / 2.0)]
        else:
            self.ramp_legs = [(contract.DEFAULT_JOINT_POS, duration)]

    def _ramp_target(self):
        t, start = self.ramp_t, self.ramp_from
        for pose, duration in self.ramp_legs:
            if t < duration:
                r = t / duration
                return (1.0 - r) * start + r * pose, False
            t -= duration
            start = pose
        return start, True

    def _send(self, q_policy, kp=contract.KP, kd=contract.KD):
        return self.drv.set_target(list(contract.policy_to_sdk(q_policy).astype(float)), kp, kd)

    def _send_stand(self, q_policy):
        return self._send(q_policy, contract.STAND_KP, contract.STAND_KD)

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
                if self._send_stand(q) and self.drv.arm():
                    self._set(STAND, "(L1+A)")
                else:
                    print("[talon_a1] arm refused (no robot state, or a fault)")
            return rec

        if self.mode == STAND:
            self.ramp_t += contract.POLICY_DT
            q, done = self._ramp_target()
            self._send_stand(q)
            rec["q_target"] = q
            if done and self.allow_walk and self.remote.pressed(START_WALK):
                self.last_action[:] = 0.0
                self.hist.reset()
                self._set(WALK, "(L1+X)")
            return rec

        # WALK
        if self.remote.pressed(STOP_WALK):
            self._start_ramp(state, 1.0)
            self._set(STAND, "(L1+Y)")
            return rec
        command = self.remote.command(state)
        x = build_observation(state, command)
        a, v_hat, z_hat = self.policy.action_mean(x, self.last_action, self.hist.buf)
        # The actor's tanh already bounds the mean to +-1; a smaller clip is for bring-up. Feed back what was applied.
        a = np.clip(a, -self.action_clip, self.action_clip).astype(np.float32)
        self.hist.push(x, a)
        self.last_action[:] = a
        q = contract.DEFAULT_JOINT_POS + contract.ACTION_SCALE * a
        self._send(q, self.policy_kp)
        rec.update(command=command, obs=x, action=a, v_hat=v_hat, z_hat=z_hat, q_target=q)
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
    REC = (("command", 3), ("obs", contract.OBS_DIM), ("action", 12), ("v_hat", contract.VEL_DIM),
           ("z_hat", contract.LATENT), ("q_target", 12))

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
