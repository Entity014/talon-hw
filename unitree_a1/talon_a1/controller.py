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
from .obs import History, PromoHistory, build_observation, build_observation_promo_v2, correct_projected_gravity, projected_gravity
from .policy import OnnxPolicy, PromoNumpyPolicy
from .remote import Remote, combo

PASSIVE, STAND, CROUCH_DOWN, WALK = "PASSIVE", "STAND", "CROUCH_DOWN", "WALK"

# Uniform observation noise of the PROMO / oursA training task (observation layout of build_observation_promo_v2):
# gyro +-0.2, projected gravity +-0.05, command 0, joint_pos +-0.01, joint_vel +-1.5, last_action 0.
TRAIN_OBS_NOISE = np.concatenate([np.full(3, 0.2), np.full(3, 0.05), np.zeros(3), np.full(12, 0.01), np.full(12, 1.5), np.zeros(12)]).astype(np.float32)


def obs_noise_amplitude(scale=0.0, dq=None):
    """Per-channel amplitude of the noise added to the observation of a PROMO policy: `scale` x the training noise on every channel, and if `dq` is given
    the joint_vel channels get exactly `dq` rad/s (so dq=1.5 with scale=0 is the dq-only A/B test). Returns None when nothing is added."""
    amp = TRAIN_OBS_NOISE * float(scale)
    if dq is not None:
        amp[21:33] = float(dq)
    return amp if np.any(amp > 0) else None

STAND_UP = combo("L1", "A")
START_WALK = combo("L1", "X")
STOP_WALK = combo("L1", "Y")
TO_PASSIVE = combo("L1", "B")
CLEAR = combo("SELECT")


def fault_names(mask, table):
    return [name for name, bit in sorted(table.items()) if mask & bit]


class Controller(object):
    def __init__(self, driver, policy, remote=None, fault_table=None, stand_time=3.0, action_clip=1.0, log=None,
                 policy_kp=None, allow_walk=True, soft_start=0.0,
                 action_scale=None, default_joint_pos=None, w=None, obs_noise=None, noise_seed=0, lead_ms=0.0,
                 action_smooth=0.0, pitch_bias=0.0, front_lift_bias=0.0,
                 imu_roll_bias=0.0, imu_pitch_bias=0.0, imu_zero=False):
        self.drv = driver
        self.policy = policy
        self.remote = remote or Remote()
        self.fault_table = fault_table or {}
        self.stand_time = stand_time
        self.action_clip = action_clip
        self.allow_walk = allow_walk
        self.soft_start = soft_start
        self.action_smooth = float(action_smooth)  # EMA smoothing factor alpha in [0, 1): a_applied = (1 - alpha) * a_new + alpha * a_prev (0 = no smoothing)
        self.pitch_bias = float(pitch_bias)  # rad: flex rear calves in WALK to shift weight off front legs
        self.front_lift_bias = float(front_lift_bias)  # rad: flex front thigh/calf proportional to forward cmd (vx > 0)
        self.imu_roll_bias = float(imu_roll_bias)
        self.imu_pitch_bias = float(imu_pitch_bias)
        self.imu_zero = bool(imu_zero)
        self._stand_g_buf = []

        # Detect PROMO v2 policy (ONNX or pure NumPy)
        self.is_promo = isinstance(policy, (OnnxPolicy, PromoNumpyPolicy))

        if default_joint_pos is not None:
            self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32)
        elif self.is_promo and hasattr(policy, "default_q"):
            self.default_joint_pos = np.asarray(policy.default_q, dtype=np.float32)
        else:
            self.default_joint_pos = contract.DEFAULT_JOINT_POS.copy()

        if action_scale is not None:
            self.action_scale = float(action_scale)
        elif self.is_promo and hasattr(policy, "action_scale"):
            self.action_scale = float(policy.action_scale)
        else:
            self.action_scale = float(contract.ACTION_SCALE)

        if policy_kp is not None:
            self.policy_kp = float(policy_kp)
        elif self.is_promo and hasattr(policy, "kp"):
            self.policy_kp = float(policy.kp)
        else:
            self.policy_kp = float(contract.KP)

        self.w = np.asarray(w if w is not None else contract.DEFAULT_W_PROMO_V2, dtype=np.float32)
        if self.is_promo and hasattr(policy, "set_preference"):
            policy.set_preference(self.w)

        self.walk_t = 0.0
        self.obs_noise = None if obs_noise is None else np.asarray(obs_noise, dtype=np.float32)  # uniform +- amplitude per observation channel (PROMO layout), None = no noise
        self.rng = np.random.default_rng(noise_seed)
        self.lead_s = float(lead_ms) / 1000.0  # lead compensation of a PROMO observation: joint_pos += joint_vel * lead (a guess of where the joint is when the delayed command acts); 0 = off
        self.log = log
        self.last_action = np.zeros(contract.ACT_DIM, dtype=np.float32)

        if self.is_promo:
            self.hist = PromoHistory(length=getattr(policy, "hist_len", contract.HIST_LEN_PROMO_V2),
                                     dim=getattr(policy, "obs_dim", contract.OBS_DIM_PROMO_V2))
        else:
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

    def _apply_imu_bias(self, x):
        if self.imu_roll_bias == 0.0 and self.imu_pitch_bias == 0.0:
            return x
        x = x.copy()
        x[3:6] = correct_projected_gravity(x[3:6], self.imu_roll_bias, self.imu_pitch_bias)
        return x

    def _noisy(self, x):
        """The observation the policy gets: x with the optional IMU bias and lead compensation, plus uniform noise per channel (as in training, where the noise is on the observation that enters the history), or x itself."""
        x = self._apply_imu_bias(x)
        if self.lead_s != 0.0:
            x = x.copy()
            x[9:21] += x[21:33] * self.lead_s  # PROMO layout: joint_pos_minus_default 9:21, joint_vel 21:33
        if self.obs_noise is None:
            return x
        return (x + self.rng.uniform(-1.0, 1.0, x.shape).astype(np.float32) * self.obs_noise).astype(np.float32)

    def _start_ramp(self, state, duration):
        self.ramp_from = contract.sdk_to_policy(np.asarray(state.q, dtype=np.float32))
        self.ramp_t = 0.0
        if self.ramp_from[8:12].mean() < contract.LYING_CALF:
            self.ramp_legs = [(contract.CROUCH_JOINT_POS, duration / 2.0), (self.default_joint_pos, duration / 2.0)]
        else:
            self.ramp_legs = [(self.default_joint_pos, duration)]

    def _start_ramp_down(self, state, duration=2.0):
        self.ramp_from = contract.sdk_to_policy(np.asarray(state.q, dtype=np.float32))
        self.ramp_t = 0.0
        # Ramp down gracefully from current pose to crouch
        self.ramp_legs = [(contract.CROUCH_JOINT_POS, duration)]

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

        if self.remote.pressed(TO_PASSIVE) and self.mode not in (PASSIVE, CROUCH_DOWN):
            self._start_ramp_down(state, duration=1.5)
            self._set(CROUCH_DOWN, "(L1+B)")
            return rec

        if self.mode == CROUCH_DOWN:
            self.ramp_t += contract.POLICY_DT
            q, done = self._ramp_target()
            self._send_stand(q)
            rec["q_target"] = q
            if done:
                self.drv.disarm()
                self._set(PASSIVE, "crouch complete")
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
            if done and self.imu_zero:
                self._stand_g_buf.append(projected_gravity(state.quat))
                if len(self._stand_g_buf) > 25:
                    self._stand_g_buf.pop(0)
            if done and self.allow_walk and self.remote.pressed(START_WALK):
                if self.imu_zero and self._stand_g_buf:
                    g_avg = np.mean(self._stand_g_buf, axis=0)
                    self.imu_pitch_bias = float(np.arctan2(-g_avg[0], -g_avg[2]))
                    self.imu_roll_bias = float(np.arctan2(-g_avg[1], -g_avg[2]))
                    print("[talon_a1] imu_zero calibrated: roll_bias=%.4f rad, pitch_bias=%.4f rad"
                          % (self.imu_roll_bias, self.imu_pitch_bias))
                self.last_action[:] = 0.0
                self.walk_t = 0.0
                cmd = self.remote.command(state)
                if self.is_promo:
                    init_obs = self._noisy(build_observation_promo_v2(state, cmd, self.last_action, self.default_joint_pos))
                    self.hist.fill(init_obs)
                else:
                    self.hist.fill(build_observation(state, cmd))
                self._set(WALK, "(L1+X)")
            return rec

        # WALK
        if self.remote.pressed(STOP_WALK):
            self._start_ramp(state, 1.0)
            self._set(STAND, "(L1+Y)")
            return rec
        command = self.remote.command(state)

        if self.is_promo:
            x = self._noisy(build_observation_promo_v2(state, command, self.last_action, self.default_joint_pos))
            self.hist.push_obs(x)
            a = self.policy.action_mean(x, self.hist.buf, self.w)
            v_hat = None
            z_hat = None
        else:
            x = self._apply_imu_bias(build_observation(state, command))
            a, v_hat, z_hat = self.policy.action_mean(x, self.last_action, self.hist.buf)

        # RMA's tanh bounds the mean to +-1; PROMO actors are linear and need |a| > 1 to step, so run.py raises the clip for them. Feed back what was applied.
        a = np.clip(a, -self.action_clip, self.action_clip).astype(np.float32)

        # Action EMA low-pass filtering to damp high-frequency chatter on real hardware
        if self.action_smooth > 0.0:
            a = ((1.0 - self.action_smooth) * a + self.action_smooth * self.last_action).astype(np.float32)

        # The settled action is far from 0 (up to 0.99 on the robot), so jumping there jerks the legs when the policy
        # opens. Ease the action and the gains in from the stand instead; training had no such ramp.
        self.walk_t += contract.POLICY_DT
        r = min(1.0, self.walk_t / self.soft_start) if self.soft_start > 0.0 else 1.0
        a = (r * a).astype(np.float32)

        if not self.is_promo:
            self.hist.push(x, a)

        self.last_action[:] = a
        q = self.default_joint_pos + self.action_scale * a
        if self.pitch_bias != 0.0:
            q[10:12] -= r * self.pitch_bias  # RL_calf, RR_calf
        if self.front_lift_bias != 0.0 and command[0] > 0.0:
            vx_scale = max(float(getattr(self.remote, "vx_max", 0.5)), 1e-6)
            fwd = min(1.0, float(command[0]) / vx_scale)
            q[4:6] -= r * fwd * 0.5 * self.front_lift_bias  # FL_thigh, FR_thigh
            q[8:10] -= r * fwd * self.front_lift_bias  # FL_calf, FR_calf
        self._send(q, (1.0 - r) * contract.STAND_KP + r * self.policy_kp, (1.0 - r) * contract.STAND_KD + r * contract.KD)
        rec.update(command=command, obs=x, action=a, q_target=q)
        if v_hat is not None:
            rec["v_hat"] = v_hat
        if z_hat is not None:
            rec["z_hat"] = z_hat
        if self.is_promo:
            rec["w"] = self.w
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

    def __init__(self, obs_dim=contract.OBS_DIM):
        self.obs_dim = obs_dim
        self.rec_spec = (
            ("command", 3),
            ("obs", self.obs_dim),
            ("action", 12),
            ("v_hat", contract.VEL_DIM),
            ("z_hat", contract.LATENT),
            ("q_target", 12),
            ("w", contract.W_DIM_PROMO_V2),
        )
        self.rows = {name: [] for name, _ in self.FIELDS + self.rec_spec}
        self.rows.update(t=[], mode=[], faults=[], compute_s=[], tick=[], loop_overruns=[])

    def append(self, state, rec):
        for name, n in self.FIELDS:
            self.rows[name].append(np.asarray(getattr(state, name), dtype=np.float32))
        for name, n in self.rec_spec:
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
