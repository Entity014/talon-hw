"""33-D policy observation x_t from a robot state, term for term as PromoRmaObservationsCfg.PolicyCfg.

Isaac Lab applies no scaling to these terms and the run manifest has observation_normalization "none"; the
training noise is a training-time corruption only. The joint position is the raw angle (mdp.joint_pos), not
relative to the default pose.
"""

import numpy as np

from . import contract


def projected_gravity(quat_wxyz):
    """World gravity direction (0, 0, -1) in the body frame: R(q)^T g (Isaac Lab quat_apply_inverse)."""
    w, x, y, z = quat_wxyz
    # -(third row of R)
    return -np.array(
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)], dtype=np.float32
    )


def correct_projected_gravity(g, roll_bias=0.0, pitch_bias=0.0):
    """Remove static IMU mounting roll/pitch offsets from projected_gravity.

    Convention matches Unitree/Isaac RPY: when level robot has positive roll_bias,
    g_y = -sin(roll_bias); positive pitch_bias gives g_x = -sin(pitch_bias).
    """
    if roll_bias == 0.0 and pitch_bias == 0.0:
        return g
    gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
    if roll_bias != 0.0:
        cr, sr = np.cos(roll_bias), np.sin(roll_bias)
        gy, gz = cr * gy - sr * gz, sr * gy + cr * gz
    if pitch_bias != 0.0:
        cp, sp = np.cos(-pitch_bias), np.sin(-pitch_bias)
        gx, gz = cp * gx + sp * gz, -sp * gx + cp * gz
    return np.array([gx, gy, gz], dtype=np.float32)


def build_observation(state, command):
    """state: driver RobotState (SDK joint order). command: (vx, vy, wz) in the body frame."""
    q = contract.sdk_to_policy(np.asarray(state.q, dtype=np.float32))
    dq = contract.sdk_to_policy(np.asarray(state.dq, dtype=np.float32))
    return np.concatenate(
        [
            np.asarray(state.gyro, dtype=np.float32),  # base_ang_vel
            projected_gravity(state.quat),
            np.asarray(command, dtype=np.float32),  # velocity_commands
            q,  # joint_pos
            dq,  # joint_vel
        ]
    )


def build_observation_promo_v2(state, command, last_action, default_q=contract.DEFAULT_JOINT_POS_PROMO_V2):
    """45-D observation layout for PROMO v2 / base_v2 / oursA.

    [base_ang_vel_body(3), projected_gravity(3), command_vx_vy_wz(3),
     joint_pos_minus_default(12), joint_vel(12), last_action(12)]
    """
    q = contract.sdk_to_policy(np.asarray(state.q, dtype=np.float32))
    dq = contract.sdk_to_policy(np.asarray(state.dq, dtype=np.float32))
    last_act = np.asarray(last_action, dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(state.gyro, dtype=np.float32),
            projected_gravity(state.quat),
            np.asarray(command, dtype=np.float32),
            q - default_q,
            dq,
            last_act,
        ]
    )


class History(object):
    """(x, a) for the last HIST steps, oldest first. As train.py: pushed after the action, so phi(buf) at step t
    has seen up to (x_{t-1}, a_{t-1}). Training resets to zeros; see fill() for what deployment starts from."""

    def __init__(self, length=contract.HIST, dim=contract.HIST_DIM):
        self.buf = np.zeros((length, dim), dtype=np.float32)

    def reset(self):
        self.buf[:] = 0.0

    def fill(self, x):
        """Every slot is (x, a = 0). Unlike training's zeros: on the robot zeros gave z_hat[0] = 4.96 against about
        -0.5 once settled, which threw the front hips outward for the first 0.1 s."""
        self.buf[:] = np.concatenate([x, np.zeros(self.buf.shape[1] - len(x), dtype=np.float32)])

    def push(self, x, a):
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = np.concatenate([x, a])
        return self.buf


class PromoHistory(object):
    """History buffer for PROMO v2 models (base_v2, oursA).
    Buffer shape: (H, obs_dim), oldest first.
    According to policy_meta.json:
    - At start and after every reset, fill with obs repeated H times.
    - Updated per step: hist = concat(hist[1:], obs) BEFORE calling the policy, so hist[-1] == obs.
    """

    def __init__(self, length=contract.HIST_LEN_PROMO_V2, dim=contract.OBS_DIM_PROMO_V2):
        self.length = length
        self.dim = dim
        self.buf = np.zeros((length, dim), dtype=np.float32)

    def reset(self):
        self.buf[:] = 0.0

    def fill(self, obs):
        self.buf[:] = np.asarray(obs, dtype=np.float32)

    def push_obs(self, obs):
        """Roll and append obs before calling policy."""
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = np.asarray(obs, dtype=np.float32)
        return self.buf

