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
