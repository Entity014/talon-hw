"""45-D policy observation o_t from a robot state, term for term as PromoObservationsCfg.PolicyCfg.

Isaac Lab applies no scaling to these terms; the training noise is a training-time corruption only.
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


class ObservationBuilder(object):
    def __init__(self):
        self.last_action = np.zeros(contract.ACT_DIM, dtype=np.float32)

    def reset(self):
        self.last_action[:] = 0.0

    def build(self, state, command):
        """state: driver RobotState (SDK joint order). command: (vx, vy, wz) in the body frame."""
        q = contract.sdk_to_policy(np.asarray(state.q, dtype=np.float32)) - contract.DEFAULT_JOINT_POS
        dq = contract.sdk_to_policy(np.asarray(state.dq, dtype=np.float32))
        return np.concatenate(
            [
                np.asarray(state.gyro, dtype=np.float32),  # base_ang_vel
                projected_gravity(state.quat),
                np.asarray(command, dtype=np.float32),  # velocity_commands
                q,  # joint_pos_rel
                dq,  # joint_vel_rel (default joint velocity is 0)
                self.last_action,  # actions
            ]
        )


class History(object):
    """o_{t-H+1..t}, oldest first. Reset fills every slot with o_0, as train.py does on episode reset."""

    def __init__(self, length=contract.HIST, dim=contract.OBS_DIM):
        self.buf = np.zeros((length, dim), dtype=np.float32)
        self.empty = True

    def reset(self):
        self.empty = True

    def push(self, o):
        if self.empty:
            self.buf[:] = o
            self.empty = False
        else:
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = o
        return self.buf
