"""Policy <-> robot interface of deployment RMA on A1 (talon-rl branch deployment, task TalonRl-PromoRma-A1).

Every value is copied from the training side; the source is named next to it. scripts/export_policy.py
writes the same values into the policy file and RmaPolicy refuses a file whose values differ.
"""

import numpy as np

# Isaac Lab articulation order of the A1 asset, joint-type grouped
# (talon-rl tasks/rma/config/a1/env_cfg.py POLICY_JOINT_NAMES).
POLICY_JOINTS = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
)
# unitree_legged_sdk order, leg grouped (quadruped.h FR_0 .. RL_2)
SDK_JOINTS = (
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)
SDK_INDEX_OF_POLICY = np.array([SDK_JOINTS.index(n) for n in POLICY_JOINTS])
POLICY_INDEX_OF_SDK = np.array([POLICY_JOINTS.index(n) for n in SDK_JOINTS])


def sdk_to_policy(x):
    return np.asarray(x)[SDK_INDEX_OF_POLICY]


def policy_to_sdk(x):
    return np.asarray(x)[POLICY_INDEX_OF_SDK]


# env_cfg.py HIP_THIGH_CALF_DEFAULT, policy order
DEFAULT_JOINT_POS = np.array(
    [0.05, -0.05, 0.05, -0.05, 0.8, 0.8, 0.8, 0.8, -1.4, -1.4, -1.4, -1.4], dtype=np.float32
)
# env_cfg.py ACTION_SCALE_BASELINE, use_default_offset=True
ACTION_SCALE = 0.4
# env_cfg.py RmaA1EnvCfg nominal gains, inside the trained Kp [50, 60], Kd [0.4, 0.8]. The SDK puts them on the
# wire as int16 Kp * 32 and Kd * 16 (truncated), measured on the robot: 55 -> 1760, 0.8 -> 12, which is Kd 0.75.
KP = 55.0
KD = 0.8
# Standing up, not part of the policy: copied from A1_sport_1's packets on the robot, which stand it up with raw
# gains 1600 / 48 and go through this crouch first, so the feet do not scrape.
STAND_KP = 50.0
STAND_KD = 3.0
CROUCH_JOINT_POS = np.array(
    [0.0, 0.0, 0.0, 0.0, 1.4, 1.4, 1.4, 1.4, -2.58, -2.58, -2.58, -2.58], dtype=np.float32
)
LYING_CALF = -2.0  # mean calf angle below this means the legs are folded under the body
# env_cfg.py PromoRmaA1EnvCfg: sim.dt 0.005 x decimation 4
POLICY_DT = 0.02
# env_cfg.py PromoRmaCommandsCfg ranges: lin_vel_x, lin_vel_y, ang_vel_z
COMMAND_LIMITS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

# talon_rl.rma.networks and the phase2_500 run manifest
OBS_DIM = 33  # ang_vel 3 + gravity 3 + cmd 3 + q 12 + dq 12
ACT_DIM = 12
HIST = 50
HIST_DIM = OBS_DIM + ACT_DIM  # one history step: (x_t, a_t)
VEL_HIST = 5
LATENT = 8
VEL_DIM = 3


def as_dict():
    """The values export_policy.py stores and RmaPolicy checks."""
    return {
        "policy_joints": list(POLICY_JOINTS),
        "default_joint_pos": [float(v) for v in DEFAULT_JOINT_POS],
        "action_scale": ACTION_SCALE,
        "kp": KP,
        "kd": KD,
        "policy_dt": POLICY_DT,
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "hist": HIST,
        "vel_hist": VEL_HIST,
        "latent": LATENT,
        "vel_dim": VEL_DIM,
    }
