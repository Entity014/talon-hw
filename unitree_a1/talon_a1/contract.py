"""Policy <-> robot interface of PROMO-on-A1 (talon-rl branch research/v1, task TalonRl-Promo-A1).

Every value is copied from the training side; the source is named next to it. scripts/export_policy.py
writes the same values into the policy file and PromoPolicy refuses a file whose values differ.
"""

import numpy as np

# Isaac Lab articulation order of the A1 asset, joint-type grouped
# (talon-rl scripts/promo/verify_mujoco_contract.py JOINTS).
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


# isaaclab_assets UNITREE_A1_CFG.init_state.joint_pos, policy order
DEFAULT_JOINT_POS = np.array(
    [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5], dtype=np.float32
)
# UnitreeA1RoughEnvCfg: actions.joint_pos.scale, use_default_offset=True
ACTION_SCALE = 0.25
# UNITREE_A1_CFG DCMotorCfg stiffness / damping
KP = 25.0
KD = 0.5
# velocity_env_cfg: sim.dt 0.005 x decimation 4
POLICY_DT = 0.02
# velocity_env_cfg CommandsCfg ranges: lin_vel_x, lin_vel_y, ang_vel_z (heading_command off in PROMO)
COMMAND_LIMITS = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

# talon_rl.promo.networks.PromoDims
OBS_DIM = 45  # ang_vel 3 + gravity 3 + cmd 3 + q 12 + dq 12 + last action 12
ACT_DIM = 12
HIST = 10
VEL_HIST = 5
LATENT = 45
PREF_DIM = 3  # tracking, stability, efficiency

# talon_rl.promo.preference.ANCHORS
PREFERENCE_ANCHORS = {
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "tracking": (0.8, 0.1, 0.1),
    "stability": (0.1, 0.8, 0.1),
    "efficiency": (0.1, 0.1, 0.8),
    "track_stab": (0.45, 0.45, 0.1),
    "track_eff": (0.45, 0.1, 0.45),
    "stab_eff": (0.1, 0.45, 0.45),
}


def as_dict():
    """The values export_policy.py stores and PromoPolicy checks."""
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
        "pref_dim": PREF_DIM,
    }
