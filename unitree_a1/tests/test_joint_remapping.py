"""Joint remapping tests — policy order vs SDK order.

Policy: FL/FR/RL/RR grouped by joint type (hip, thigh, calf)
SDK: FR/FL/RR/RL grouped by leg (all joints per leg)
"""

import numpy as np
import pytest

from talon_a1 import contract


def test_policy_joint_order():
    """Policy uses joint-type grouped order."""
    expected = (
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    )
    
    assert contract.POLICY_JOINTS == expected
    assert len(contract.POLICY_JOINTS) == 12


def test_sdk_joint_order():
    """SDK uses leg-grouped order (FR, FL, RR, RL)."""
    expected = (
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    )
    
    assert contract.SDK_JOINTS == expected
    assert len(contract.SDK_JOINTS) == 12


def test_index_mapping_arrays():
    """Verify index mapping arrays are correct."""
    # SDK_INDEX_OF_POLICY: policy index → SDK index
    assert len(contract.SDK_INDEX_OF_POLICY) == 12
    
    # POLICY_INDEX_OF_SDK: SDK index → policy index
    assert len(contract.POLICY_INDEX_OF_SDK) == 12


def test_sdk_to_policy_conversion():
    """Convert SDK order to policy order."""
    # SDK order: FR FL RR RL, each leg has hip/thigh/calf
    sdk_values = np.arange(12, dtype=np.float32)  # 0-11
    
    policy_values = contract.sdk_to_policy(sdk_values)
    
    # FL hip (policy idx 0) = SDK idx 3
    assert policy_values[0] == sdk_values[3]
    
    # FR hip (policy idx 1) = SDK idx 0
    assert policy_values[1] == sdk_values[0]
    
    # RL hip (policy idx 2) = SDK idx 9
    assert policy_values[2] == sdk_values[9]
    
    # RR hip (policy idx 3) = SDK idx 6
    assert policy_values[3] == sdk_values[6]


def test_policy_to_sdk_conversion():
    """Convert policy order to SDK order."""
    # Policy order: hip×4, thigh×4, calf×4
    policy_values = np.arange(12, dtype=np.float32)  # 0-11
    
    sdk_values = contract.policy_to_sdk(policy_values)
    
    # FR hip (SDK idx 0) = policy idx 1
    assert sdk_values[0] == policy_values[1]
    
    # FL hip (SDK idx 3) = policy idx 0
    assert sdk_values[3] == policy_values[0]
    
    # RR hip (SDK idx 6) = policy idx 3
    assert sdk_values[6] == policy_values[3]
    
    # RL hip (SDK idx 9) = policy idx 2
    assert sdk_values[9] == policy_values[2]


def test_roundtrip_conversion():
    """Converting SDK→policy→SDK should be identity."""
    original_sdk = np.random.randn(12).astype(np.float32)
    
    policy = contract.sdk_to_policy(original_sdk)
    back_to_sdk = contract.policy_to_sdk(policy)
    
    np.testing.assert_allclose(back_to_sdk, original_sdk)


def test_roundtrip_policy_first():
    """Converting policy→SDK→policy should be identity."""
    original_policy = np.random.randn(12).astype(np.float32)
    
    sdk = contract.policy_to_sdk(original_policy)
    back_to_policy = contract.sdk_to_policy(sdk)
    
    np.testing.assert_allclose(back_to_policy, original_policy)


def test_default_pose_in_policy_order():
    """DEFAULT_JOINT_POS is in policy order."""
    default = contract.DEFAULT_JOINT_POS
    
    # First 4 are hips
    fl_hip, fr_hip, rl_hip, rr_hip = default[0:4]
    
    # Left legs positive, right legs negative
    assert fl_hip > 0
    assert fr_hip < 0
    assert rl_hip > 0
    assert rr_hip < 0
    
    # Thighs are 0.8
    assert np.allclose(default[4:8], 0.8)
    
    # Calves are -1.4
    assert np.allclose(default[8:12], -1.4)


def test_promo_v2_default_pose_in_policy_order():
    """PROMO v2 default pose is in policy order."""
    default = contract.DEFAULT_JOINT_POS_PROMO_V2
    
    # First 4 are hips
    fl_hip, fr_hip, rl_hip, rr_hip = default[0:4]
    
    # FL/RL positive, FR/RR negative
    assert fl_hip == 0.1
    assert fr_hip == -0.1
    assert rl_hip == 0.1
    assert rr_hip == -0.1
    
    # Front thighs 0.8, rear thighs 1.0
    assert default[4] == 0.8  # FL
    assert default[5] == 0.8  # FR
    assert default[6] == 1.0  # RL
    assert default[7] == 1.0  # RR
    
    # All calves -1.5
    assert np.allclose(default[8:12], -1.5)


def test_index_mapping_is_permutation():
    """Index mappings should be permutations (bijective)."""
    # SDK_INDEX_OF_POLICY should contain each index 0-11 exactly once
    sdk_indices = list(contract.SDK_INDEX_OF_POLICY)
    assert sorted(sdk_indices) == list(range(12))
    
    # POLICY_INDEX_OF_SDK should contain each index 0-11 exactly once
    policy_indices = list(contract.POLICY_INDEX_OF_SDK)
    assert sorted(policy_indices) == list(range(12))


def test_specific_joint_mappings():
    """Test specific joint name mappings."""
    # FL_hip is at policy index 0
    assert contract.POLICY_JOINTS[0] == "FL_hip_joint"
    
    # Find FL_hip in SDK order
    fl_hip_sdk_idx = contract.SDK_JOINTS.index("FL_hip_joint")
    assert fl_hip_sdk_idx == 3  # fourth position in SDK
    
    # Verify index mapping
    assert contract.SDK_INDEX_OF_POLICY[0] == 3
    assert contract.POLICY_INDEX_OF_SDK[3] == 0


def test_leg_names():
    """Test leg name consistency."""
    legs = ["FL", "FR", "RL", "RR"]
    joints = ["hip", "thigh", "calf"]
    
    # Check all combinations exist
    for leg in legs:
        for joint in joints:
            joint_name = f"{leg}_{joint}_joint"
            assert joint_name in contract.POLICY_JOINTS
            assert joint_name in contract.SDK_JOINTS


def test_crouch_pose_in_policy_order():
    """CROUCH_JOINT_POS (stand-up intermediate) is in policy order."""
    crouch = contract.CROUCH_JOINT_POS
    
    # Hips at zero
    assert np.allclose(crouch[0:4], 0.0)
    
    # Thighs tucked: 1.4
    assert np.allclose(crouch[4:8], 1.4)
    
    # Calves tucked: -2.58
    assert np.allclose(crouch[8:12], -2.58)


def test_lying_detection():
    """LYING_CALF threshold detects folded legs."""
    crouch_calf_mean = contract.CROUCH_JOINT_POS[8:12].mean()
    
    # Crouch calf is below lying threshold
    assert crouch_calf_mean < contract.LYING_CALF


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
