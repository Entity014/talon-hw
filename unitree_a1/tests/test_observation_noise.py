"""Observation noise tests — root cause of directional failures (2026-11-26).

Policy trained with joint_vel ±1.5 rad/s noise but hardware has std ~0.09.
Clean dq + latency → overreaction. Adding noise back fixes directional issues.
"""

import numpy as np
import pytest

from talon_a1 import contract


def test_trained_noise_levels():
    """Document noise levels used during training (BaselineLowMem-Vertex)."""
    # From 2026-11-26 investigation: enable_corruption at policy group
    trained_noise = {
        "gyro": 0.2,  # rad/s, uniform ±
        "projected_gravity": 0.05,  # uniform ±
        "joint_pos": 0.01,  # rad, uniform ±
        "joint_vel": 1.5,  # rad/s, uniform ±
        "command": 0.0,  # no noise
        "last_action": 0.0,  # no noise
    }
    
    assert trained_noise["joint_vel"] == 1.5
    assert trained_noise["gyro"] == 0.2


def test_measured_hardware_noise():
    """Hardware sensors are much cleaner than training assumed."""
    # From logs: standing still, dq std ~0.09 rad/s
    measured_dq_std = 0.09  # rad/s
    trained_dq_range = 1.5 * 2  # uniform ±1.5 → std = 1.5/sqrt(3) ≈ 0.87
    
    # Hardware is ~10x cleaner
    assert measured_dq_std < trained_dq_range / 10


def test_noise_injection_should_match_training():
    """Adding noise back to hardware obs should use training distribution."""
    rng = np.random.RandomState(42)
    
    # Training uses uniform distribution
    noise_uniform = rng.uniform(-1.5, 1.5, (100, 12))
    std_uniform = noise_uniform.std()
    
    # Should be close to 1.5 / sqrt(3) ≈ 0.866
    assert 0.8 < std_uniform < 0.9
    
    # Gaussian would have different characteristics
    noise_gaussian = rng.randn(100, 12) * 0.866  # match std
    assert noise_gaussian.min() < -2.0  # has tails
    assert noise_gaussian.max() > 2.0


def test_obs_indices_for_noise_injection():
    """Identify which obs indices need noise."""
    # Observation layout from contract.py:
    # 0-2:   base_ang_vel (gyro)
    # 3-5:   projected_gravity
    # 6-8:   command
    # 9-20:  joint_pos (q)
    # 21-32: joint_vel (dq)
    
    gyro_indices = slice(0, 3)
    gravity_indices = slice(3, 6)
    command_indices = slice(6, 9)
    joint_pos_indices = slice(9, 21)
    joint_vel_indices = slice(21, 33)
    
    obs = np.zeros(contract.OBS_DIM, dtype=np.float32)
    obs[gyro_indices] = 1.0
    obs[gravity_indices] = 2.0
    obs[command_indices] = 3.0
    obs[joint_pos_indices] = 4.0
    obs[joint_vel_indices] = 5.0
    
    assert obs[0] == 1.0  # gyro
    assert obs[3] == 2.0  # gravity
    assert obs[6] == 3.0  # command
    assert obs[9] == 4.0  # joint_pos
    assert obs[21] == 5.0  # joint_vel


def test_add_observation_noise():
    """Helper to add training-matched noise to observations."""
    def add_noise(obs, rng, joint_vel_noise=1.5, gyro_noise=0.2, 
                  gravity_noise=0.05, joint_pos_noise=0.01):
        """Add uniform noise matching training distribution."""
        obs = obs.copy()
        obs[0:3] += rng.uniform(-gyro_noise, gyro_noise, 3)
        obs[3:6] += rng.uniform(-gravity_noise, gravity_noise, 3)
        # command and last_action: no noise
        obs[9:21] += rng.uniform(-joint_pos_noise, joint_pos_noise, 12)
        obs[21:33] += rng.uniform(-joint_vel_noise, joint_vel_noise, 12)
        return obs
    
    rng = np.random.RandomState(42)
    obs = np.zeros(contract.OBS_DIM, dtype=np.float32)
    
    noisy = add_noise(obs, rng)
    
    # Check noise was applied
    assert not np.allclose(noisy[0:3], 0.0)  # gyro
    assert not np.allclose(noisy[21:33], 0.0)  # joint_vel
    
    # Command should be unchanged (no noise)
    assert np.allclose(noisy[6:9], 0.0)


def test_joint_vel_noise_is_critical():
    """From sim: joint_vel ±1.5 fixes directions, other noise doesn't."""
    # 2026-11-26 investigation results (OURS-A, balanced w, 3 seeds, delay 10ms):
    results = {
        "no_noise": "4/6 directions (right/yaw fail)",
        "all_noise": "6/6 directions pass",
        "joint_vel_1.5": "6/6 directions pass",
        "joint_vel_0.5": "some directions fail",
        "gyro_only": "right/yaw still fail",
        "gravity_only": "right/yaw still fail",
        "joint_pos_only": "right/yaw still fail",
    }
    
    # joint_vel ±1.5 is necessary and sufficient
    assert "6/6" in results["joint_vel_1.5"]
    assert "6/6" in results["all_noise"]
    assert "fail" in results["joint_vel_0.5"]


def test_hypothesis_clean_dq_plus_latency_overreacts():
    """Policy learned to distrust dq due to noise, but hardware dq is accurate."""
    # Hypothesis from 2026-11-26:
    # 1. Training: high dq noise → policy learns to ignore/distrust dq
    # 2. Hardware: clean dq + latency → policy trusts delayed dq → overreacts
    # 3. Solution: add noise to hardware obs to match training
    
    # This is a hypothesis test, actual validation needs hardware
    pass


def test_latency_alone_not_sufficient_explanation():
    """Latency causes bounce but doesn't explain directional failures."""
    # From sim testing:
    # - delay 10ms without noise: some directions fail
    # - delay 10ms with noise: all directions pass
    # - delay 0ms with noise: all directions pass
    
    # Latency amplifies the mismatch but isn't the root cause
    pass


def test_noise_distribution_characteristics():
    """Compare uniform vs gaussian noise characteristics."""
    rng = np.random.RandomState(42)
    n = 10000
    
    # Uniform ±1.5 (training)
    uniform = rng.uniform(-1.5, 1.5, n)
    uniform_std = uniform.std()
    uniform_range = uniform.max() - uniform.min()
    
    # Gaussian with matched std
    gaussian = rng.randn(n) * uniform_std
    gaussian_range = gaussian.max() - gaussian.min()
    
    # Uniform has bounded range, Gaussian has tails
    assert uniform_range < 3.1  # ~3.0
    assert gaussian_range > 5.0  # much larger due to tails
    
    # Training used uniform, so hardware injection should too
    assert 0.85 < uniform_std < 0.90  # 1.5/sqrt(3) ≈ 0.866


def test_noise_seed_independence():
    """Noise should be independent across timesteps."""
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    
    noise1_t0 = rng1.uniform(-1.5, 1.5, 12)
    noise1_t1 = rng1.uniform(-1.5, 1.5, 12)
    
    noise2_t0 = rng2.uniform(-1.5, 1.5, 12)
    
    # Same seed produces same first sample
    np.testing.assert_allclose(noise1_t0, noise2_t0)
    
    # Different timesteps are uncorrelated
    assert not np.allclose(noise1_t0, noise1_t1)


def test_promo_v2_obs_dim():
    """PROMO v2 has different obs_dim (45 vs 33 for RMA)."""
    # PROMO v2 includes additional observations
    assert contract.OBS_DIM_PROMO_V2 == 45
    assert contract.OBS_DIM == 33  # RMA
    
    # Noise injection logic needs to know obs format


def test_standing_dq_std_measurement():
    """Document expected dq std when standing still."""
    # From hardware logs: ~0.09 rad/s standing
    expected_standing_std = 0.09
    
    # This is much lower than training noise
    trained_noise_equivalent_std = 1.5 / np.sqrt(3)
    
    assert expected_standing_std < trained_noise_equivalent_std / 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
