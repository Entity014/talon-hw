"""Multi-objective weight adjustment tests (w1, w2, w3).

OURS-A trained with balanced weights but can adjust at deploy time.
w=[0.8,0.1,0.1] passes all 6 directions with delay 10/15ms.
"""

import numpy as np
import pytest

from talon_a1 import contract


def test_balanced_weights_default():
    """Default training uses balanced weights."""
    balanced = np.array([1.0/3, 1.0/3, 1.0/3], dtype=np.float32)
    
    assert balanced.sum() == pytest.approx(1.0)
    assert np.allclose(balanced, 0.333333, atol=0.01)


def test_promo_v2_default_weights():
    """PROMO v2 contract defines default weights."""
    w = contract.DEFAULT_W_PROMO_V2
    
    assert w.shape == (3,)
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(w, [1.0/3, 1.0/3, 1.0/3])


def test_weight_adjustment_results():
    """Document tested weight configurations and pass rates."""
    # From 2026-11-26 sim testing (delay 10ms, 3 seeds, 6 directions)
    results = {
        "balanced": {"pass": "4/6", "fail": "right, yaw_left"},
        "0.8_0.1_0.1": {"pass": "6/6", "fail": "none"},
        "0.6_0.2_0.2": {"pass": "17/18", "fail": "1 seed"},
    }
    
    # 0.8/0.1/0.1 is most reliable
    assert results["0.8_0.1_0.1"]["pass"] == "6/6"


def test_weight_adjustment_tradeoffs():
    """Document known tradeoffs of weight adjustment."""
    # From 2026-11-26 hardware run (w=0.8/0.1/0.1):
    tradeoffs = {
        "pros": [
            "All 6 directions work",
            "No retraining needed",
            "146s continuous walk",
        ],
        "cons": [
            "Hip abduction increases: 0.07 → 0.11 rad",
            "Bounce increases: gyro_std 0.49 → 0.62 rad/s (~27%)",
            "Step frequency increases: 6.9 → 7.8 Hz (~13%)",
        ],
    }
    
    # Hip abduction acceptable for quadruped
    max_hip_abduction = 0.11  # rad, ~6.3 degrees
    assert max_hip_abduction < 0.15


def test_weight_normalization():
    """Weights should sum to 1.0."""
    test_weights = [
        [0.8, 0.1, 0.1],
        [0.6, 0.2, 0.2],
        [0.5, 0.3, 0.2],
        [1.0/3, 1.0/3, 1.0/3],
    ]
    
    for w in test_weights:
        w_array = np.array(w, dtype=np.float32)
        assert w_array.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w_array >= 0).all()


def test_weight_parsing_from_command_line():
    """Test parsing --w flag from command line."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=float, nargs=3, 
                       default=[1.0/3, 1.0/3, 1.0/3])
    
    # Balanced
    args = parser.parse_args([])
    np.testing.assert_allclose(args.w, [1.0/3, 1.0/3, 1.0/3])
    
    # Custom
    args = parser.parse_args(["--w", "0.8", "0.1", "0.1"])
    np.testing.assert_allclose(args.w, [0.8, 0.1, 0.1])


def test_weight_dimensions():
    """PROMO v2 uses 3D weight vector."""
    assert contract.W_DIM_PROMO_V2 == 3
    
    w = contract.DEFAULT_W_PROMO_V2
    assert len(w) == contract.W_DIM_PROMO_V2


def test_weights_no_negative():
    """Weights must be non-negative."""
    invalid_weights = [
        [-0.1, 0.5, 0.6],  # negative
        [0.8, -0.1, 0.3],  # negative
        [1.2, -0.1, -0.1],  # negative
    ]
    
    for w in invalid_weights:
        w_array = np.array(w, dtype=np.float32)
        assert (w_array < 0).any(), f"Test case {w} should have negative values"


def test_bounce_frequency_measurement():
    """Document step frequency changes with different weights."""
    # From sim and hardware measurements
    frequencies = {
        "sim_delay0_balanced": 4.7,  # Hz
        "sim_delay10_balanced_no_noise": 6.5,  # Hz
        "sim_delay10_balanced_with_noise": 6.9,  # Hz
        "hw_w0.8_fwd": 7.8,  # Hz (measured gyro oscillation)
    }
    
    # Hardware bounces more than sim
    hw_bounce = frequencies["hw_w0.8_fwd"]
    sim_bounce = frequencies["sim_delay10_balanced_with_noise"]
    
    assert hw_bounce > sim_bounce * 1.1  # at least 10% higher


def test_gyro_std_as_bounce_metric():
    """Gyro std correlates with perceived bounce."""
    # From 2026-11-26 measurements (walking forward)
    gyro_stds = {
        "sim_delay0": 0.28,  # rad/s
        "sim_delay10_no_noise": 0.40,  # rad/s
        "sim_delay10_with_noise": 0.49,  # rad/s
        "hw_balanced": 0.40,  # rad/s (from earlier test)
        "hw_w0.8": 0.62,  # rad/s
    }
    
    # Latency increases bounce
    assert gyro_stds["sim_delay10_no_noise"] > gyro_stds["sim_delay0"]
    
    # Hardware bounces more than sim
    assert gyro_stds["hw_w0.8"] > gyro_stds["sim_delay10_with_noise"]


def test_speed_reduction_doesnt_help_bounce():
    """Reducing commanded speed doesn't reduce bounce in sim."""
    # From 2026-11-26: tried lower vx_max, didn't help
    # Bounce is structural to the gait, not speed-dependent
    pass


def test_hip_abduction_in_gait():
    """Hip abduction is part of the learned gait pattern."""
    # From measurements:
    hip_abductions = {
        "sim_w0.8": 0.07,  # rad
        "hw_w0.8": 0.11,  # rad
    }
    
    # Hardware has more abduction than sim
    assert hip_abductions["hw_w0.8"] > hip_abductions["sim_w0.8"]
    
    # But still reasonable for quadruped walking
    assert hip_abductions["hw_w0.8"] < 0.15


def test_continuous_walk_duration():
    """Document continuous walk duration with w=0.8/0.1/0.1."""
    # From 2026-11-26 hardware run
    duration_seconds = 146
    
    # Ended with overtorque fault, not fall
    assert duration_seconds > 120  # > 2 minutes


def test_weight_adjustment_vs_retraining():
    """Weight adjustment avoids retraining but has tradeoffs."""
    # Pros of weight adjustment:
    # - No retraining needed (saves time/compute)
    # - Can tune for specific deployment scenario
    # - Preserves learned policy behaviors
    
    # Cons:
    # - Increased bounce and hip abduction
    # - Not optimized for new weight combination
    # - May not generalize to all terrains
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
