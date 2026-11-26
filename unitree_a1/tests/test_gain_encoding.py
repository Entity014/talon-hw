"""PD gain encoding tests — Kd 0.8 becomes 0.75 on the wire (2026-11-06).

SDK encodes Kp as int16 × 32, Kd as int16 × 16 (truncated).
Kp 55 → 1760, Kd 0.8 → 12 → 0.75 actual.
"""

import numpy as np
import pytest

from talon_a1 import contract


def test_kp_encoding():
    """Kp is multiplied by 32 and truncated to int16."""
    kp = contract.KP  # 55.0
    
    # Encoding
    kp_wire = int(kp * 32)
    assert kp_wire == 1760
    
    # Decoding
    kp_decoded = kp_wire / 32.0
    assert kp_decoded == 55.0


def test_kd_encoding_truncates():
    """Kd 0.8 → 12 → 0.75 due to int16 truncation."""
    kd = contract.KD  # 0.8
    
    # Encoding (truncation, not rounding)
    kd_wire = int(kd * 16)
    assert kd_wire == 12  # 0.8 × 16 = 12.8 → 12
    
    # Decoding
    kd_actual = kd_wire / 16.0
    assert kd_actual == 0.75  # not 0.8!


def test_kd_rounding_error():
    """Document the Kd rounding error."""
    requested = 0.8
    actual = int(requested * 16) / 16.0
    
    error = abs(requested - actual)
    assert error == pytest.approx(0.05)  # 6.25% error


def test_stand_gains_encoding():
    """Standing gains encode cleanly."""
    kp = contract.STAND_KP  # 50.0
    kd = contract.STAND_KD  # 3.0
    
    # Encoding
    kp_wire = int(kp * 32)
    kd_wire = int(kd * 16)
    
    assert kp_wire == 1600
    assert kd_wire == 48
    
    # Decoding (no error)
    assert kp_wire / 32.0 == 50.0
    assert kd_wire / 16.0 == 3.0


def test_gain_encoding_range():
    """int16 × 32/16 gives reasonable gain range."""
    # int16 range: -32768 to 32767
    max_kp = 32767 / 32.0
    max_kd = 32767 / 16.0
    
    assert max_kp > 1000  # plenty of headroom
    assert max_kd > 2000
    
    # Policy gains fit comfortably
    assert contract.KP < max_kp
    assert contract.KD < max_kd


def test_gain_resolution():
    """Document gain resolution from integer encoding."""
    kp_resolution = 1.0 / 32.0
    kd_resolution = 1.0 / 16.0
    
    assert kp_resolution == 0.03125  # ~3% at Kp=1
    assert kd_resolution == 0.0625   # ~6% at Kd=1


def test_trained_gain_range():
    """Policy trained with Kp [50, 60], Kd [0.4, 0.8]."""
    trained_kp_range = (50.0, 60.0)
    trained_kd_range = (0.4, 0.8)
    
    # Nominal gains are inside trained range
    assert trained_kp_range[0] <= contract.KP <= trained_kp_range[1]
    assert trained_kd_range[0] <= contract.KD <= trained_kd_range[1]


def test_kd_values_near_0_8():
    """Compare Kd values near 0.8 after encoding."""
    test_values = [0.7, 0.75, 0.8, 0.85, 0.9]
    
    results = {}
    for kd in test_values:
        kd_wire = int(kd * 16)
        kd_actual = kd_wire / 16.0
        results[kd] = (kd_wire, kd_actual)
    
    # 0.75 encodes perfectly, 0.8 doesn't
    assert results[0.75] == (12, 0.75)
    assert results[0.8] == (12, 0.75)  # same as 0.75!
    assert results[0.85] == (13, 0.8125)


def test_policy_kp_limit():
    """Controller has policy_kp limit (default 60) to prevent runaway."""
    # From test_controller.py test_bring_up_limits_reach_the_motor_and_the_history
    # policy_kp limits Kp scaling during walk
    max_policy_kp = 60.0
    
    # Nominal Kp is below limit
    assert contract.KP <= max_policy_kp


def test_sport_mode_gains():
    """Sport mode uses low Kp + feedforward, policy uses high Kp."""
    # From 2026-11-06 log: Sport mode Kp 1-3, Kd 1-3
    # Policy mode: Kp 55, Kd 0.75
    
    sport_kp_range = (1.0, 3.0)
    policy_kp = contract.KP
    
    # Policy Kp is much higher
    assert policy_kp > sport_kp_range[1] * 10


def test_gain_encoding_roundtrip():
    """Test encode/decode roundtrip for various gains."""
    test_kp = [1.0, 10.0, 50.0, 55.0, 100.0]
    test_kd = [0.1, 0.5, 0.8, 1.0, 5.0]
    
    for kp in test_kp:
        kp_wire = int(kp * 32)
        kp_decoded = kp_wire / 32.0
        # Check if roundtrip is close (within resolution)
        assert abs(kp - kp_decoded) <= 1.0/32.0
    
    for kd in test_kd:
        kd_wire = int(kd * 16)
        kd_decoded = kd_wire / 16.0
        # Check if roundtrip is close (within resolution)
        assert abs(kd - kd_decoded) <= 1.0/16.0


def test_measured_gains_from_robot():
    """Document gains measured from actual robot packets (2026-11-06)."""
    # Measured from robot: Kp wire 1760, Kd wire 12
    measured_kp_wire = 1760
    measured_kd_wire = 12
    
    kp_actual = measured_kp_wire / 32.0
    kd_actual = measured_kd_wire / 16.0
    
    assert kp_actual == 55.0
    assert kd_actual == 0.75


def test_zero_gains():
    """Zero gains encode and decode correctly."""
    kp_zero = 0.0
    kd_zero = 0.0
    
    assert int(kp_zero * 32) == 0
    assert int(kd_zero * 16) == 0


def test_negative_gains_not_used():
    """Policy doesn't use negative gains."""
    # Gains are always non-negative
    assert contract.KP >= 0
    assert contract.KD >= 0
    assert contract.STAND_KP >= 0
    assert contract.STAND_KD >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
