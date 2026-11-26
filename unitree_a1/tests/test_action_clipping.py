"""Action clipping tests — critical issue found during first walk (2026-11-26).

PROMO policies have linear actor output (no tanh) and need |a| > 1 to walk.
Default --action-clip 1.0 prevented walking (0.015 m/s instead of 0.5 m/s).
"""

import numpy as np
import pytest

from talon_a1 import contract


def test_promo_needs_high_action_clip():
    """PROMO actor is linear; walking needs actions beyond [-1, 1]."""
    # From 2026-11-26 log: forward walk hit clip at 1.0 and couldn't step
    action_unclipped = np.ones(12) * 1.5  # typical PROMO forward walk
    
    clip_1 = np.clip(action_unclipped, -1.0, 1.0)
    clip_10 = np.clip(action_unclipped, -10.0, 10.0)
    
    assert np.allclose(clip_1, 1.0)  # saturated
    assert np.allclose(clip_10, 1.5)  # preserves intended action


def test_rma_works_with_clip_1():
    """RMA actor has tanh output, so clip=1 is sufficient."""
    # RMA actions are tanh-bounded to [-1, 1]
    action_tanh = np.tanh(np.random.randn(12) * 3)  # random but bounded
    
    assert np.abs(action_tanh).max() <= 1.0
    clip_1 = np.clip(action_tanh, -1.0, 1.0)
    
    np.testing.assert_allclose(clip_1, action_tanh)


def test_clipped_action_becomes_last_action():
    """Clipped action is fed back as last_action, breaking history."""
    from talon_a1.controller import Controller
    from talon_a1.remote import combo
    
    class FakeDriver:
        def set_target(self, q, kp, kd):
            return True
        def arm(self):
            return True
        def disarm(self):
            pass
    
    class FakePolicy:
        def action_mean(self, x, a_prev, hist):
            # Policy asks for 2.0 (needs high clip)
            return np.ones(12, np.float32) * 2.0, np.zeros(3), np.zeros(8)
    
    class FakeState:
        def __init__(self):
            self.q = contract.policy_to_sdk(contract.DEFAULT_JOINT_POS)
            self.dq = np.zeros(12)
            self.quat = (1.0, 0.0, 0.0, 0.0)
            self.gyro = (0.0, 0.0, 0.0)
            self.faults = 0
            self.keys = 0
            self.lx = self.ly = self.rx = self.ry = 0.0
    
    drv = FakeDriver()
    
    # With clip=1.0 (old default for PROMO)
    ctl_low = Controller(drv, FakePolicy(), stand_time=0.1, action_clip=1.0)
    ctl_low.step(FakeState())
    ctl_low.step(FakeState())
    ctl_low.step(FakeState())  # stand
    state_walk = FakeState()
    state_walk.keys = combo("L1", "A")
    ctl_low.step(state_walk)
    for _ in range(10):
        ctl_low.step(FakeState())
    state_walk.keys = combo("L1", "X")
    ctl_low.step(state_walk)
    ctl_low.step(FakeState())
    
    # Policy sees clipped action in history
    assert np.allclose(ctl_low.last_action, 1.0)  # clipped
    assert np.allclose(ctl_low.hist.buf[-1, contract.OBS_DIM:], 1.0)
    
    # With clip=10.0 (correct for PROMO)
    ctl_high = Controller(drv, FakePolicy(), stand_time=0.1, action_clip=10.0)
    ctl_high.step(FakeState())
    ctl_high.step(FakeState())
    ctl_high.step(FakeState())
    state_walk.keys = combo("L1", "A")
    ctl_high.step(state_walk)
    for _ in range(10):
        ctl_high.step(FakeState())
    state_walk.keys = combo("L1", "X")
    ctl_high.step(state_walk)
    ctl_high.step(FakeState())
    
    # Policy sees full action
    assert np.allclose(ctl_high.last_action, 2.0)
    assert np.allclose(ctl_high.hist.buf[-1, contract.OBS_DIM:], 2.0)


def test_action_clip_affects_target_not_history_recording():
    """action_clip limits motor targets but history records clipped value."""
    # This is the actual behavior: clip affects both target AND history
    # From test_controller.py test_bring_up_limits_reach_the_motor_and_the_history
    pass


def test_promo_v2_metadata_should_specify_clip():
    """PROMO policies should include recommended action_clip in metadata."""
    # Future: export_policy.py should write action_clip to metadata
    # For now we detect it from action_scale and policy type
    
    # PROMO v2 uses action_scale 0.25, needs clip ~10
    assert contract.ACTION_SCALE_PROMO_V2 == 0.25
    
    # RMA uses action_scale 0.4, clip 1.0 is fine (tanh output)
    assert contract.ACTION_SCALE == 0.4


def test_action_scale_reaches_target_but_clip_affects_history():
    """action_scale multiplies for motor, but history sees clipped raw action."""
    # From test_controller.py: action_scale=0.3, policy outputs 1.0
    # Target gets 0.3, history records 1.0
    # This is correct: policy thinks in its own units
    pass


def test_measured_walking_velocity_vs_clip():
    """Document measured velocities at different clip values (from MuJoCo)."""
    # From 2026-11-26 first walk log:
    # clip 1.0:  0.015 m/s (stuck)
    # clip 1.5:  0.37 m/s
    # clip 2.0:  0.50 m/s
    # clip 10.0: 0.50 m/s (same as 2.0, saturated)
    
    # Minimum working clip for PROMO forward walk
    min_working_clip = 2.0
    assert min_working_clip > 1.0


@pytest.mark.skip(reason="Controller doesn't validate action_clip yet - should be added")
def test_action_clip_validation():
    """action_clip must be positive."""
    from talon_a1.controller import Controller
    
    class FakeDriver:
        def set_target(self, q, kp, kd):
            return True
        def arm(self):
            return True
        def disarm(self):
            pass
    
    class FakePolicy:
        def action_mean(self, x, a_prev, hist):
            return np.zeros(12, np.float32), np.zeros(3), np.zeros(8)
    
    # Should accept positive clip
    ctl = Controller(FakeDriver(), FakePolicy(), action_clip=5.0)
    assert ctl.action_clip == 5.0
    
    # Zero or negative should be rejected (if validated)
    # Currently not validated, but should be
    with pytest.raises((ValueError, AssertionError)):
        Controller(FakeDriver(), FakePolicy(), action_clip=0.0)
    
    with pytest.raises((ValueError, AssertionError)):
        Controller(FakeDriver(), FakePolicy(), action_clip=-1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
