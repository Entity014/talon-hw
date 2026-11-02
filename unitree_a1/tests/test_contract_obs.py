import math

import numpy as np

from talon_a1 import contract
from talon_a1.obs import History, ObservationBuilder, projected_gravity


class FakeState(object):
    def __init__(self, q, dq, quat=(1.0, 0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0)):
        self.q, self.dq, self.quat, self.gyro = q, dq, quat, gyro


def test_joint_permutation_by_name():
    names_in_policy_order = contract.sdk_to_policy(np.array(contract.SDK_JOINTS))
    assert tuple(names_in_policy_order) == contract.POLICY_JOINTS
    x = np.arange(12)
    assert np.array_equal(contract.policy_to_sdk(contract.sdk_to_policy(x)), x)
    assert np.array_equal(contract.sdk_to_policy(contract.policy_to_sdk(x)), x)


def test_default_pose_inside_sdk_limits():
    # a1_const.h, per joint type
    lo = {"hip": -0.802, "thigh": -1.05, "calf": -2.7}
    hi = {"hip": 0.802, "thigh": 4.19, "calf": -0.916}
    for name, q in zip(contract.POLICY_JOINTS, contract.DEFAULT_JOINT_POS):
        kind = name.split("_")[1]
        assert lo[kind] < q < hi[kind], name


def test_projected_gravity_known_orientations():
    s = math.sqrt(0.5)
    np.testing.assert_allclose(projected_gravity((1, 0, 0, 0)), [0, 0, -1], atol=1e-7)
    # +90 deg roll about x: body y points up, so gravity is -y in the body frame
    np.testing.assert_allclose(projected_gravity((s, s, 0, 0)), [0, -1, 0], atol=1e-6)
    # +90 deg pitch about y: nose down, body x points at the ground
    np.testing.assert_allclose(projected_gravity((s, 0, s, 0)), [1, 0, 0], atol=1e-6)
    # yaw does not change gravity
    np.testing.assert_allclose(projected_gravity((s, 0, 0, s)), [0, 0, -1], atol=1e-6)


def test_observation_layout():
    q_policy = contract.DEFAULT_JOINT_POS + np.arange(12, dtype=np.float32) * 0.01
    dq_policy = np.arange(12, dtype=np.float32) + 100.0
    state = FakeState(contract.policy_to_sdk(q_policy), contract.policy_to_sdk(dq_policy), gyro=(0.1, 0.2, 0.3))
    b = ObservationBuilder()
    b.last_action[:] = 7.0
    o = b.build(state, (0.5, -0.2, 0.3))
    assert o.shape == (contract.OBS_DIM,) and o.dtype == np.float32
    np.testing.assert_allclose(o[0:3], [0.1, 0.2, 0.3], rtol=1e-6)
    np.testing.assert_allclose(o[3:6], [0, 0, -1])
    np.testing.assert_allclose(o[6:9], [0.5, -0.2, 0.3], rtol=1e-6)
    np.testing.assert_allclose(o[9:21], np.arange(12) * 0.01, atol=1e-6)
    np.testing.assert_allclose(o[21:33], dq_policy)
    np.testing.assert_allclose(o[33:45], 7.0)


def test_history_matches_train_py():
    # train.py: reset fills all slots with o_0; each step drops the oldest and appends the newest
    h = History(length=3, dim=2)
    a = h.push(np.array([1.0, 1.0]))
    np.testing.assert_array_equal(a, [[1, 1], [1, 1], [1, 1]])
    a = h.push(np.array([2.0, 2.0]))
    np.testing.assert_array_equal(a, [[1, 1], [1, 1], [2, 2]])
    a = h.push(np.array([3.0, 3.0]))
    a = h.push(np.array([4.0, 4.0]))
    np.testing.assert_array_equal(a, [[2, 2], [3, 3], [4, 4]])
    h.reset()
    a = h.push(np.array([9.0, 9.0]))
    np.testing.assert_array_equal(a, [[9, 9], [9, 9], [9, 9]])
