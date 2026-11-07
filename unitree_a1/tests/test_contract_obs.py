import math

import numpy as np

from talon_a1 import contract
from talon_a1.obs import History, build_observation, projected_gravity


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


def test_default_pose_by_name():
    # env_cfg.py HIP_THIGH_CALF_DEFAULT: left hips +0.05, right hips -0.05, thigh 0.8, calf -1.4
    for name, q in zip(contract.POLICY_JOINTS, contract.DEFAULT_JOINT_POS):
        leg, kind = name.split("_")[:2]
        want = {"hip": 0.05 if leg[1] == "L" else -0.05, "thigh": 0.8, "calf": -1.4}[kind]
        assert abs(q - want) < 1e-6, name


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
    o = build_observation(state, (0.5, -0.2, 0.3))
    assert o.shape == (contract.OBS_DIM,) and o.dtype == np.float32
    np.testing.assert_allclose(o[0:3], [0.1, 0.2, 0.3], rtol=1e-6)
    np.testing.assert_allclose(o[3:6], [0, 0, -1])
    np.testing.assert_allclose(o[6:9], [0.5, -0.2, 0.3], rtol=1e-6)
    np.testing.assert_allclose(o[9:21], q_policy, rtol=1e-6)  # raw angle, not relative to the default pose
    np.testing.assert_allclose(o[21:33], dq_policy)


def test_history_matches_train_py():
    # train.py: zeros on reset; after each action, drop the oldest and append (x, a)
    h = History(length=3, dim=2)
    np.testing.assert_array_equal(h.buf, np.zeros((3, 2)))
    a = h.push(np.array([1.0]), np.array([-1.0]))
    np.testing.assert_array_equal(a, [[0, 0], [0, 0], [1, -1]])
    h.push(np.array([2.0]), np.array([-2.0]))
    a = h.push(np.array([3.0]), np.array([-3.0]))
    a = h.push(np.array([4.0]), np.array([-4.0]))
    np.testing.assert_array_equal(a, [[2, -2], [3, -3], [4, -4]])
    h.reset()
    np.testing.assert_array_equal(h.buf, np.zeros((3, 2)))


def test_history_fill_holds_the_observation_with_zero_action():
    h = History(length=3, dim=3)
    h.fill(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(h.buf, [[1, 2, 0], [1, 2, 0], [1, 2, 0]])
