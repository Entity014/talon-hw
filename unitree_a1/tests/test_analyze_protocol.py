import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import analyze_protocol as A


def fake_replay(scale=1.0, yaw_scale=1.0):
    from eval_protocol import REGIMES, profile_command
    seg = REGIMES["replay"]; T = sum(d for d, _ in seg); n = int(T / 0.02)
    t = np.arange(n) * 0.02; c = np.array([profile_command(x, seg) for x in t])
    return {"mode": np.array(["WALK"] * n), "t": t, "command": c, "gyro": np.stack([0 * t, 0 * t, c[:, 2] * yaw_scale], 1),
            "v_hat": np.hstack([c[:, :2] * scale, c[:, :1] * 0])}


class D(dict):
    files = property(lambda self: list(self.keys()))


def test_all_six_segments_pass_when_robot_follows():
    seg = A.segment_metrics(D(fake_replay()))
    assert [s["label"] for s in seg] == ["fwd", "back", "left", "right", "yawL", "yawR"]
    assert all(s["passed"] for s in seg)
    assert abs(seg[4]["yaw_deg"] - np.degrees(0.4 * 3.5)) < 2  # the 0.5 s ramp is outside the constant-command run


def test_slow_translation_and_no_yaw_fail():
    seg = A.segment_metrics(D(fake_replay(scale=0.5, yaw_scale=0.0)))
    assert not any(s["passed"] for s in seg)
