"""export_policy.py against randomly initialised talon_rl.rma networks. Needs torch and talon_rl (training machine)."""

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
networks = pytest.importorskip("talon_rl.rma.networks")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def checkpoint(path, **overrides):
    torch.manual_seed(0)
    manifest = {"task": "TalonRl-PromoRma-A1", "phase": 2, "state_dim": 33, "velocity_dim": 3,
                "observation_normalization": "none"}
    manifest.update(overrides)
    torch.save({"it": 0, "run_manifest": manifest, "observation_normalizer": None,
                "pi": networks.BasePolicy(state_dim=33, velocity_dim=3).state_dict(),
                "phi": networks.AdaptationModule(state_dim=33).state_dict(),
                "velocity_estimator": networks.VelocityEstimator(state_dim=33).state_dict()}, path)


def export(ckpt, out):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_policy.py"), "--checkpoint", str(ckpt),
                           "--out", str(out), "--samples", "64"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_export_parity(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    checkpoint(ckpt)
    assert export(ckpt, tmp_path / "policy.npz").returncode == 0
    report = json.loads((tmp_path / "policy.parity.json").read_text())
    assert report["verdict"] == "PASS"


def test_export_refuses_other_tasks(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    checkpoint(ckpt, task="TalonRl-Promo-A1")
    result = export(ckpt, tmp_path / "policy.npz")
    assert result.returncode != 0 and b"disagrees" in result.stderr
    assert not (tmp_path / "policy.npz").exists()


def test_export_refuses_normalized_observations(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    checkpoint(ckpt, observation_normalization="running_mean_std")
    result = export(ckpt, tmp_path / "policy.npz")
    assert result.returncode != 0 and b"normalization" in result.stderr
