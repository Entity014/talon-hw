"""export_policy.py against a randomly initialised PromoActorCritic. Needs torch and talon_rl (training machine)."""

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
networks = pytest.importorskip("talon_rl.promo.networks")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_export_parity(tmp_path):
    torch.manual_seed(0)
    ac = networks.PromoActorCritic(networks.PromoDims())
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"format_version": 1, "iteration": 0, "model": ac.state_dict(),
                "manifest": {"task": "TalonRl-Promo-A1", "policy_observation_dim": 45, "preference_dim": 3,
                             "action_dim": 12}}, ckpt)
    out = tmp_path / "policy.npz"
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_policy.py"), "--checkpoint", str(ckpt),
                    "--out", str(out), "--samples", "64"], check=True)
    report = json.loads((tmp_path / "policy.parity.json").read_text())
    assert report["verdict"] == "PASS"
