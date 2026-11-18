import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import PPO_CAMPAIGN_MODELS, app
from pipelines.policies import CampaignNumpyPPOPolicy


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("level,run_id", list(enumerate(PPO_CAMPAIGN_MODELS, start=1)))
def test_campaign_adapter_matches_approved_checkpoint_and_exposes_real_decision(level, run_id):
    adapter_path = ROOT / "catalog" / "campaign_adapters" / f"{run_id}.npz"
    approved = json.loads((ROOT / "models" / run_id / "approved-version.json").read_text())
    policy = CampaignNumpyPPOPolicy(adapter_path)
    assert policy.source_sha256 == approved["weights_sha256"]
    assert policy.describe()["architecture"]["policy"] == [132, 64, 64, 7]
    observation = np.zeros(132, dtype=np.float32)
    observation[-1] = level / 10
    action = policy(observation)
    assert 0 <= action < 7
    assert len(policy.last_decision["probabilities"]) == 7
    assert sum(policy.last_decision["probabilities"]) == pytest.approx(1.0, abs=1e-5)


def test_l7_visual_session_uses_checkpoint_actions_and_real_coordinates(monkeypatch, tmp_path):
    monkeypatch.setenv("POLICY_BACKEND", "numpy")
    import api.main as api_main
    monkeypatch.setattr(api_main, "EPISODE_DB", tmp_path / "episodes.sqlite")
    client = TestClient(app)
    run_id = PPO_CAMPAIGN_MODELS[6]
    reset = client.post("/sim/reset", json={
        "environment": "planetary", "difficulty_level": 7, "seed": 910007,
        "policy": f"trained:{run_id}",
    })
    assert reset.status_code == 200, reset.text
    body = reset.json()
    assert body["policy_model"]["model_sha256"] == json.loads(
        (ROOT / "models" / run_id / "approved-version.json").read_text()
    )["weights_sha256"]
    first_position = body["frame"][:2]
    step = client.post("/sim/step", json={"session_id": body["session_id"]})
    assert step.status_code == 200, step.text
    decision = step.json()["decision"]
    assert decision["action"] == step.json()["action"]
    assert len(decision["observation"]) == 132
    assert len(decision["probabilities"]) == 7
    assert step.json()["frame"][:2] != first_position
