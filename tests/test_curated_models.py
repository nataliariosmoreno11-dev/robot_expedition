import json
from scripts.curate_models import verified_levels
from api import training


def test_inherited_approval_does_not_certify_new_weights(tmp_path):
    for name in ("ppo_model.zip", "optimizer.pt", "vecnormalize.pkl"):
        (tmp_path / name).write_bytes(b"weights")
    (tmp_path / "checkpoint_state.json").write_text(json.dumps({"total_timesteps": 200}))
    state = {"mastered": [1], "evaluations": {"1": {"passed": True, "episodes": 50,
             "success_rate": 1, "training_step": 100}}}
    (tmp_path / "curriculum_state.json").write_text(json.dumps(state))
    assert verified_levels(tmp_path) == []
    state["evaluations"]["1"]["training_step"] = 200
    (tmp_path / "curriculum_state.json").write_text(json.dumps(state))
    assert verified_levels(tmp_path) == [1]


def test_portable_export_recreates_job_and_rejects_corrupt_weights(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "JOBS", tmp_path / "jobs")
    run = "a" * 32
    model = tmp_path / "models" / run
    model.mkdir(parents=True)
    training.write(model / "approved-version.json", {"run_id": run, "completed_scenarios": [1]})
    training.write(model / "checkpoint_state.json", {"total_timesteps": 200})
    for name in ("ppo_model.zip", "optimizer.pt", "vecnormalize.pkl"):
        (model / name).write_bytes(b"weights")
    training.write(model / "sha256.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in model.iterdir() if p.is_file()})
    state = training.describe(training.job_path(run))
    assert state["approved_version"] and state["completed_scenarios"] == [1]
    (model / "ppo_model.zip").write_bytes(b"corrupt")
    assert not training.describe(training.job_path(run))["approved_version"]
