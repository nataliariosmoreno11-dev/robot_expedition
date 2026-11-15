import hashlib
import json

import pytest
from fastapi import HTTPException

from api import training


def make_job(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "JOBS", tmp_path / "jobs")
    job_id, run_id = "a" * 32, "b" * 32
    training.write(training.JOBS / job_id / "state.json", {
        "run_id": run_id, "status": "interrupted", "job_id": job_id,
    })
    model = tmp_path / "models" / run_id
    model.mkdir(parents=True)
    hashes = {}
    for name in ("ppo_model.zip", "vecnormalize.pkl", "optimizer.pt", "checkpoint_state.json"):
        (model / name).write_bytes(b"checkpoint")
        hashes[name] = hashlib.sha256(b"checkpoint").hexdigest()
    training.write(model / "sha256.json", hashes)
    training.write(model / "config.json", {"environment": {"campaign": True, "history_steps": 4}})
    return job_id, model


def test_resume_accepts_interrupted_checkpoint_and_rejects_tampering(tmp_path, monkeypatch):
    job_id, model = make_job(tmp_path, monkeypatch)
    assert training.checkpoint_for(job_id) == model
    (model / "ppo_model.zip").write_bytes(b"changed weights")
    with pytest.raises(HTTPException) as exc:
        training.checkpoint_for(job_id)
    assert exc.value.status_code == 409


def test_cannot_load_active_checkpoint_or_traverse_path(tmp_path, monkeypatch):
    job_id, _ = make_job(tmp_path, monkeypatch)
    path = training.JOBS / job_id / "state.json"
    state = json.loads(path.read_text())
    state["status"] = "running"
    training.write(path, state)
    with pytest.raises(HTTPException) as exc:
        training.checkpoint_for(job_id)
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc:
        training.job_path("../../models")
    assert exc.value.status_code == 404


def test_remote_training_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_LOCAL_TRAINING", raising=False)
    with pytest.raises(HTTPException) as exc:
        training.start(training.StartTraining())
    assert exc.value.status_code == 403


def test_stop_is_persisted_without_killing_worker(tmp_path, monkeypatch):
    job_id, model = make_job(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_LOCAL_TRAINING", "1")
    before = (model / "ppo_model.zip").read_bytes()
    training.stop(job_id)
    assert (training.JOBS / job_id / "stop").exists()
    assert (model / "ppo_model.zip").read_bytes() == before
