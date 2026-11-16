import hashlib
import json

import pytest

from scripts import training_control as control


def test_reused_pid_is_not_a_training_process():
    import os
    assert not control.alive({"pid": os.getpid()}, "api.campaign_worker")


def test_resume_active_campaign_is_noop(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "alive", lambda *args: True)
    monkeypatch.setattr(control.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("duplicate process"))
    control.resume()
    assert "ya está activa" in capsys.readouterr().out


def test_resume_without_verified_weights_preserves_state(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "ROOT", tmp_path)
    state = tmp_path / "logs/autonomous_campaign/state.json"
    control.write(state, {"status": "failed", "active_job_id": "old"})
    original = state.read_text()
    with pytest.raises(ValueError, match="No hay checkpoint"):
        control.resume()
    assert state.read_text() == original


def make_job(root, number, steps, corrupt=False):
    identifier = f"{number:032x}"
    model = root / "models" / identifier
    model.mkdir(parents=True)
    data = {"ppo_model.zip": b"weights", "vecnormalize.pkl": b"normalizer",
            "optimizer.pt": b"optimizer", "checkpoint_state.json": json.dumps({"total_timesteps": steps}).encode()}
    for name, content in data.items():
        (model / name).write_bytes(content)
    control.write(model / "sha256.json", {name: hashlib.sha256(content).hexdigest() for name, content in data.items()})
    control.write(model / "config.json", {"environment": {"history_steps": 4, "campaign": True}})
    if corrupt:
        (model / "ppo_model.zip").write_bytes(b"incomplete")
    job = {"job_id": identifier, "run_id": identifier, "status": "running"}
    control.write(root / "logs/training_jobs" / identifier / "state.json", job)
    return job


def test_resume_skips_corrupt_newest_and_detaches(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control.time, "sleep", lambda _: None)
    older = make_job(tmp_path, 1, 100)
    make_job(tmp_path, 2, 200, corrupt=True)
    calls = []
    class Process:
        pid = 123
        def poll(self):
            return None
    def spawn(*args, **kwargs):
        calls.append((args, kwargs))
        return Process()
    monkeypatch.setattr(control.subprocess, "Popen", spawn)
    control.resume()
    state = control.read(tmp_path / "logs/autonomous_campaign/state.json")
    assert state["parent_job_id"] == older["job_id"]
    assert state["active_job_id"] is None
    assert calls[0][1]["start_new_session"] is True
    assert control.read(tmp_path / "logs/training_jobs" / older["job_id"] / "state.json")["status"] == "interrupted"


def test_status_does_not_trust_saved_running_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "ROOT", tmp_path)
    control.write(tmp_path / "logs/autonomous_campaign/state.json", {"status": "running", "pid": 99999999})
    assert "DETENIDO" in control.snapshot()
