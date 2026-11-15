from types import SimpleNamespace

import pytest

from api import campaign_worker as campaign
from api import training


def report():
    return {"cases": [{"difficulty_level": level, "seed": seed,
                        "success": seed < 43, "collisions": 2}
                       for level in range(1, 11) for seed in range(50)]}


def test_publication_requires_every_level_and_distinct_evaluations():
    evidence = report()
    assert campaign.qualifies(evidence)
    evidence["cases"] = [case for case in evidence["cases"] if case["difficulty_level"] != 10]
    assert not campaign.qualifies(evidence)
    evidence = report()
    for case in evidence["cases"]:
        case["seed"] = 1
    assert not campaign.qualifies(evidence)


@pytest.mark.parametrize("field,value", [("success", False), ("collisions", 3)])
def test_global_score_cannot_hide_a_failing_scenario(field, value):
    evidence = report()
    for case in evidence["cases"]:
        if case["difficulty_level"] == 10:
            case[field] = value
    evidence["metrics"] = {"success_rate": 1.0}
    assert not campaign.qualifies(evidence)


def test_supervisor_continues_saved_weights_until_all_levels_validate(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "JOBS", tmp_path / "jobs")
    monkeypatch.setattr(campaign.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    parents = []
    def start(request):
        parents.append(request.parent_job_id)
        job_id = str(len(parents)) * 32
        training.write(training.JOBS / job_id / "state.json", {"job_id": job_id})
        training.write(tmp_path / "models" / job_id / "curriculum_state.json",
                       {"mastered": list(range(1, 11)) if len(parents) == 2 else [1], "current_level": 2})
        return {"job_id": job_id}
    monkeypatch.setattr(training, "start", start)
    monkeypatch.setattr(training, "describe", lambda path: {
        "status": "completed", "run_id": path.name, "progress": {"total_timesteps": 100},
        "evaluation": {"success_rate": .1}})
    monkeypatch.setattr(campaign, "certify", lambda job: {"job_id": job})
    campaign.run("parent", 1024)
    state = training.read(tmp_path / "logs/autonomous_campaign/state.json")
    assert parents == ["parent", "1" * 32]
    assert state["status"] == "completed"
    assert state["release"]["job_id"] == "2" * 32
    assert len(state["rounds"]) == 1


def test_explicit_stop_prevents_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    path = tmp_path / "logs/autonomous_campaign/stop"
    path.parent.mkdir(parents=True)
    path.touch()
    with pytest.raises(RuntimeError, match="stopped explicitly"):
        campaign.run("parent", 1024)


def test_stop_campaign_signals_current_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "JOBS", tmp_path / "jobs")
    monkeypatch.setenv("ENABLE_LOCAL_TRAINING", "1")
    job = "a" * 32
    training.write(training.JOBS / job / "state.json", {})
    training.write(tmp_path / "logs/autonomous_campaign/state.json", {"active_job_id": job})
    training.stop_campaign()
    assert (tmp_path / "logs/autonomous_campaign/stop").exists()
    assert (training.JOBS / job / "stop").exists()


def test_job_can_be_observed_before_mlflow_assigns_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "worker_alive", lambda _: True)
    path = tmp_path / "jobs" / ("a" * 32)
    training.write(path / "state.json", {"status": "starting", "run_id": None})
    assert training.describe(path)["status"] == "starting"
