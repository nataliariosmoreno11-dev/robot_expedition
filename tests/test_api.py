"""Pruebas de la API (paso 4) con TestClient."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.store import Store
from pipelines.policies import sha256_file
from robot_expedition.qlearning import STATE_SHAPE

client = TestClient(api_main.app)


def assert_frame(frame):
    assert len(frame) == 13 and len(frame[12]) == 9          # contrato de cuadros
    assert 1 <= frame[7] <= 3


def test_root_points_to_docs_and_routes():
    body = client.get("/").json()
    assert body["docs"] == "/docs" and "/evaluate" in body["routes"]


def test_health_version_and_monitor():
    assert client.get("/health").json()["status"] == "ok"
    version = client.get("/version").json()
    assert version["environment_version"] == "robot-expedition-3d/1.1.0"
    assert set(version["environment_hashes"]) == {"n1", "n2", "n3"}
    routes = client.get("/monitor").json()["routes"]
    assert "GET /health" in routes and routes["GET /health"]["requests"] >= 1


def test_catalog_lists_baselines_and_runs_detail():
    runs = {r["run_id"]: r for r in client.get("/catalog").json()["runs"]}
    assert {"baseline-rule", "baseline-random"} <= runs.keys()
    assert client.get("/runs/baseline-rule").json()["algorithm"] == "rule"
    assert client.get("/runs/no-existe").status_code == 404


def test_evaluate_rule_returns_episode_frames_and_metrics():
    res = client.post("/evaluate", json={"run_id": "baseline-rule", "level_id": "n1", "seeds": [100000, 100001]})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["metrics"]["episodes"] == 2 and len(body["cases"]) == 2
    episode = body["episode"]
    assert episode["mode"] == "evaluación" and episode["run_id"] == "baseline-rule"
    assert len(episode["frames"]) == episode["steps"] + 1
    for frame in episode["frames"][:3]:
        assert_frame(frame)
    assert len(body["map"]["beacons"]) == 1 and body["map"]["terrain"]["n"] == 121


def test_generalization_suite_exposes_the_first_scenario_map():
    body = client.post("/evaluate", json={"run_id": "baseline-rule", "level_id": "n2",
                                          "suite_id": "suite-n2-generalization-v1", "episodes": 2}).json()
    assert body["environment"] == {"scenario_set": "varied-v1", "max_steps": 400}
    assert body["episode"]["scenario_id"] == body["cases"][0]["scenario_id"]
    assert body["map"]["beacons"] == body["cases"][0]["beacons"]
    assert body["metrics"]["episodes_by_difficulty"] == {"easy": 2}


def test_evaluate_matches_the_offline_suite():
    from pipelines.evaluate import evaluate
    from pipelines.policies import RulePolicy

    online = client.post("/evaluate", json={"run_id": "baseline-rule", "level_id": "n2",
                                            "suite_id": "suite-n2-v1", "episodes": 2}).json()
    offline = evaluate(RulePolicy(), "n2", "suite-n2-v1", episodes=2)
    def strip(cases):
        return [{k: v for k, v in case.items() if not k.startswith("latency")} for case in cases]
    assert strip(online["cases"]) == strip(offline["cases"])


def test_evaluate_errors():
    assert client.post("/evaluate", json={"run_id": "x", "level_id": "n1", "seeds": [1]}).status_code == 404
    assert client.post("/evaluate", json={"run_id": "baseline-rule", "level_id": "n9", "seeds": [1]}).status_code == 422
    too_many = list(range(api_main.MAX_SEEDS + 1))
    assert client.post("/evaluate", json={"run_id": "baseline-rule", "level_id": "n1",
                                          "seeds": too_many}).status_code == 422


def test_manual_session_flow():
    reset = client.post("/sim/reset", json={"level_id": "n3", "seed": 4}).json()
    assert reset["mode"] == "manual" and reset["session_id"]
    assert_frame(reset["frame"])
    sid = reset["session_id"]
    step = client.post("/sim/step", json={"session_id": sid, "action": 0}).json()
    assert_frame(step["frame"]) and step["frame"][9] == 0 and step["end"] == "running"
    assert client.post("/sim/step", json={"session_id": sid}).status_code == 422      # manual sin accion
    assert client.post("/sim/step", json={"session_id": sid, "action": 9}).status_code == 422
    assert client.post("/sim/step", json={"session_id": "nope", "action": 0}).status_code == 404


def test_planetary_manual_session_and_persisted_replay(tmp_path, monkeypatch):
    database = tmp_path / "api-episodes.sqlite"
    monkeypatch.setattr(api_main, "EPISODE_DB", database)
    reset = client.post(
        "/sim/reset",
        json={"environment": "planetary", "difficulty_level": 4, "seed": 404},
    )
    assert reset.status_code == 200, reset.text
    body = reset.json()
    assert body["environment"] == "planetary"
    assert body["map"]["difficulty_level"] == 4
    assert len(body["map"]["samples"]) > body["map"]["mission"]["requested_samples"]
    assert body["actions"][5] == "recolectar"
    assert body["actions"][6] == "retroceder"

    stepped = client.post("/sim/step", json={"session_id": body["session_id"], "action": 5})
    assert stepped.status_code == 200, stepped.text
    assert stepped.json()["requested_samples"] >= 1

    summaries = client.get("/episodes").json()["episodes"]
    assert len(summaries) == 1 and summaries[0]["difficulty_level"] == 4
    replay = client.get(f"/episodes/{summaries[0]['episode_number']}/replay").json()
    assert replay["episode"]["objective"].startswith("Recolectar")
    assert len(replay["episode"]["frames"]) == 2
    assert len(replay["episode"]["mission_timeline"]) == 2

    second = client.post(
        "/sim/reset",
        json={"environment": "planetary", "difficulty_level": 2, "seed": 405},
    ).json()
    listing = client.get("/episodes").json()
    assert listing["schema_version"] == 2
    assert listing["storage"]["max_experiments"] is None
    protected = client.post(
        f"/episodes/{summaries[0]['episode_number']}/protect", json={"protected": True}
    )
    assert protected.status_code == 200 and protected.json()["protected"] is True
    with api_main.sessions_lock:
        api_main.sessions[body["session_id"]].env._finish_interrupted()
    metrics = client.get("/episodes/metrics/summary").json()
    assert metrics["episodes"] >= 1 and metrics["evolution"]
    numbers = [item["episode_number"] for item in client.get("/episodes").json()["episodes"]]
    compared = client.post("/episodes/compare", json={"episode_numbers": numbers}).json()
    assert len(compared["episodes"]) == 2

    with api_main.sessions_lock:
        session = api_main.sessions.pop(body["session_id"])
        second_session = api_main.sessions.pop(second["session_id"])
    session.env.close()
    second_session.env.close()


def test_policy_session_runs_to_the_end():
    sid = client.post("/sim/reset", json={"level_id": "n1", "seed": 100000, "policy": "baseline-rule"}).json()["session_id"]
    for _ in range(400):
        step = client.post("/sim/step", json={"session_id": sid}).json()
        if step["terminated"] or step["truncated"]:
            break
    assert step["end"] == "exito" and step["success"]
    assert client.post("/sim/step", json={"session_id": sid}).status_code == 409   # ya termino


@pytest.fixture
def trained_store(tmp_path, monkeypatch):
    """Un manifiesto de Q-learning con sus pesos, en un repositorio temporal."""
    run_id = "run-test"
    model_dir = tmp_path / "models" / run_id
    model_dir.mkdir(parents=True)
    weights = model_dir / "qlearning.npz"
    np.savez_compressed(weights, q=np.zeros((*STATE_SHAPE, 5), dtype=np.float32),
                        visits=np.zeros(STATE_SHAPE), epsilon=0.0)
    manifest = {"run_id": run_id, "algorithm": "qlearning", "level_id": "n2", "status": "completed",
                "environment_version": "robot-expedition-3d/1.0.0", "experiment_id": "exp-test",
                "model_sha256": sha256_file(weights), "metrics": {"success_rate": 0.0}}
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / f"{run_id}.json").write_text(json.dumps(manifest), encoding="utf-8")
    store = Store(tmp_path)
    monkeypatch.setattr(api_main, "store", store)
    return store, weights


def test_trained_model_is_listed_verified_and_evaluated(trained_store):
    runs = {r["run_id"] for r in client.get("/catalog").json()["runs"]}
    assert "run-test" in runs
    res = client.post("/evaluate", json={"run_id": "run-test", "level_id": "n1", "seeds": [100000]})
    assert res.status_code == 200, res.text
    assert res.json()["trained_level_id"] == "n2"


def test_tampered_model_is_rejected(trained_store):
    store, weights = trained_store
    weights.write_bytes(weights.read_bytes() + b"x")
    res = client.post("/evaluate", json={"run_id": "run-test", "level_id": "n2", "seeds": [1]})
    assert res.status_code == 409 and "SHA-256" in res.json()["detail"]


def test_missing_weights_is_a_clear_error(trained_store):
    store, weights = trained_store
    weights.unlink()
    res = client.post("/evaluate", json={"run_id": "run-test", "level_id": "n2", "seeds": [1]})
    assert res.status_code == 503 and "DVC" in res.json()["detail"]


def test_monitor_p95_uses_nearest_rank():
    monitor = api_main.Monitor()
    for ms in (1, 2, 3, 100):
        monitor.record("GET /x", 200, ms)
    route = monitor.snapshot()["routes"]["GET /x"]
    assert route["latency_ms_p95"] == 100 and route["latency_ms_p95"] >= route["latency_ms_mean"]


def test_cors_allows_the_vercel_origin_only():
    ok = client.options("/catalog", headers={"Origin": "https://web-five-gules-83.vercel.app",
                                             "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "https://web-five-gules-83.vercel.app"
    bad = client.options("/catalog", headers={"Origin": "https://evil.example",
                                              "Access-Control-Request-Method": "GET"})
    assert "access-control-allow-origin" not in bad.headers
