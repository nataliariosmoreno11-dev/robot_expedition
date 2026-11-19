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


def test_planetary_rule_session_advances_without_manual_actions(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "EPISODE_DB", tmp_path / "automatic-episodes.sqlite")
    reset = client.post(
        "/sim/reset",
        json={
            "environment": "planetary",
            "difficulty_level": 3,
            "seed": 303,
            "policy": "planetary-rule",
        },
    )
    assert reset.status_code == 200, reset.text
    body = reset.json()
    assert body["mode"] == "política"
    assert body["policy"] == "planetary-rule"
    with api_main.sessions_lock:
        # Se respeta el límite de diseño del escenario; la batería es el límite real.
        session = api_main.sessions[body["session_id"]]
        assert session.env.max_steps == session.env.scenario.max_steps
        assert session.policy.map is not None          # el guía recibió el mapa de misión

    step = client.post("/sim/step", json={"session_id": body["session_id"]})
    assert step.status_code == 200, step.text
    payload = step.json()
    assert 0 <= payload["action"] < len(body["actions"])
    assert payload["steps"] == 1

    with api_main.sessions_lock:
        session = api_main.sessions.pop(body["session_id"])
    session.env.close()


def test_planetary_rule_escapes_when_rover_stays_stalled():
    policy = api_main.PlanetaryRulePolicy()
    observation = np.ones(33, dtype=np.float32)
    observation[10:14] = 0.0            # sin objetivo visible, sin velocidad ni giro
    observation[15] = 0.0
    actions = [policy(observation) for _ in range(6)]
    assert 6 in actions                 # retrocede para despegarse
    assert any(action in {3, 4} for action in actions)


def test_planetary_rule_guide_never_receives_sample_positions(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "EPISODE_DB", tmp_path / "guide-episodes.sqlite")
    reset = client.post("/sim/reset", json={"environment": "planetary", "difficulty_level": 2,
                                           "seed": 3, "policy": "planetary-rule"}).json()
    with api_main.sessions_lock:
        session = api_main.sessions.pop(reset["session_id"])
    mission = session.policy.map
    sample_points = {(s["x"], s["y"]) for s in session.env.samples}
    assert not any(hasattr(mission, name) for name in ("samples", "targets"))
    assert not sample_points & set(map(tuple, mission.rocks))
    assert session.policy.targets == []                 # las muestras se descubren con la cámara
    session.env.close()


def test_planetary_rule_guide_completes_a_multi_sample_mission(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "EPISODE_DB", tmp_path / "mission-episodes.sqlite")
    sid = client.post("/sim/reset", json={"environment": "planetary", "difficulty_level": 2,
                                         "seed": 5, "policy": "planetary-rule"}).json()["session_id"]
    for _ in range(400):
        step = client.post("/sim/step", json={"session_id": sid}).json()
        if step["terminated"] or step["truncated"]:
            break
    assert step["end"] == "exito", step
    assert step["samples_collected"] == step["requested_samples"] == 2
    with api_main.sessions_lock:
        api_main.sessions.pop(sid).env.close()


def test_planetary_qlearning_session_updates_and_saves_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "planetary-qlearning.npz"
    monkeypatch.setattr(api_main, "PLANETARY_Q_CHECKPOINT", checkpoint)
    monkeypatch.setattr(api_main, "EPISODE_DB", tmp_path / "q-episodes.sqlite")
    reset = client.post("/sim/reset", json={
        "environment": "planetary", "difficulty_level": 1,
        "seed": 707, "policy": "planetary-qlearning",
    })
    assert reset.status_code == 200, reset.text
    sid = reset.json()["session_id"]
    client.post("/sim/step", json={"session_id": sid})
    step = client.post("/sim/step", json={"session_id": sid}).json()
    # La actualización Q se completa un paso después, cuando existe la sugerencia
    # del guía para el estado siguiente.
    assert step["training"]["updates"] == 1
    with api_main.sessions_lock:
        session = api_main.sessions[sid]
        assert session.env.training_reward_limit == 0.0
        assert session.env.training_reward_floor == 0.0
        assert session.env.training_no_progress_limit == api_main.QLEARNING_NO_PROGRESS_LIMIT
        assert session.env.training_collision_penalty == 4.0
        assert session.env.max_steps == session.env.scenario.max_steps
        session.policy.agent.finish_episode()
        session.policy.agent.save(checkpoint)
        api_main.sessions.pop(sid)
    session.env.close()
    assert checkpoint.exists()
    status = client.get("/training/qlearning")
    assert status.status_code == 200
    assert status.json()["updates"] == 1


def test_planetary_qlearning_keeps_scenario_step_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "PLANETARY_Q_CHECKPOINT", tmp_path / "planetary-qlearning.npz")
    monkeypatch.setattr(api_main, "EPISODE_DB", tmp_path / "q-episodes.sqlite")
    for level in (1, 4, 5, 8):
        reset = client.post("/sim/reset", json={
            "environment": "planetary", "difficulty_level": level,
            "seed": 700 + level, "policy": "planetary-qlearning",
        })
        assert reset.status_code == 200, reset.text
        payload = reset.json()
        with api_main.sessions_lock:
            session = api_main.sessions.pop(payload["session_id"])
        # Antes 50/100 pasos: imposible para los niveles 2-8 (necesitan 90-450).
        assert payload["max_steps"] == session.env.scenario.max_steps >= 650
        assert session.env.training_reward_floor == 0.0
        session.env.close()


def test_failed_episode_is_saved_and_loaded_by_the_next_scenario(tmp_path, monkeypatch):
    checkpoint = tmp_path / "planetary-qlearning.npz"
    monkeypatch.setattr(api_main, "PLANETARY_Q_CHECKPOINT", checkpoint)
    monkeypatch.setattr(api_main, "EPISODE_DB", tmp_path / "q-episodes.sqlite")
    reset = client.post("/sim/reset", json={
        "environment": "planetary", "difficulty_level": 1,
        "seed": 801, "policy": "planetary-qlearning",
    }).json()
    with api_main.sessions_lock:
        session = api_main.sessions[reset["session_id"]]
        session.env.max_steps = session.env.steps + 2      # fuerza el corte por tiempo
    client.post("/sim/step", json={"session_id": reset["session_id"]})
    failed = client.post("/sim/step", json={"session_id": reset["session_id"]}).json()
    assert failed["truncated"] is True
    assert failed["end"] == "tiempo"
    assert checkpoint.exists()

    resumed = client.post("/sim/reset", json={
        "environment": "planetary", "difficulty_level": 8,
        "seed": 808, "policy": "planetary-qlearning",
    }).json()
    with api_main.sessions_lock:
        restored = api_main.sessions.pop(resumed["session_id"])
        failed_session = api_main.sessions.pop(reset["session_id"], None)
    assert restored.policy.agent.episodes == failed["training"]["episodes"]
    assert restored.policy.agent.updates == failed["training"]["updates"]
    restored.env.close()
    if failed_session is not None:
        failed_session.env.close()


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


def test_campaign_progress_picks_most_advanced_run_not_newest_file(tmp_path, monkeypatch):
    import json as _json, os as _os
    for run_id, passed, step, mtime in (("a" * 32, 8, 7_000_000, 100), ("b" * 32, 4, 2_000_000, 999)):
        path = tmp_path / "models" / run_id / "curriculum_state.json"
        path.parent.mkdir(parents=True)
        path.write_text(_json.dumps({"schema": "fixed-campaign-curriculum/v1", "current_level": passed + 1,
                                     "evaluations": {str(i): {"passed": True, "training_step": step}
                                                     for i in range(1, passed + 1)}}))
        _os.utime(path, (mtime, mtime))                      # el menos avanzado es el mas reciente
    monkeypatch.setattr(api_main, "ROOT", tmp_path)
    assert client.get("/campaign/progress").json()["run_id"] == "a" * 32


def test_qlearning_seed_downloads_only_when_checkpoint_is_missing(tmp_path, monkeypatch):
    from robot_expedition.planetary.qlearning import PlanetaryQLearningAgent

    seed = PlanetaryQLearningAgent()
    seed.episodes = 42
    seed_file = seed.save(tmp_path / "seed.npz")
    checkpoint = tmp_path / "data" / "models" / "planetary_qlearning.npz"
    monkeypatch.setattr(api_main, "PLANETARY_Q_CHECKPOINT", checkpoint)
    monkeypatch.setattr(api_main, "PLANETARY_Q_SEED_URL", seed_file.as_uri())

    assert api_main._seed_q_checkpoint() == "seeded"
    assert PlanetaryQLearningAgent().load(checkpoint).episodes == 42

    learned = PlanetaryQLearningAgent()
    learned.episodes = 7
    learned.save(checkpoint)
    assert api_main._seed_q_checkpoint() == "existing"
    assert PlanetaryQLearningAgent().load(checkpoint).episodes == 7     # nunca pisa lo aprendido


def test_qlearning_seed_failure_keeps_api_usable(tmp_path, monkeypatch):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"no es un npz")
    checkpoint = tmp_path / "models" / "planetary_qlearning.npz"
    monkeypatch.setattr(api_main, "PLANETARY_Q_CHECKPOINT", checkpoint)
    monkeypatch.setattr(api_main, "PLANETARY_Q_SEED_URL", bad.as_uri())

    assert api_main._seed_q_checkpoint() == "failed"
    assert not checkpoint.exists() and not list(checkpoint.parent.glob("*.tmp"))


def test_incompatible_q_checkpoint_is_kept_renamed_and_api_keeps_working(tmp_path, monkeypatch):
    old = tmp_path / "planetary_qlearning.npz"
    np.savez_compressed(old, states=np.zeros((1, 11), dtype=np.int16), q=np.zeros((1, 7), np.float32),
                        visits=np.zeros((1, 7), np.int32), metadata=np.asarray('{"episodes": 9}'))
    monkeypatch.setattr(api_main, "PLANETARY_Q_CHECKPOINT", old)
    monkeypatch.setattr(api_main, "PLANETARY_Q_CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(api_main, "PLANETARY_Q_SELECTION", tmp_path / "active.json")
    monkeypatch.setattr(api_main, "EPISODE_DB", tmp_path / "q-episodes.sqlite")
    assert client.get("/training/qlearning").json()["checkpoint_compatible"] is False
    reset = client.post("/sim/reset", json={"environment": "planetary", "difficulty_level": 1,
                                           "seed": 1, "policy": "planetary-qlearning"})
    assert reset.status_code == 200, reset.text
    backups = list(tmp_path.glob("planetary_qlearning-estado-anterior*.npz"))
    assert len(backups) == 1 and not old.exists()                 # conservado, nunca borrado
    listing = client.get("/training/qlearning/checkpoints").json()
    assert backups[0].name in {item["name"] for item in listing["incompatible"]}
    with api_main.sessions_lock:
        api_main.sessions.pop(reset.json()["session_id"]).env.close()


def test_batched_steps_match_single_steps_and_stop_at_episode_end(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "EPISODE_DB", tmp_path / "batch-episodes.sqlite")
    body = {"environment": "planetary", "difficulty_level": 1, "seed": 11, "policy": "planetary-rule"}
    single = client.post("/sim/reset", json=body).json()["session_id"]
    batched = client.post("/sim/reset", json=body).json()["session_id"]
    single_frames = []
    while True:
        step = client.post("/sim/step", json={"session_id": single}).json()
        single_frames.append(step["frame"])
        if step["terminated"] or step["truncated"]:
            break
    batched_frames, requests = [], 0
    while True:
        step = client.post("/sim/step", json={"session_id": batched, "steps": 8}).json()
        requests += 1
        assert step["frame"] == step["frames"][-1]
        batched_frames += step["frames"]
        if step["terminated"] or step["truncated"]:
            break
    assert batched_frames == single_frames              # mismo episodio, menos viajes de red
    assert step["steps"] == len(single_frames) and requests == -(-len(single_frames) // 8)
    assert client.post("/sim/step", json={"session_id": batched, "steps": 8}).status_code == 409
    for sid in (single, batched):
        with api_main.sessions_lock:
            api_main.sessions.pop(sid).env.close()


def test_manual_action_ignores_batch_size(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "EPISODE_DB", tmp_path / "manual-batch.sqlite")
    sid = client.post("/sim/reset", json={"environment": "planetary", "difficulty_level": 1,
                                         "seed": 2, "policy": "manual"}).json()["session_id"]
    step = client.post("/sim/step", json={"session_id": sid, "action": 0, "steps": 20}).json()
    assert step["steps"] == 1 and len(step["frames"]) == 1
    assert client.post("/sim/step", json={"session_id": sid, "steps": 41}).status_code == 422
    with api_main.sessions_lock:
        api_main.sessions.pop(sid).env.close()
