from pathlib import Path

import numpy as np
import pytest

from robot_expedition.envs.planetary_exploration import (
    COLLECT_ACTION,
    PlanetaryExplorationEnv,
)
from robot_expedition.planetary.campaign import CATALOG, SCENARIO_IDS, load_scenario
from robot_expedition.planetary.campaign_curriculum import CampaignCurriculum
from robot_expedition.planetary.episode_store import EpisodeStore
from robot_expedition.planetary.reference import ReferencePlanner, campaign_reference


def test_exactly_ten_independent_fixed_maps_and_no_sensor_seed_leak():
    assert sorted(p.stem for p in CATALOG.glob("*.json")) == list(SCENARIO_IDS)
    env = PlanetaryExplorationEnv(curriculum=False)
    for level in range(1, 11):
        env.reset(seed=1, options={"difficulty_level": level})
        first = env.scenario.to_dict()
        env.reset(seed=777, options={"difficulty_level": level})
        assert env.scenario.to_dict() == first
        assert env.scenario_id == f"scenario_{level:02d}"
        assert len(env._rock_ids) == len(
            env.rocks
        )  # every visible wall actually exists in physics
    env.close()


def test_campaign_api_lists_all_maps_and_preserves_restarted_attempt(
    tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient
    from api import main

    monkeypatch.setattr(main, "EPISODE_DB", tmp_path / "api.sqlite")
    client = TestClient(main.app)
    response = client.get("/scenarios")
    assert response.status_code == 200 and len(response.json()["scenarios"]) == 10
    assert client.get("/campaign/progress").status_code == 200
    first = client.post(
        "/sim/reset", json={"environment": "planetary", "difficulty_level": 10}
    ).json()
    step = client.post(
        "/sim/step", json={"session_id": first["session_id"], "action": 6}
    )
    assert step.status_code == 200
    second = client.post(
        "/sim/reset",
        json={
            "environment": "planetary",
            "difficulty_level": 10,
            "previous_session_id": first["session_id"],
        },
    )
    assert second.status_code == 200
    history = client.get("/scenarios/scenario_10/history?mode=manual").json()
    assert len(history) == 2 and history[-1]["end"] == "reinicio"
    assert (
        client.get("/scenarios/scenario_10/progress?mode=manual").json()["episodes"]
        == 1
    )
    assert (
        client.get(f"/episodes/{history[-1]['episode_number']}/replay").status_code
        == 200
    )
    main.sessions.pop(second.json()["session_id"]).env.close()


def test_all_reference_routes_have_clearance_and_sufficient_budget():
    for sid in SCENARIO_IDS:
        s = load_scenario(sid)
        p = ReferencePlanner(s)
        r = campaign_reference(sid)
        assert all(p.free[p.cell(point)] for point in r["reference_path"])
        assert (
            r["estimated_steps"] < s.max_steps
            and r["estimated_energy"] < s.battery_budget
        )
        assert len(r["target_order"]) == s.requested_samples
    nine, ten = campaign_reference("scenario_09"), campaign_reference("scenario_10")
    assert ten["relevant_branch_count"] > nine["relevant_branch_count"]
    assert ten["multi_target_route_cost"] > nine["multi_target_route_cost"]
    assert campaign_reference("scenario_08")["route_detour_ratio"] > 2


def mastery(level, successes=45, count=50, kind="mastery", collisions=0):
    return {
        "scenario_level": level,
        "evaluation_kind": kind,
        "cases": [
            {
                "seed": 900000 + i,
                "difficulty_level": level,
                "success": i < successes,
                "samples_collected": 1 if i < successes else 0,
                "requested_samples": 1,
                "battery": 50,
                "collisions": collisions,
            }
            for i in range(count)
        ],
    }


def test_mastery_requires_independent_complete_safe_evaluations(tmp_path):
    c = CampaignCurriculum(tmp_path / "curriculum.json")
    for _ in range(100):
        assert not c.record_result(True, evaluated_level=1)
    assert not c.advance_scenario()
    assert not c.evaluate_scenario_mastery(mastery(1, count=49))
    assert not c.evaluate_scenario_mastery(mastery(1, successes=42))
    assert not c.evaluate_scenario_mastery(mastery(1, collisions=3))
    duplicate = mastery(1)
    duplicate["cases"][1]["seed"] = duplicate["cases"][0]["seed"]
    assert not c.evaluate_scenario_mastery(duplicate)
    with pytest.raises(ValueError):
        c.evaluate_scenario_mastery(mastery(1, kind="robustness"))
    assert c.evaluate_scenario_mastery(mastery(1)) and c.advance_scenario()
    assert CampaignCurriculum(c.state_path).current_level == 2
    levels = {c.select_level(np.random.default_rng(i)) for i in range(100)}
    assert levels == {1, 2}
    c.return_to_previous_scenario(1)
    assert c.current_level == 1


def test_all_attempts_survive_reopen_and_replay_is_immutable(tmp_path):
    db = tmp_path / "history.sqlite"
    env = PlanetaryExplorationEnv(
        curriculum=False, max_steps=1, record_episodes=True, episode_db=db
    )
    for seed in range(15):
        env.reset(seed=seed)
        env.step(COLLECT_ACTION)
    env.close()
    store = EpisodeStore(db)
    assert len(store.list_episodes()) == 15
    assert len(store.get_scenario_history("scenario_01")) == 15
    replay = store.replay_episode(1)
    assert len(replay["episode"]["frames"]) == 2
    assert store.get_episode(1)["timeline"][-1]["truncated"]
    assert store.get_episode(1)["timeline"][-1]["state"]["observation"]
    assert (
        store.compare_scenario_attempts("scenario_01", 1, 15)["episodes"][1][
            "episode_number"
        ]
        == 15
    )
    assert store.scenario_report("scenario_01")["episodes"] == 15
    store.close()
    store = EpisodeStore(db)
    assert store.replay_episode(1) == replay
    store.close()


def test_mixed_samples_and_reset_restore_mission():
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=10)
    env.reset(seed=1)
    assert len({s["sample_type"] for s in env.samples if s["target"]}) >= 2
    wrong = next(s for s in env.samples if not s["target"])
    env.set_pose(wrong["x"], wrong["y"])
    assert env.step(5)[4]["samples_collected"] == 0
    correct = next(s for s in env.samples if s["target"])
    env.set_pose(correct["x"], correct["y"])
    env.step(5)
    assert env.step(5)[4]["samples_collected"] == 1
    env.reset(seed=2)
    assert not env.inventory and not any(s["collected"] for s in env.samples)
    env.close()


def test_history_and_inspection_do_not_reveal_hidden_objectives():
    env = PlanetaryExplorationEnv(curriculum=False, history_steps=4)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (132,)
    env.visibility_range = 0.01
    obs = env._observation()
    assert np.all(obs[-33 + 10 : -33 + 12] == 0)
    before = env._history[-1].copy()
    env.map_spec()
    assert np.array_equal(before, env._history[-1])
    env.close()


def test_reward_does_not_use_hidden_target_distance():
    a = PlanetaryExplorationEnv(curriculum=False)
    b = PlanetaryExplorationEnv(curriculum=False)
    a.reset(seed=12)
    b.reset(seed=12)
    # Perturb only the diagnostic oracle; action reward must not consult it.
    b._distance_to_target = lambda: 100000.0
    for action in (0, 3, 0, 4):
        left = a.step(action)
        right = b.step(action)
        assert left[1] == pytest.approx(right[1])
    a.close()
    b.close()


def test_physical_contacts_for_more_than_five_rocks():
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=10)
    env.reset(seed=3)
    assert len(env.rocks) > 5
    cx, cy, hx, _, _ = env.rocks[6]
    env.set_pose(cx - hx - 0.38, cy, 0.0)
    collision = False
    for _ in range(12):
        _, _, done, truncated, info = env.step(0)
        collision |= info["collision"]
        if done or truncated:
            break
    assert collision and env.x < cx + hx
    env.close()


def test_robustness_is_separate_and_never_updates_curriculum(tmp_path):
    from pipelines.planetary_evaluate import evaluate_planetary

    class Policy:
        def __call__(self, obs):
            return 5

    state = tmp_path / "state.json"
    c = CampaignCurriculum(state)
    c._save()
    before = state.read_bytes()
    result = evaluate_planetary(
        Policy(),
        {
            "campaign": True,
            "history_steps": 4,
            "max_steps": 2,
            "curriculum_state": str(state),
        },
        episodes=2,
        scenario_level=1,
        evaluation_kind="robustness",
    )
    assert result["evaluation_kind"] == "robustness" and state.read_bytes() == before
    assert (
        result["cases"][0]["initial_position"] != result["cases"][1]["initial_position"]
    )


def test_real_ppo_updates_and_resumes_optimizer_and_replay(tmp_path, monkeypatch):
    import torch
    from stable_baselines3 import PPO

    import pipelines.train as training

    config = training.read_config(Path("configs/experiments/exp-007-planetary.yaml"))
    config["environment"].update(
        max_steps=4,
        episode_db=str(tmp_path / "episodes.sqlite"),
        curriculum_state=str(tmp_path / "curriculum.json"),
    )
    config["periodic_evaluation_freq"] = 0
    config["hyperparams"].update(n_epochs=1)
    monkeypatch.setattr(training.mlflow, "log_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(training.mlflow, "log_metrics", lambda *args, **kwargs: None)
    first = tmp_path / "first"
    first.mkdir()
    model_path, steps, _ = training._train_ppo(config, 1, 32, first, None, 4, 8)
    assert steps == 32
    model = PPO.load(model_path)
    weights = {k: v.clone() for k, v in model.policy.state_dict().items()}
    monkeypatch.setattr(training, "_download_checkpoint", lambda _: first)
    second = tmp_path / "second"
    second.mkdir()
    model_path, steps, _ = training._train_ppo(config, 1, 32, second, "parent", 4, 8)
    resumed = PPO.load(model_path)
    assert steps == 32 and resumed.num_timesteps == 64
    assert any(
        not torch.equal(v, resumed.policy.state_dict()[k]) for k, v in weights.items()
    )
    assert (second / "optimizer.pt").exists()
    store = EpisodeStore(tmp_path / "episodes.sqlite")
    assert len(store.list_episodes()) >= 16
    store.close()
