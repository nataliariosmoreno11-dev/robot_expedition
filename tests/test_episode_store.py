import sqlite3

from robot_expedition.envs.beacon_expedition import BASE
from robot_expedition.envs.planetary_exploration import COLLECT_ACTION, PlanetaryExplorationEnv
from robot_expedition.planetary.episode_store import EpisodeStore
from pipelines.planetary_evaluate import evaluate_planetary


def _record_direct(store, number, *, level, success, terrain):
    scenario = {
        "scenario_id": f"scenario-{number}", "planet": {"planet_id": "mars"},
        "difficulty_level": level, "difficulty_name": f"L{level}",
        "target_type": "basalto", "requested_samples": 1,
        "rocks": [[number, 0, 1, 1, 1]] * terrain[0],
        "dunes": [[0, 0, 1, 1]] * terrain[1],
        "craters": [[0, 0, -1, 1]] * terrain[2], "traction_zones": [],
        "wind_speed_m_s": level * 2, "dust_opacity": level / 10,
    }
    writer = store.start_episode(
        run_id="long-training", policy_version="checkpoint", mode="entrenamiento",
        seed=number, scenario=scenario,
        map_spec={"environment_version": "test/1", "scenario": number},
    )
    writer.record_step(
        state={"frame": [number], "mission": {"collected": int(success)}},
        action=number % 6, action_name="prueba", reward=float(number),
        cumulative_reward=float(number), event="fin", collision=not success,
        terminated=True, truncated=False,
    )
    writer.finish(
        end="exito" if success else "colision", success=success,
        cumulative_reward=float(number), duration_s=1.0, steps=1,
        samples_collected=int(success), collisions=int(not success), path_length=float(number),
    )
    return writer.episode_number


def finish_episode(env):
    sample = next(item for item in env.samples if item["sample_type"] == env.target_type)
    env.set_pose(sample["x"], sample["y"])
    env.step(COLLECT_ACTION)
    env.set_pose(BASE[0], BASE[1])
    env.step(COLLECT_ACTION)


def test_every_attempt_is_incremental_and_replay_uses_saved_frames(tmp_path):
    database = tmp_path / "episodes.sqlite"
    env = PlanetaryExplorationEnv(
        curriculum=False,
        difficulty_level=1,
        episode_db=database,
        record_episodes=True,
        run_id="test-run",
        policy_version="checkpoint-a",
    )
    env.reset(seed=10)
    finish_episode(env)
    env.reset(seed=11)
    finish_episode(env)
    env.close()

    store = EpisodeStore(database)
    episodes = list(reversed(store.list_episodes()))
    assert [episode["episode_number"] for episode in episodes] == [1, 2]
    assert all(episode["success"] for episode in episodes)
    first = store.get_episode(1)
    assert first["policy_version"] == "checkpoint-a"
    assert len(first["timeline"]) == 3
    assert first["timeline"][-1]["event"] == "misión completada"
    replay_before = store.replay_episode(1)

    # Cambiar la política declarada no vuelve a ejecutar nada: los cuadros son inmutables.
    env = PlanetaryExplorationEnv(
        curriculum=False,
        difficulty_level=1,
        episode_db=database,
        record_episodes=True,
        run_id="test-run",
        policy_version="checkpoint-b",
    )
    env.reset(seed=12)
    finish_episode(env)
    env.close()
    replay_after = store.replay_episode(1)
    assert replay_after == replay_before
    assert store.get_episode(1)["seed"] == 10
    assert store.load_scenario(1)["seed"] == 41001
    assert store.load_scenario(1) == store.load_scenario(2)
    comparison = store.compare_episodes(1, 2, 3)
    assert len(comparison["episodes"]) == 3
    assert comparison["shortest_success"] == 1
    assert all("scenario" not in episode and "map" not in episode for episode in comparison["episodes"])
    store.close()


def test_failed_or_interrupted_attempt_is_also_saved(tmp_path):
    database = tmp_path / "episodes.sqlite"
    env = PlanetaryExplorationEnv(
        curriculum=False,
        episode_db=database,
        record_episodes=True,
    )
    env.reset(seed=30)
    env.step(0)
    env.reset(seed=31)
    env.close()
    store = EpisodeStore(database)
    episodes = list(reversed(store.list_episodes()))
    assert episodes[0]["end"] == "reinicio"
    assert episodes[1]["end"] == "reinicio"
    assert all(episode["success"] is False for episode in episodes)
    store.close()


def test_multi_episode_evaluation_is_separate_and_replayable(tmp_path):
    database = tmp_path / "evaluation.sqlite"

    class StationaryPolicy:
        def reset(self, seed):
            self.seed = seed

        def __call__(self, observation):
            return COLLECT_ACTION

    result = evaluate_planetary(
        StationaryPolicy(),
        {
            "kind": "planetary",
            "difficulty_level": 1,
            "max_steps": 2,
            "episode_db": str(database),
            "record_episodes": True,
            "run_id": "evaluation-test",
            "policy_version": "checkpoint-test",
        },
        episodes=3,
        max_level=2,
    )
    assert result["metrics"]["episodes"] == 3
    store = EpisodeStore(database)
    attempts = store.list_episodes()
    assert len(attempts) == 3
    assert all(item["mode"] == "evaluación" for item in attempts)
    replay = store.replay_episode(attempts[-1]["episode_number"])
    assert len(replay["episode"]["frames"]) == 3
    assert replay["episode"]["end"] == "tiempo"
    store.close()


def test_retains_only_ten_diverse_replays_but_summarizes_every_episode(tmp_path):
    store = EpisodeStore(tmp_path / "limited.sqlite", max_experiments=10)
    numbers = []
    for index in range(25):
        numbers.append(_record_direct(
            store, index + 1, level=1 + index % 8, success=index % 3 != 0,
            terrain=(index % 6, (index // 2) % 5, (index // 3) % 4),
        ))
    status = store.storage_status()
    assert status == {
        "max_experiments": 10, "retained_experiments": 10,
        "protected_experiments": 0, "historical_untouched": 0,
        "summarized_episodes": 25,
    }
    retained = store.list_episodes(limit=100)
    assert len(retained) == 10
    assert len({item["difficulty_level"] for item in retained}) >= 5
    assert {item["success"] for item in retained} == {True, False}
    metrics = store.metrics()
    assert metrics["episodes"] == 25
    assert metrics["samples_collected"] == 16
    assert len(metrics["by_difficulty"]) == 8
    assert metrics["evolution"]
    for item in retained:
        replay = store.replay_episode(item["episode_number"])
        assert replay["map"]["scenario"] == item["seed"]
        assert replay["episode"]["frames"] == [[item["seed"]]]
    store.close()


def test_protected_replay_survives_replacement_and_replays_exactly(tmp_path):
    store = EpisodeStore(tmp_path / "protected.sqlite", max_experiments=3)
    first = _record_direct(store, 1, level=1, success=False, terrain=(1, 0, 0))
    expected = store.replay_episode(first)
    protected = store.protect_episode(first, True)
    assert protected["protected"] is True
    for index in range(2, 9):
        _record_direct(store, index, level=index, success=index % 2 == 0, terrain=(index, index % 3, index % 2))
    assert store.storage_status()["retained_experiments"] == 3
    assert store.replay_episode(first) == expected
    assert store.get_episode(first)["selection_reason"] == "protegido manualmente"
    store.protect_episode(first, False)
    assert store.get_episode(first)["protected"] is False
    store.close()


def test_historical_episode_is_never_pruned(tmp_path):
    database = tmp_path / "historical.sqlite"
    store = EpisodeStore(database, max_experiments=10)
    historical = _record_direct(store, 1, level=1, success=True, terrain=(1, 0, 0))
    store.close()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE episodes SET retention_managed=0, selection_reason='histórico' WHERE episode_number=?",
            (historical,),
        )
    store = EpisodeStore(database, max_experiments=10)
    for index in range(2, 17):
        _record_direct(store, index, level=1 + index % 8, success=index % 2 == 0,
                       terrain=(index % 5, index % 4, index % 3))
    assert store.storage_status()["historical_untouched"] == 1
    assert store.storage_status()["retained_experiments"] == 10
    assert store.replay_episode(historical)["episode"]["frames"] == [[1]]
    assert len(store.list_episodes(limit=100)) == 11
    store.close()
