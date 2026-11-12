from robot_expedition.envs.beacon_expedition import BASE
from robot_expedition.envs.planetary_exploration import COLLECT_ACTION, PlanetaryExplorationEnv
from robot_expedition.planetary.episode_store import EpisodeStore
from pipelines.planetary_evaluate import evaluate_planetary


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
    assert store.load_scenario(1)["seed"] == 10
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
