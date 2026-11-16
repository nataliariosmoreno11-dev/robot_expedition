import pytest

from robot_expedition.envs.planetary_exploration import PlanetaryExplorationEnv


def test_small_learning_rate_round_trips_through_worker_json(tmp_path):
    import json
    from pathlib import Path
    from pipelines.train import read_config
    config = read_config(Path('configs/experiments/exp-009-retention.yaml'))
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    loaded = read_config(path)
    assert isinstance(loaded['hyperparams']['learning_rate'], float)
    assert loaded['hyperparams']['learning_rate'] == pytest.approx(0.00005)


def test_collecting_in_place_resets_training_without_rewarding_a_success():
    env = PlanetaryExplorationEnv(curriculum=False, training_no_progress_limit=12)
    try:
        env.reset(seed=1)
        for _ in range(20):
            _, _, terminated, truncated, info = env.step(5)
            if terminated or truncated:
                break
        assert truncated and not terminated
        assert info["end"] == "sin_progreso" and not info["success"]
        assert info["no_progress_steps"] == 12
        assert env.steps <= 14
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["manual", "evaluación"])
def test_training_early_reset_does_not_make_evaluation_easier(mode):
    env = PlanetaryExplorationEnv(curriculum=False, training_no_progress_limit=4, mode=mode)
    try:
        env.reset(seed=1)
        for _ in range(12):
            _, _, terminated, truncated, info = env.step(5)
            assert not terminated and not truncated
        assert info["no_progress_steps"] >= 4
    finally:
        env.close()


def test_new_coverage_resets_stagnation_clock():
    env = PlanetaryExplorationEnv(curriculum=False, training_no_progress_limit=12)
    try:
        env.reset(seed=1)
        for _ in range(7):
            env.step(5)
        assert env._no_progress_steps >= 5
        cells = len(env.explored_cells)
        for _ in range(4):
            _, _, terminated, truncated, info = env.step(0)
            assert not terminated and not truncated
        assert len(env.explored_cells) > cells
        assert info["no_progress_steps"] < 5
    finally:
        env.close()


def test_retention_guard_restores_weights_normalization_and_keeps_step_count(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from pipelines import train

    parent = tmp_path / 'parent'
    parent.mkdir()
    train.save_json(parent / 'curriculum_state.json', {'mastered':[1]})
    model_dir = tmp_path / 'candidate'
    model_dir.mkdir()
    env = VecNormalize(DummyVecEnv([lambda: PlanetaryExplorationEnv(curriculum=False, history_steps=4)]))
    try:
        model = PPO('MlpPolicy', env, n_steps=4, batch_size=4)
        model._last_obs = env.reset()
        config = {'environment': {'history_steps':4}, 'retention_guard_freq':4096}
        guard = train.RetentionGuardCallback(config, model_dir, parent)
        guard.init_callback(model)
        guard.on_training_start({}, {})
        weights = {k:v.clone() for k,v in model.policy.state_dict().items()}
        mean = env.obs_rms.mean.copy()
        with torch.no_grad():
            next(model.policy.parameters()).add_(1)
        env.obs_rms.mean += 20
        model.num_timesteps = 4096
        monkeypatch.setattr(train, 'evaluate_planetary', lambda *a, **kw: {'metrics':{'success_rate':0.0,'collisions_per_episode':0}})
        monkeypatch.setattr(train.mlflow, 'log_metric', lambda *a, **kw: None)
        guard.verify()
        assert model.num_timesteps == 4096
        assert all(torch.equal(weights[k],v) for k,v in model.policy.state_dict().items())
        assert np.array_equal(env.obs_rms.mean, mean)
        assert (model_dir/'rejected-4096'/'ppo_model.zip').exists()
        assert guard.safe_step == 0
        curriculum = tmp_path / 'curriculum.json'
        config['environment']['curriculum_state'] = str(curriculum)
        train.save_json(curriculum, {'mastered': [1, 2]})
        guard.protect_mastery([1, 2])
        train.save_json(curriculum, {'mastered': [1, 2, 3]})
        model.num_timesteps = 8192
        checked = []
        def reject(*args, **kwargs):
            checked.append(kwargs['scenario_level'])
            return {'metrics': {'success_rate': 0, 'collisions_per_episode': 0}}
        monkeypatch.setattr(train, 'evaluate_planetary', reject)
        guard.verify()
        import json
        assert checked == [1, 2]
        assert json.loads(curriculum.read_text())['mastered'] == [1, 2]
    finally:
        env.close()


@pytest.mark.parametrize('threshold,expected', [(0.85, 8), (0.0, 50)])
def test_evaluation_only_rejects_when_passing_is_impossible(threshold, expected):
    from pipelines.planetary_evaluate import evaluate_planetary
    result = evaluate_planetary(lambda obs: 5, {'max_steps': 1, 'history_steps': 4},
        episodes=50, scenario_level=1, abort_below_success_rate=threshold)
    assert len(result['cases']) == expected
    assert result['requested_episodes'] == 50
    assert result['early_rejection'] == (expected < 50)
    assert result['partial'] == (expected < 50)


@pytest.mark.parametrize("mode,level", [("evaluaci?n", 6), ("manual", 6), ("entrenamiento", 3)])
def test_navigation_hint_does_not_change_evaluation_or_rehearsal(mode, level):
    options = dict(curriculum=False, difficulty_level=level, mode=mode)
    plain = PlanetaryExplorationEnv(**options)
    shaped = PlanetaryExplorationEnv(**options, training_navigation_shaping=8)
    try:
        import numpy as np
        plain.reset(seed=21)
        shaped.reset(seed=21)
        for action in [0, 1, 0, 2, 5, 6]:
            a, b = plain.step(action), shaped.step(action)
            assert np.array_equal(a[0], b[0])
            assert a[1:4] == b[1:4]
            assert b[4]["navigation_shaping"] == 0
    finally:
        plain.close()
        shaped.close()


def test_navigation_potential_uses_only_visible_targets(monkeypatch):
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=6,
                                  training_navigation_shaping=8)
    try:
        env.reset(seed=21)
        monkeypatch.setattr(env, "_nearest_visible_sample", lambda: (None, 1.0))
        assert env._navigation_potential() == 0
        monkeypatch.setattr(env, "_nearest_visible_sample", lambda: ({"target": False}, 1.0))
        assert env._navigation_potential() == 0
        monkeypatch.setattr(env, "_nearest_visible_sample", lambda: ({"target": True}, 1.0))
        assert 0 < env._navigation_potential() <= 8
    finally:
        env.close()


def test_navigation_shaping_telescopes_and_terminal_clears_potential(monkeypatch):
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=6,
                                  training_navigation_shaping=8)
    try:
        env.reset(seed=21)
        potentials = iter([2.0, 3.0, 3.0, 2.0, 2.0])
        monkeypatch.setattr(env, "_navigation_potential", lambda: next(potentials))
        first = env.step(5)[4]["navigation_shaping"]
        second = env.step(5)[4]["navigation_shaping"]
        gamma = env.training_shaping_gamma
        assert first + gamma * second == pytest.approx((gamma ** 2 - 1) * 2)
        env.battery = 0.000001
        _, _, terminated, _, info = env.step(5)
        assert terminated
        assert info["navigation_shaping"] == -2
    finally:
        env.close()


def test_navigation_experiment_uses_matching_discount():
    from pathlib import Path
    from pipelines.train import read_config
    config = read_config(Path("configs/experiments/exp-012-observable-navigation.yaml"))
    assert config["hyperparams"]["gamma"] == config["environment"]["training_shaping_gamma"]
