import hashlib

from pipelines.train import _manifest, read_config, sha256_file, sha256_json


def test_experiment_config_and_hash_are_reproducible(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "experiment_id: exp-test\n"
        "hypothesis: baseline\n"
        "level_id: n2\n"
        "algorithm: qlearning\n"
        "budget_timesteps: 8\n"
        "seeds: [1]\n",
        encoding="utf-8",
    )

    expected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert sha256_file(config_path) == expected_hash
    assert sha256_file(config_path) == expected_hash
    assert read_config(config_path)["environment"]["max_steps"] == 400
    assert sha256_json({"seed": 1}) == sha256_json({"seed": 1})


def test_manifest_contains_effective_budget_and_failure_diagnosis(tmp_path):
    config = {
        "experiment_id": "exp-test",
        "level_id": "n2",
        "algorithm": "qlearning",
    }

    manifest = _manifest(
        run_id="run-test",
        config=config,
        config_path=tmp_path / "experiment.yaml",
        config_hash="a" * 64,
        source_config_hash="c" * 64,
        seed=1,
        git_commit="b" * 40,
        origin="https://example.test/robot-expedition.git",
        budget=8,
        effective_budget=8,
        parent_run_id=None,
        status="failed",
        diagnosis="RuntimeError: test; sin checkpoint disponible",
        model_uri=None,
        model_hash=None,
        normalization_uri=None,
        metrics={},
        checkpoint_uri=None,
    )

    assert manifest["config_sha256"] == "a" * 64
    assert manifest["source_config_sha256"] == "c" * 64
    assert manifest["effective_budget_timesteps"] == 8
    assert manifest["status"] == "failed"
    assert "sin checkpoint disponible" in manifest["diagnosis"]


def _experiment(tmp_path, level_id, extra=""):
    path = tmp_path / f"exp-{level_id}.yaml"
    path.write_text(
        "experiment_id: exp-test\n"
        "hypothesis: niveles\n"
        f"level_id: {level_id}\n"
        "algorithm: qlearning\n"
        "budget_timesteps: 8\n"
        "seeds: [1]\n" + extra,
        encoding="utf-8",
    )
    return path


def test_training_uses_the_level_config(tmp_path):
    from pipelines.train import build_env, evaluation_suite

    for level_id, noise, beacons in (("n1", 0.02, 1), ("n2", 0.02, 2), ("n3", 0.08, 2)):
        config = read_config(_experiment(tmp_path, level_id))
        env = build_env(config)
        assert env.ray_noise == noise and env.beacon_count == beacons
        assert evaluation_suite(level_id) == f"suite-{level_id}-v1"
    # environment solo sobrescribe si se pide explicitamente
    config = read_config(_experiment(tmp_path, "n3", "environment:\n  ray_noise: 0.0\n"))
    assert build_env(config).ray_noise == 0.0


def test_unknown_level_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        read_config(_experiment(tmp_path, "n9"))


def test_final_evaluation_uses_the_fixed_suite(tmp_path):
    import numpy as np

    from pipelines.train import _evaluate
    from robot_expedition.qlearning import STATE_SHAPE

    model = tmp_path / "qlearning.npz"
    np.savez_compressed(model, q=np.zeros((*STATE_SHAPE, 5), dtype=np.float32),
                        visits=np.zeros(STATE_SHAPE), epsilon=0.0)
    config = read_config(_experiment(tmp_path, "n1"))
    result = _evaluate(config, model, None, episodes=2)
    assert result["suite_id"] == "suite-n1-v1" and result["partial"]
    assert result["policy"]["model_sha256"] and len(result["cases"]) == 2


def test_planetary_config_contract_and_curriculum_checkpoint_resume(tmp_path):
    import json

    from pipelines.train import _restore_curriculum_checkpoint, build_env

    curriculum = tmp_path / "state" / "curriculum.json"
    config_path = tmp_path / "planetary.yaml"
    config_path.write_text(
        "experiment_id: exp-planetary-test\n"
        "hypothesis: curriculum\n"
        "level_id: n2\n"
        "algorithm: ppo\n"
        "budget_timesteps: 8\n"
        "seeds: [1]\n"
        "environment:\n"
        "  kind: planetary\n"
        "  difficulty_level: 1\n"
        "  curriculum: true\n"
        f"  curriculum_state: {curriculum.as_posix()}\n"
        "  record_episodes: false\n",
        encoding="utf-8",
    )
    config = read_config(config_path)
    env = build_env(config)
    observation, _ = env.reset(seed=7)
    assert observation.shape == (33,)
    assert env.action_space.n == 7
    env.close()

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payload = {
        "schema_version": 1,
        "current_level": 4,
        "total_episodes": 30,
        "outcomes": [True, False],
    }
    (checkpoint / "curriculum_state.json").write_text(json.dumps(payload), encoding="utf-8")
    assert _restore_curriculum_checkpoint(checkpoint, config) == curriculum
    assert json.loads(curriculum.read_text())["current_level"] == 4


def test_periodic_planetary_evaluation_uses_a_frozen_holdout(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import numpy as np

    import pipelines.train as train

    calls = []
    logged = []

    class FakeNormalizer:
        def normalize_obs(self, observation):
            return observation + 2.0

    class FakeModel:
        num_timesteps = 9

        def get_env(self):
            return FakeNormalizer()

        def predict(self, observation, deterministic):
            assert deterministic and np.all(observation == 2.0)
            return np.asarray(3), None

    def fake_evaluate(policy, environment, **options):
        assert policy(np.zeros(33, dtype=np.float32)) == 3
        calls.append((environment, options))
        return {
            "suite_id": "placeholder",
            "metrics": {
                "success_rate": 0.5,
                "return_mean": 10.0,
                "collisions_per_episode": 1.0,
                "samples_collected_mean": 2.0,
                "mission_completion": 0.75,
            },
        }

    monkeypatch.setattr(train, "evaluate_planetary", fake_evaluate)
    monkeypatch.setattr(
        train,
        "mlflow",
        SimpleNamespace(log_metric=lambda *args, **kwargs: logged.append((args, kwargs))),
    )
    callback = train.PeriodicPlanetaryEvaluationCallback(
        {"environment": {"kind": "planetary", "campaign": False}, "evaluation_max_level": 8},
        tmp_path,
        frequency=10,
        episodes=4,
        seed_start=910_000,
    )
    model = FakeModel()
    callback.model = model
    assert callback._on_step() and not calls
    model.num_timesteps = 10
    assert callback._on_step() and len(calls) == 1
    assert calls[0][1] == {"episodes": 4, "seed_start": 910_000, "max_level": 8, "scenario_level": None}
    payload = json.loads((tmp_path / "periodic-evaluation-10.json").read_text())
    assert payload["suite_id"] == "planetary-periodic-holdout-v1"
    assert payload["training_step"] == 10
    assert len(logged) == 5
