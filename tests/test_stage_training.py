import json
from pathlib import Path
import numpy as np
import pytest
from robot_expedition.envs.planetary_exploration import PlanetaryExplorationEnv

BANK = "configs/training/level6-stages.json"


def make(**kwargs):
    return PlanetaryExplorationEnv(curriculum=False, difficulty_level=6, history_steps=4,
                                  training_stage_bank=BANK, training_stage_probability=1, **kwargs)


def test_stage_states_preserve_spent_battery_time_and_inventory():
    bank = json.loads(Path(BANK).read_text())
    env = make()
    seen = set()
    try:
        for seed in range(12):
            obs, info = env.reset(seed=seed)
            stage = info["training_stage"]
            seen.add(stage)
            saved = next(s for s in bank["states"] if len(s["inventory"]) == stage)
            assert env.battery == saved["battery"] < env.scenario.battery_budget
            assert env.steps == saved["steps"] > 0
            assert np.allclose(env.data.qpos, saved["qpos"])
            assert set(env.inventory) == env.collected_ids
            assert sum(s["collected"] for s in env.samples) == stage
            assert obs.shape == (132,) and np.isfinite(obs).all()
            assert not info["success"]
            env.step(0)
        assert seen == {2, 3, 4}
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["evaluaci?n", "manual"])
def test_evaluation_ignores_even_missing_stage_bank(mode):
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=6, history_steps=4,
        training_stage_bank="missing.json", training_stage_probability=1, mode=mode)
    try:
        _, info = env.reset(seed=1)
        assert env.training_stage == 0 and not env.inventory
        assert env.steps == 0 and env.battery == env.scenario.battery_budget
    finally:
        env.close()


def test_rehearsal_and_full_starts_do_not_apply_stage_state():
    env = make()
    try:
        env.reset(seed=1, options={"difficulty_level": 3})
        assert env.training_stage == 0 and not env.inventory
        env.training_stage_probability = 0
        env.reset(seed=1)
        assert env.training_stage == 0 and not env.inventory
    finally:
        env.close()


def test_stage_episode_cannot_update_curriculum(monkeypatch):
    env = make()
    try:
        env.curriculum_enabled = True
        env.reset(seed=1, options={"difficulty_level": 6})
        monkeypatch.setattr(env.curriculum, "record_result", lambda *a, **k: pytest.fail("stage counted as full mission"))
        env.battery = 0.000001
        _,_,terminated,_,info = env.step(5)
        assert terminated and "curriculum_advanced" not in info
    finally:
        env.close()


def test_incompatible_stage_bank_fails_closed(tmp_path):
    bank = json.loads(Path(BANK).read_text());bank["catalog_hash"] = "stale"
    path = tmp_path / "bank.json";path.write_text(json.dumps(bank))
    env = make();env.training_stage_bank = str(path)
    try:
        with pytest.raises(ValueError, match="incompatible"):
            env.reset(seed=1)
    finally:
        env.close()
