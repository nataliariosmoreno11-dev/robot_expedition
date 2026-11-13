import json

import numpy as np
import pytest

from robot_expedition.envs.beacon_expedition import BASE
from robot_expedition.envs.planetary_exploration import (
    COLLECT_ACTION,
    PlanetaryExplorationEnv,
)
from robot_expedition.planetary.curriculum import CurriculumManager
from robot_expedition.planetary.scenario import ScenarioGenerator


def test_procedural_scenarios_are_reproducible_and_varied():
    generator = ScenarioGenerator()
    first = generator.generate(12345, 6)
    again = generator.generate(12345, 6)
    other = generator.generate(12346, 6)
    assert first.to_dict() == again.to_dict()
    assert first.scenario_id != other.scenario_id
    assert not np.allclose(first.terrain_heights, other.terrain_heights)


def test_all_curriculum_levels_generate_solvable_missions():
    generator = ScenarioGenerator()
    for level in range(1, 9):
        scenario = generator.generate(8000 + level, level)
        valid, reasons = generator.validate(scenario)
        assert valid, reasons
        assert len(scenario.target_samples) >= scenario.requested_samples
        assert scenario.capacity >= scenario.requested_samples
        assert 1 <= scenario.requested_samples <= 4


def test_collection_is_explicit_not_repeatable_and_wrong_type_does_not_complete():
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=2)
    env.reset(seed=91)
    wrong = next(sample for sample in env.samples if sample["sample_type"] != env.target_type)
    env.set_pose(wrong["x"], wrong["y"])
    battery_before = env.battery
    _, reward, terminated, _, info = env.step(COLLECT_ACTION)
    assert reward <= -10 and not terminated
    assert env.battery <= battery_before - 6
    assert wrong["collected"] is True
    assert info["samples_collected"] == 0 and not info["success"]

    correct = next(sample for sample in env.samples if sample["sample_type"] == env.target_type)
    env.set_pose(correct["x"], correct["y"])
    _, reward, _, _, info = env.step(COLLECT_ACTION)
    assert reward > 0 and info["samples_collected"] == 1
    _, duplicate_reward, _, _, duplicate = env.step(COLLECT_ACTION)
    assert duplicate_reward < 0 and duplicate["samples_collected"] == 1
    env.close()


def test_sandstorm_changes_visibility_wind_and_energy_deterministically():
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=8)
    env.reset(seed=808)
    base_visibility = env.visibility_range
    intensities = []
    visibilities = []
    for _ in range(80):
        env._update_storm()
        intensities.append(env.storm_intensity)
        visibilities.append(env.visibility_range)
        env.steps += 1
    assert max(intensities) > 0.8
    assert min(visibilities) < base_visibility
    env.steps = intensities.index(max(intensities))
    env._update_storm()
    storm_force = np.linalg.norm(env._wind_force())
    env.storm_intensity = 0.0
    calm_force = np.linalg.norm(env._wind_force())
    assert storm_force > calm_force
    env.close()


def test_planet_and_terrain_parameters_reach_mujoco_and_observation():
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=5)
    observation, _ = env.reset(seed=52)
    assert observation.shape == (33,) and env.observation_space.contains(observation)
    assert env.model.opt.gravity[2] == pytest.approx(-3.71)
    assert env.model.geom_friction[env._terrain_id, 0] == pytest.approx(env.scenario.traction)
    assert np.ptp(env.heights) > 0.1
    assert observation[28] == pytest.approx(env.visibility_range / 5.0)
    env.close()


def test_hidden_samples_do_not_leak_their_direction_and_traction_zones_are_physical():
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=3)
    env.reset(seed=303)
    env.visibility_range = 0.1
    observation = env._observation()
    assert observation[9] == pytest.approx(5.0)
    assert observation[10] == pytest.approx(0.0)
    assert observation[11] == pytest.approx(0.0)
    assert sum(observation[18:21]) == pytest.approx(1.0)  # conoce el tipo pedido, no la ubicación

    x, y, _, _, coefficient = env.scenario.traction_zones[0]
    env.set_pose(x, y)
    observation = env._observation()
    assert observation[27] == pytest.approx(coefficient)
    env.step(COLLECT_ACTION)
    assert env.model.geom_friction[env._terrain_id, 0] == pytest.approx(coefficient)
    env.close()


def test_rock_collision_blocks_forward_motion():
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=4)
    env.reset(seed=404)
    cx, cy, hx, _, _ = env.rocks[0]
    env.set_pose(cx - hx - 0.38, cy, 0.0)
    collided = False
    for _ in range(8):
        _, _, terminated, truncated, info = env.step(0)
        collided |= bool(info["collision"])
        if terminated or truncated:
            break
    assert collided
    assert env.x < cx + hx
    env.close()


def test_curriculum_advances_by_success_and_keeps_state(tmp_path):
    state = tmp_path / "curriculum.json"
    manager = CurriculumManager(
        state,
        initial_level=1,
        window_size=4,
        minimum_episodes=3,
        success_threshold=2 / 3,
    )
    assert not manager.record_result(True, evaluated_level=1)
    assert not manager.record_result(False, evaluated_level=1)
    assert manager.record_result(True, evaluated_level=1)
    assert manager.current_level == 2
    restored = CurriculumManager(state)
    assert restored.current_level == 2
    assert json.loads(state.read_text())["schema_version"] == 1


def test_reading_map_does_not_change_simulation():
    plain = PlanetaryExplorationEnv(curriculum=False, difficulty_level=3)
    inspected = PlanetaryExplorationEnv(curriculum=False, difficulty_level=3)
    obs_a, _ = plain.reset(seed=77)
    obs_b, _ = inspected.reset(seed=77)
    inspected.map_spec()
    assert np.allclose(obs_a, obs_b)
    for action in (0, 1, 0, 2):
        result_a = plain.step(action)
        inspected.map_spec()
        result_b = inspected.step(action)
        assert np.allclose(result_a[0], result_b[0])
        assert result_a[1:4] == pytest.approx(result_b[1:4])
    plain.close()
    inspected.close()


def test_complete_mission_requires_samples_and_return_to_base():
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=1)
    env.reset(seed=12)
    sample = next(item for item in env.samples if item["sample_type"] == env.target_type)
    env.set_pose(sample["x"], sample["y"])
    _, _, terminated, _, _ = env.step(COLLECT_ACTION)
    assert not terminated
    env.set_pose(BASE[0], BASE[1])
    _, _, terminated, _, info = env.step(COLLECT_ACTION)
    assert terminated and info["success"] and info["end"] == "exito"
    env.close()
