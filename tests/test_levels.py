import math

import numpy as np

from robot_expedition import BeaconExpeditionEnv
from robot_expedition.envs.beacon_expedition import BASE, BEACONS, BLOCK, load_scenario_set


def place(env, x, y, theta=0.0):
    env.set_pose(x, y, theta, speed=0.6)


def test_all_levels_reset_and_step_with_same_contract():
    observations = []
    for level in ("n1", "n2", "n3"):
        env = BeaconExpeditionEnv(level=level)
        obs, _ = env.reset(seed=4)
        next_obs, _, _, _, _ = env.step(0)
        observations.extend((obs, next_obs))
        assert obs.shape == (20,)
        assert env.action_space.n == 5
    assert all(item.shape == observations[0].shape for item in observations)


def test_n1_success_after_b1_and_base():
    env = BeaconExpeditionEnv(level="n1", ray_noise=0.0)
    env.reset(seed=1)
    place(env, BEACONS[0][0] - 0.7, BEACONS[0][1])
    _, _, terminated, _, info = env.step(0)
    assert info["stage"] == 3 and not terminated        # etapa 3 = volver a la base en todo nivel
    assert env.target() == BASE[:2]
    place(env, BASE[0] + BASE[2] + 0.2, BASE[1], math.pi)
    _, _, terminated, _, info = env.step(0)
    assert terminated and info["success"] and info["end"] == "exito"


def test_n3_activates_block_and_dead_ray_degrades_sensor():
    env = BeaconExpeditionEnv(level="n3")
    env.reset(seed=2)
    place(env, BEACONS[0][0] - 0.7, BEACONS[0][1])
    env.step(0)
    assert env.block_active and BLOCK[1:] in env.obstacles()
    rays = env.rays()
    assert rays[4] == 5.0
    assert env.ray_noise == 0.08


def test_level_rocks_match_physics_after_reset_and_jitter():
    # Antes reset() volvia a cargar las 5 rocas en la lista aunque N1 solo tiene 2 en MuJoCo.
    for jitter in (0.0, 0.5):
        env = BeaconExpeditionEnv(level="n1", rock_jitter=jitter)
        env.reset(seed=3)
        assert len(env.rocks) == len(env._rock_ids) == 2
        assert [r[0] for r in env.map_spec()["rocks"]] == ["R1", "R2"]


def test_varied_scenarios_cover_goals_obstacles_and_difficulties():
    scenario_set = load_scenario_set("varied-v1")
    env = BeaconExpeditionEnv(level="n2", scenario_set="varied-v1", ray_noise=0.0)
    seen = []
    for seed, expected in enumerate(scenario_set["scenarios"]):
        _, info = env.reset(seed=seed)
        seen.append((info["scenario_id"], info["difficulty"]))
        assert info["scenario_id"] == expected["scenario_id"]
        assert env.beacons == expected["beacons"]
        assert len(env.rocks) == expected["active_rocks"]
        assert env._route_exists()
        assert env.map_spec()["beacons"] == expected["beacons"]
    assert {difficulty for _, difficulty in seen} == {"easy", "medium", "hard"}
    assert len({tuple(scenario["beacons"][0]) for scenario in scenario_set["scenarios"]}) == 6


def test_varied_scenario_is_reproducible_and_can_be_forced():
    first = BeaconExpeditionEnv(level="n2", scenario_set="varied-v1")
    second = BeaconExpeditionEnv(level="n2", scenario_set="varied-v1")
    obs_a, info_a = first.reset(seed=17)
    obs_b, info_b = second.reset(seed=17)
    assert info_a["scenario_id"] == info_b["scenario_id"]
    assert first.beacons == second.beacons
    assert np.allclose(first.rocks, second.rocks)
    assert np.allclose(obs_a, obs_b)

    _, forced = first.reset(seed=17, options={"scenario_id": "hard-blocked-return"})
    assert forced["difficulty"] == "hard"
    assert first.episode_blocking
    first.close()
    second.close()


def test_hard_scenario_activates_block_without_closing_the_route():
    env = BeaconExpeditionEnv(level="n2", scenario_set="varied-v1", ray_noise=0.0)
    env.reset(seed=5, options={"scenario_id": "hard-blocked-return"})
    bx, by = env.beacons[0]
    place(env, bx - 0.7, by)
    env.step(0)
    assert env.block_active
    assert env._route_exists()
    env.close()
