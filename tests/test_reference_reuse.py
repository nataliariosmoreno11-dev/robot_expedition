from robot_expedition.planetary.campaign import load_scenario
from robot_expedition.planetary.reference import ReferencePlanner, comparable_reference


def test_shared_reference_grid_preserves_cost_and_route():
    scenario = load_scenario("scenario_01")
    planner = ReferencePlanner(scenario)
    start = scenario.start[:2]
    samples = [(s.x, s.y) for s in scenario.samples if s.target]
    expected = comparable_reference(scenario, start, samples, start)
    assert expected is not None
    for _ in range(2):
        assert comparable_reference(scenario, start, samples, start, planner=planner) == expected


def test_evaluation_reuses_only_reference_math_not_episodes(monkeypatch):
    from pipelines.planetary_evaluate import evaluate_planetary
    from robot_expedition.planetary import reference
    from robot_expedition.envs.planetary_exploration import PlanetaryExplorationEnv
    real = reference.comparable_reference
    calls = []
    def tracked(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(reference, "comparable_reference", tracked)
    # Level one's nearby sample can be collected and the rover is still in base.
    # Use the existing successful deterministic action sequence to avoid a fake env.
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=1, record_episodes=False)
    obs, _ = env.reset(seed=1)
    actions = []
    import math
    target = next(sample for sample in env.samples if sample["target"])
    for _ in range(200):
        tx, ty = (1.5, 0.0) if env.inventory else (target["x"], target["y"])
        angle = (math.atan2(ty-env.y, tx-env.x)-env.theta+math.pi)%(2*math.pi)-math.pi
        action = (3 if angle > 0 else 4) if abs(angle) > .24 else 0
        if not env.inventory and math.hypot(tx-env.x, ty-env.y) < .48:
            action = 5
        actions.append(action)
        obs, _, terminal, truncated, info = env.step(action)
        if terminal or truncated:
            break
    env.close()
    assert info["success"]
    class ReplayActions:
        def reset(self, seed):
            self.index = 0
        def __call__(self, observation):
            action = actions[self.index]
            self.index += 1
            return action
    report = evaluate_planetary(ReplayActions(), {"campaign":True, "record_episodes":False},
                                episodes=2, scenario_level=1, seed_start=1234)
    assert report["metrics"]["episodes"] == 2
    assert report["metrics"]["success_rate"] == 1
    assert len({case["seed"] for case in report["cases"]}) == 2
    assert len(calls) == 1
    assert report["cases"][0]["reference_path_cost"] == report["cases"][1]["reference_path_cost"]
