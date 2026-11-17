"""Generate small training-only reset states from a physically successful demonstration.

Privileged planner data is used ONLY to construct practice starts, never by the
learned policy or the independent evaluation. No demonstration counts as mastery.
"""
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
from robot_expedition.envs.planetary_exploration import PlanetaryExplorationEnv
from robot_expedition.planetary.reference import ReferencePlanner, campaign_reference
from robot_expedition.planetary.campaign import catalog_hash


def capture(env):
    return dict(qpos=env.data.qpos.tolist(), qvel=env.data.qvel.tolist(),
                time=float(env.data.time), battery=env.battery, steps=env.steps,
                inventory=list(env.inventory), collection_positions=env.collection_positions.copy(),
                path_length=env.path_length, route_cost=env.route_cost,
                explored_cells=list(env.explored_cells),
                history=[o.tolist() for o in env._history])


def build(output, trajectory_output=None, level=6, seed=700001):
    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=level, history_steps=4,
                                  mode="demostraci?n", record_episodes=False)
    states = []
    observations, actions, stages = [], [], []
    try:
        observation, _ = env.reset(seed=seed)
        planner = ReferencePlanner(env.scenario)
        order = campaign_reference(f"scenario_{level:02d}")["target_order"]
        targets = [next(s for s in env.samples if s["sample_id"] == sid) for sid in order]
        targets.append(dict(x=1.5, y=0.0, sample_id="base"))
        done = False
        for target in targets:
            _, previous = planner.search((env.x, env.y))
            start = planner.cell((env.x, env.y))
            u = planner.cell((target["x"], target["y"]))
            cells = [u]
            while u != start:
                u = previous[u]
                cells.append(u)
            route = [planner.point(c) for c in reversed(cells)]
            index = 0
            while not done:
                if math.hypot(target["x"]-env.x, target["y"]-env.y) < (0.48 if target["sample_id"] != "base" else 0.6):
                    action = 5
                else:
                    while index < len(route)-1 and math.dist((env.x,env.y),route[index]) < 0.25:
                        index += 1
                    x,y = route[index]
                    error = (math.atan2(y-env.y,x-env.x)-env.theta+math.pi)%(2*math.pi)-math.pi
                    action = (3 if error > 0 else 4) if abs(error) > 0.24 else 0
                observations.append(observation.copy())
                actions.append(action)
                stages.append(len(env.inventory))
                observation,_,terminated,truncated,info = env.step(action)
                done = terminated or truncated
                if target["sample_id"] in env.inventory:
                    if len(env.inventory) >= 2 and not done:
                        states.append(capture(env))
                    break
            if done:
                break
        if not info.get("success") or env.collisions > 2 or len(states) != max(0, env.requested_samples - 1):
            raise RuntimeError(f"Demonstration rejected: {info}")
        result = dict(schema="training-stage-bank/v1", scenario_id=f"scenario_{level:02d}",
                      catalog_hash=catalog_hash(), history_steps=4,
                      purpose="privileged demonstration reset states; not learned-policy evaluation",
                      demonstration=dict(success=True, steps=env.steps, collisions=env.collisions,
                                         battery=env.battery), states=states)
        if trajectory_output is not None:
            import numpy as np
            trajectory_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trajectory_output, observations=np.asarray(observations, dtype=np.float32),
                                actions=np.asarray(actions, dtype=np.int64), stages=np.asarray(stages, dtype=np.int64), catalog_hash=catalog_hash())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, separators=(",", ":"))+"\n")
        print(json.dumps(dict(path=str(output), bytes=output.stat().st_size,
                             demonstration=result["demonstration"])))
    finally:
        env.close()

if __name__ == "__main__":
    build(ROOT / "configs/training/level6-stages.json")
