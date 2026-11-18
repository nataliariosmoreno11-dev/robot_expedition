"""Validate fixed missions or evaluate a frozen PPO. Planner never trains a policy."""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robot_expedition.planetary.campaign import SCENARIO_IDS, load_scenario
from robot_expedition.planetary.reference import ReferencePlanner, campaign_reference


def physical_validation(sid):
    """Test-only waypoint follower demonstrates a feasible MuJoCo mission.

    Explicitly NOT an RL baseline, training controller, or evidence of learning.
    """
    from robot_expedition.envs.planetary_exploration import (
        COLLECT_ACTION,
        PlanetaryExplorationEnv,
    )

    env = PlanetaryExplorationEnv(curriculum=False, difficulty_level=int(sid[-2:]))
    env.reset(seed=700001)
    planner = ReferencePlanner(env.scenario)
    reference = campaign_reference(sid)
    destinations = [
        next(s for s in env.samples if s["sample_id"] == sample_id)
        for sample_id in reference["target_order"]
    ]
    destinations.append({"x": 1.5, "y": 0.0, "sample_id": "base"})
    done = False
    info = {}
    trajectory = []
    try:
        for target in destinations:
            _distances, previous = planner.search((env.x, env.y))
            u = planner.cell((target["x"], target["y"]))
            start = planner.cell((env.x, env.y))
            cells = [u]
            while u != start:
                u = previous[u]
                cells.append(u)
            route = [planner.point(c) for c in reversed(cells)]
            index = 0
            while not done:
                distance = math.hypot(target["x"] - env.x, target["y"] - env.y)
                if distance < (0.48 if target["sample_id"] != "base" else 0.6):
                    action = COLLECT_ACTION
                else:
                    while (
                        index < len(route) - 1
                        and math.dist((env.x, env.y), route[index]) < 0.25
                    ):
                        index += 1
                    x, y = route[index]
                    error = (math.atan2(y - env.y, x - env.x) - env.theta + math.pi) % (
                        2 * math.pi
                    ) - math.pi
                    action = (3 if error > 0 else 4) if abs(error) > 0.24 else 0
                _, _, terminated, truncated, info = env.step(action)
                trajectory.append([env.x, env.y, env.z])
                done = terminated or truncated
                if target["sample_id"] in env.inventory or done:
                    break
            if done:
                break
        return {
            "scenario_id": sid,
            "success": info.get("success", False),
            "end": info.get("end"),
            "steps": env.steps,
            "collisions": env.collisions,
            "battery": env.battery,
            "samples_collected": len(env.inventory),
            "path_length": env.path_length,
            "trajectory": trajectory,
            "controller": "validation-only waypoint follower, NOT learned policy",
        }
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--physical", action="store_true")
    validate.add_argument("--scenario", choices=SCENARIO_IDS)
    validate.add_argument("--output", type=Path)
    show = sub.add_parser("show")
    show.add_argument("scenario", choices=SCENARIO_IDS)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--vecnormalize", type=Path, required=True)
    evaluate.add_argument("--scenario", choices=SCENARIO_IDS, required=True)
    evaluate.add_argument(
        "--kind", choices=["mastery", "robustness"], default="mastery"
    )
    evaluate.add_argument("--episodes", type=int, default=50)
    evaluate.add_argument("--history-steps", type=int, default=4)
    evaluate.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "show":
        result = load_scenario(args.scenario).to_dict()
    elif args.command == "validate":
        result = []
        for sid in [args.scenario] if args.scenario else SCENARIO_IDS:
            row = dict(campaign_reference(sid))
            if args.physical:
                row["physical_validation"] = physical_validation(sid)
            result.append(row)
    else:
        from pipelines.planetary_evaluate import evaluate_planetary
        from pipelines.policies import PPOPolicy

        policy = PPOPolicy(args.model, args.vecnormalize)
        result = evaluate_planetary(
            policy,
            {
                "kind": "planetary",
                "campaign": True,
                "history_steps": args.history_steps,
                "record_episodes": True,
                "episode_db": str(ROOT / "logs/campaign_evaluations.sqlite"),
                "policy_version": str(args.model.resolve()),
            },
            episodes=args.episodes,
            scenario_level=int(args.scenario[-2:]),
            evaluation_kind=args.kind,
            seed_start=920000 if args.kind == "robustness" else 900000,
        )
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(args.output)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if (
        args.command == "validate"
        and args.physical
        and not all(r["physical_validation"]["success"] for r in result)
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
