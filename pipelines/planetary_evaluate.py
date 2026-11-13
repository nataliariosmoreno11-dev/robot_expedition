"""Evaluación separada para el contrato planetario de 33 observaciones."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from robot_expedition.envs.planetary_exploration import PlanetaryExplorationEnv


def evaluate_planetary(
    policy,
    environment: dict,
    *,
    episodes: int = 24,
    seed_start: int = 900_000,
    max_level: int = 10,
    scenario_level: int | None = None,
    evaluation_kind: str = "mastery",
) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if evaluation_kind not in ("mastery", "robustness"):
        raise ValueError("invalid evaluation kind")
    options = dict(environment)
    options.pop("kind", None)
    persist_evaluation = bool(
        options.get("record_episodes") and options.get("episode_db")
    )
    options.update(
        curriculum=False,
        record_episodes=persist_evaluation,
        update_curriculum=False,
        mode="evaluación",
    )
    env = PlanetaryExplorationEnv(**options)
    cases = []
    started = time.perf_counter()
    for index in range(int(episodes)):
        seed = int(seed_start + index)
        level = scenario_level or 1 + index % max(
            1, min(10 if options.get("campaign", True) else 8, int(max_level))
        )
        if hasattr(policy, "reset"):
            policy.reset(seed)
        observation, info = env.reset(
            seed=seed,
            options={
                "difficulty_level": level,
                "robustness": evaluation_kind == "robustness",
            },
        )
        total = 0.0
        latencies = []
        while True:
            before = time.perf_counter()
            action = int(policy(observation))
            latencies.append((time.perf_counter() - before) * 1000.0)
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        cases.append(
            {
                "seed": seed,
                "scenario_id": info["scenario_id"],
                "difficulty_level": level,
                "difficulty": info["difficulty"],
                "success": bool(info["success"]),
                "end": info["end"],
                "samples_collected": info["samples_collected"],
                "requested_samples": info["requested_samples"],
                "collisions": info["collisions"],
                "steps": env.steps,
                "battery": round(float(info["battery"]), 3),
                "path_length": round(float(info["path_length"]), 3),
                "route_cost": info["route_cost"],
                "initial_position": info["initial_position"],
                "collection_positions": info["collection_positions"],
                "final_position": [info["x"], info["y"]],
                "energy_consumed": info["energy_consumed"],
                "return": round(float(total), 4),
                "latency_ms_mean": round(float(np.mean(latencies)), 4),
            }
        )
    env.close()
    if options.get("campaign", True):
        from robot_expedition.planetary.reference import campaign_reference

        for case in cases:
            ref = campaign_reference(case["scenario_id"])
            case["reference_path_cost"] = ref["reference_path_cost"]
            case["path_efficiency"] = None
            if case["success"] and case["route_cost"] > 0:
                from robot_expedition.planetary.campaign import load_scenario
                from robot_expedition.planetary.reference import comparable_reference

                try:
                    matched = comparable_reference(
                        load_scenario(case["scenario_id"]),
                        case["initial_position"],
                        case["collection_positions"],
                        case["final_position"],
                    )
                except ValueError:
                    matched = None
                if matched:
                    case["reference_path_cost"] = matched["reference_path_cost"]
                    case["path_efficiency"] = (
                        matched["reference_path_cost"] / case["route_cost"]
                    )
    successes = [case for case in cases if case["success"]]

    def mean(key: str, rows=cases):
        return round(float(np.mean([row[key] for row in rows])), 4) if rows else None

    levels = sorted({case["difficulty_level"] for case in cases})
    metrics = {
        "episodes": len(cases),
        "success_rate": round(len(successes) / len(cases), 4),
        "success_rate_by_difficulty": {
            str(level): round(
                sum(
                    case["success"]
                    for case in cases
                    if case["difficulty_level"] == level
                )
                / sum(case["difficulty_level"] == level for case in cases),
                4,
            )
            for level in levels
        },
        "failures_by_cause": dict(
            Counter(case["end"] for case in cases if not case["success"])
        ),
        "return_mean": mean("return"),
        "return_std": round(float(np.std([case["return"] for case in cases])), 4),
        "collisions_per_episode": mean("collisions"),
        "steps_mean": mean("steps"),
        "samples_collected_mean": mean("samples_collected"),
        "mission_completion": round(
            float(
                np.mean(
                    [
                        case["samples_collected"] / case["requested_samples"]
                        for case in cases
                    ]
                )
            ),
            4,
        ),
        "path_length_mean": mean("path_length"),
        "latency_ms_mean": mean("latency_ms_mean"),
    }
    return {
        "suite_id": f"campaign-{evaluation_kind}-v1",
        "evaluation_kind": evaluation_kind,
        "scenario_level": scenario_level,
        "partial": (
            int(episodes) < 50 * (1 if scenario_level else max(1, min(10, int(max_level))))
            if options.get("campaign", True)
            else int(episodes) < 24
        ),
        "environment": environment,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "metrics": metrics,
        "cases": cases,
    }
