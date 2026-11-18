"""Evaluación separada para el contrato planetario de 33 observaciones."""

from __future__ import annotations

import time
import os
from pathlib import Path
from collections import Counter

import numpy as np

from robot_expedition.envs.planetary_exploration import PlanetaryExplorationEnv


def evaluate_planetary(
    policy,
    environment: dict,
    *,
    episodes: int = 24,
    seed_start: int = 900_000,
    max_level: int = 8,
    scenario_level: int | None = None,
    evaluation_kind: str = "mastery",
    abort_below_success_rate: float | None = None,
    on_progress=None,
) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if evaluation_kind not in ("mastery", "robustness"):
        raise ValueError("invalid evaluation kind")
    if abort_below_success_rate is not None and (scenario_level is None or not 0 <= abort_below_success_rate <= 1):
        raise ValueError("early rejection requires one scenario and a threshold in [0, 1]")
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
            1, min(8, int(max_level))
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
            stop_file = os.environ.get("EXPEDITION_STOP_FILE")
            if stop_file and Path(stop_file).exists():
                env.close()
                raise KeyboardInterrupt("Evaluación detenida; se conservan los pesos entrenados")
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
                "explored_cells": info["explored_cells"],
                "stagnation_steps": info.get("stagnation_steps", 0),
                "stagnation_events": info.get("stagnation_events", 0),
                "stagnation_reasons": info.get("stagnation_reasons", {}),
                "returned_to_base": bool(info["success"]),
                "return": round(float(total), 4),
                "latency_ms_mean": round(float(np.mean(latencies)), 4),
            }
        )
        if on_progress is not None:
            on_progress({"completed": len(cases), "total": int(episodes),
                         "successes": sum(case["success"] for case in cases), "level": level})
        # Reject only when even success in ALL remaining cases cannot pass.
        # Never approve early; promotion still requires the complete suite.
        if abort_below_success_rate is not None:
            possible = sum(case["success"] for case in cases) + int(episodes) - len(cases)
            if possible / int(episodes) < abort_below_success_rate:
                break
    env.close()
    if options.get("campaign", True):
        from robot_expedition.planetary.reference import campaign_reference

        reference_planners = {}
        matched_references = {}
        for case in cases:
            ref = campaign_reference(case["scenario_id"])
            case["reference_path_cost"] = ref["reference_path_cost"]
            case["path_efficiency"] = None
            if case["success"] and case["route_cost"] > 0:
                from robot_expedition.planetary.campaign import load_scenario
                from robot_expedition.planetary.reference import comparable_reference, ReferencePlanner

                try:
                    scenario_id = case["scenario_id"]
                    if scenario_id not in reference_planners:
                        reference_planners[scenario_id] = ReferencePlanner(load_scenario(scenario_id))
                    planner = reference_planners[scenario_id]
                    key = (scenario_id, tuple(case["initial_position"]),
                           tuple(tuple(point) for point in case["collection_positions"]),
                           tuple(case["final_position"]))
                    if key not in matched_references:
                        matched_references[key] = comparable_reference(
                            planner.scenario, case["initial_position"],
                            case["collection_positions"], case["final_position"], planner=planner)
                    matched = matched_references[key]
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
        "return_to_base_rate": round(float(np.mean([case["returned_to_base"] for case in cases])), 4),
        "energy_consumed_mean": mean("energy_consumed"),
        "explored_cells_mean": mean("explored_cells"),
        "stagnation_steps_mean": mean("stagnation_steps"),
        "stagnation_events_mean": mean("stagnation_events"),
        "episodes_with_stagnation_rate": round(
            float(np.mean([case["stagnation_steps"] > 0 for case in cases])), 4
        ),
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
        "requested_episodes": int(episodes),
        "early_rejection": len(cases) < int(episodes),
        "partial": (
            len(cases) < 50 * (1 if scenario_level else max(1, min(8, int(max_level))))
            if options.get("campaign", True)
            else len(cases) < 24
        ),
        "environment": environment,
        "wall_time_s": round(time.perf_counter() - started, 3),
        "metrics": metrics,
        "cases": cases,
    }
