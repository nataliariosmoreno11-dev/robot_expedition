"""Diagnóstico reproducible en mapas procedurales reservados.

No entrena ni modifica checkpoints. Compara decisiones aprendidas con líneas
base y guarda únicamente métricas agregadas y casos JSON ligeros.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.planetary_evaluate import evaluate_planetary
from pipelines.policies import CampaignNumpyPPOPolicy, PPOPolicy
from robot_expedition.planetary.qlearning import PlanetaryQLearningAgent, discretize_planetary


RUNS = (
    "f7c104a304f25b458721258b540a337e",
    "d0ecdae6689f527b835c1e473e893d0c",
    "d0c438b791c156688c71cf58c69fceb5",
    "bdbc0d04fffe5c37a5a5ac5a05013889",
    "646fbc70700f5ab8b2363524540fddca",
    "156aaf20543e523eb1fa4594547d956e",
    "6109346a105e5f3e82134dc9cafbbcb9",
    "4bcfd8f641df50b9b5460d76b8c56482",
)


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, _observation) -> int:
        return int(self.rng.integers(7))


class FrozenGreedyQPolicy:
    """Q original sin guía de reglas; estados desconocidos usan acción esperar."""

    def __init__(self, checkpoint: Path):
        self.agent = PlanetaryQLearningAgent(epsilon=0.0).load(checkpoint)

    def __call__(self, observation) -> int:
        state = discretize_planetary(observation)
        values = self.agent.q.get(state)
        return int(np.argmax(values)) if values is not None and np.any(values) else 6


def environment() -> dict:
    return {
        "planet": "mars",
        "campaign": False,
        "history_steps": 4,
        "record_episodes": False,
        "training_reward_limit": 0.0,
        "training_reward_floor": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-level", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=970_000)
    parser.add_argument("--output", default="docs/evidencia/diagnostico-aprendizaje-procedural.json")
    parser.add_argument("--q-checkpoint", default=os.environ.get("PLANETARY_Q_CHECKPOINT", "models/planetary_qlearning.npz"))
    parser.add_argument("--improved-ppo")
    parser.add_argument("--improved-normalization")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes debe ser positivo")
    if not 1 <= args.max_level <= 8:
        parser.error("--max-level debe estar entre 1 y 8")
    if bool(args.improved_ppo) != bool(args.improved_normalization):
        parser.error("--improved-ppo y --improved-normalization deben usarse juntos")

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    report = {
        "schema": "planetary-learning-diagnostic/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "semillas procedurales no usadas por la campaña fija",
        "episodes_per_level": args.episodes,
        "seed_start": args.seed_start,
        "policies": {},
    }

    random_levels = {}
    q_levels = {}
    checkpoint = Path(args.q_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    q_policy = None
    if checkpoint.is_file():
        try:
            q_policy = FrozenGreedyQPolicy(checkpoint)
        except (ValueError, KeyError) as exc:
            report["qlearning_checkpoint"] = {
                "path": str(checkpoint), "compatible": False, "error": str(exc)
            }
    for level in range(1, args.max_level + 1):
        seed = args.seed_start + level * 100
        random_levels[str(level)] = evaluate_planetary(
            RandomPolicy(seed), environment(), episodes=args.episodes,
            seed_start=seed, scenario_level=level,
        )
        if q_policy is not None:
            q_levels[str(level)] = evaluate_planetary(
                q_policy, environment(), episodes=args.episodes,
                seed_start=seed, scenario_level=level,
            )
    report["policies"]["random"] = random_levels
    if q_levels:
        report["policies"]["qlearning_unassisted"] = q_levels
        report["qlearning_checkpoint"] = {"path": str(checkpoint), "compatible": True}

    ppo_levels = {}
    for level, run_id in enumerate(RUNS[: args.max_level], 1):
        adapter = ROOT / "catalog" / "campaign_adapters" / f"{run_id}.npz"
        if not adapter.is_file():
            continue
        ppo_levels[str(level)] = evaluate_planetary(
            CampaignNumpyPPOPolicy(adapter), environment(), episodes=args.episodes,
            seed_start=args.seed_start + level * 100, scenario_level=level,
        )
    report["policies"]["ppo_tomas_procedural"] = ppo_levels

    if args.improved_ppo:
        model_path = Path(args.improved_ppo)
        normalization_path = Path(args.improved_normalization)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not normalization_path.is_absolute():
            normalization_path = ROOT / normalization_path
        improved_policy = PPOPolicy(model_path, normalization_path)
        improved_levels = {}
        for level in range(1, args.max_level + 1):
            improved_levels[str(level)] = evaluate_planetary(
                improved_policy,
                environment(),
                episodes=args.episodes,
                seed_start=args.seed_start + level * 100,
                scenario_level=level,
            )
        report["policies"]["ppo_improved_procedural"] = improved_levels

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        policy: {
            level: result["metrics"]
            for level, result in levels.items()
        }
        for policy, levels in report["policies"].items()
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nEvidencia: {output}")


if __name__ == "__main__":
    main()
